#include <cassert>
#include <iostream>
#include "cufile.h"
#include "gds_backend.h"


nixlGdsIOBatch::nixlGdsIOBatch(int size)
{
    max_reqs			= size;
    io_batch_events		= new CUfileIOEvents_t[size];
    io_batch_params		= new CUfileIOParams_t[size];
    current_status		= NIXL_XFER_INIT;
    entries_completed		= 0;
    batch_size			= 0;

    init_err = cuFileBatchIOSetUp(&batch_handle, size);
    if (init_err.err != 0) {
        std::cerr << "Error in creating the batch\n";
    }
}

nixlGdsIOBatch::~nixlGdsIOBatch()
{
    if (current_status != NIXL_XFER_PROC &&
            current_status != NIXL_XFER_ERR) {
        delete io_batch_events;
        delete io_batch_params;
    } else {
        std::cerr<<"Attempting to delete a batch before completion\n";
    }
}

nixl_status_t nixlGdsIOBatch::addToBatch(CUfileHandle_t fh, void *buffer,
					 size_t size, size_t file_offset,
					 size_t ptr_offset,
					 CUfileOpcode_t type)
{
    CUfileIOParams_t    *params	= nullptr;

    if (batch_size >= max_reqs)
        return NIXL_ERR_BACKEND;

    params				= &io_batch_params[batch_size];
    params->mode			= CUFILE_BATCH;
    params->fh				= fh;
    params->u.batch.devPtr_base		= buffer;
    params->u.batch.file_offset		= file_offset;
    params->u.batch.devPtr_offset	= ptr_offset;
    params->u.batch.size		= size;
    params->opcode			= type;
    params->cookie			= params;
    batch_size++;

    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsIOBatch::submitBatch(int flags)
{
    CUfileError_t	err;

    err = cuFileBatchIOSubmit(batch_handle, batch_size,
                              io_batch_params, flags);
    if (err.err != 0) {
        std::cerr << "Error in setting up Batch\n" << std::endl;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_xfer_state_t nixlGdsIOBatch::checkStatus()
{
    CUfileError_t		errBatch;
    unsigned int		nr = max_reqs;

    errBatch = cuFileBatchIOGetStatus(batch_handle, 0, &nr,
                                      io_batch_events, NULL);
    if (errBatch.err != 0) {
        std::cerr << "Error in IO Batch Get Status" << std::endl;
        current_status = NIXL_XFER_ERR;
    }

    if (nr < (unsigned int)max_reqs)
        current_status = NIXL_XFER_PROC;
    else if (nr > max_reqs)
        current_status = NIXL_XFER_ERR;
    else
        current_status = NIXL_XFER_DONE;

    entries_completed += nr;
    return current_status;
}

nixlGdsEngine::nixlGdsEngine (const nixlGdsInitParams* init_params)
    : nixlBackendEngine ((const nixlBackendInitParams *) init_params)
{
    gds_utils = new gdsUtil();

    this->initErr = false;
    if (gds_utils->openGdsDriver() == NIXL_ERR_BACKEND)
        this->initErr = true;
}


nixl_status_t nixlGdsEngine::registerMem (const nixlStringDesc &mem,
        const nixl_mem_t &nixl_mem,
        nixlBackendMD* &out)
{
    nixl_status_t status;
    nixlGdsMetadata *md  = new nixlGdsMetadata();

    if (nixl_mem == FILE_SEG) {
        status = gds_utils->registerFileHandle(mem.devId, mem.len,
                                               mem.metaInfo, md->handle);
        if (NIXL_SUCCESS != status) {
	     delete md;
	     return status;
	}
        md->handle.fd = mem.devId;
        md->handle.size = mem.len;
        md->handle.metadata = mem.metaInfo;
        md->type = nixl_mem;
        gds_file_map[mem.devId] = md->handle;

    } else if (nixl_mem == VRAM_SEG) {
	status = gds_utils->registerBufHandle((void *)mem.addr, mem.len, 0);
        if (NIXL_SUCCESS != status) {
	     delete md;
	     return status;
	}
        md->buf.base = (void *)mem.addr;
        md->buf.size = mem.len;
        md->type = nixl_mem;
    } else {
        // Unsupported in the backend.
	return NIXL_ERR_BACKEND;
    }
    out = (nixlBackendMD*) md;
    // set value for gds handle here.
    return status;
}

void nixlGdsEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlGdsMetadata *md = (nixlGdsMetadata *)meta;
    if (md->type == FILE_SEG) {
        gds_utils->deregisterFileHandle(md->handle);
    } else {
        gds_utils->deregisterBufHandle(md->buf.base);
    }
    return;
}

nixl_xfer_state_t nixlGdsEngine::postXfer (const nixl_meta_dlist_t &local,
					   const nixl_meta_dlist_t &remote,
					   const nixl_xfer_op_t &operation,
					   const std::string &remote_agent,
					   const std::string &notif_msg,
					   nixlBackendReqH* &handle)
{
    gdsFileHandle	fh;
    void		*addr;
    size_t		size;
    size_t		offset;
    nixl_xfer_state_t	ret = NIXL_XFER_INIT;
    int			rc = 0;
    size_t		buf_cnt  = local.descCount();
    size_t		file_cnt = remote.descCount();

    if ((buf_cnt != file_cnt) ||
            ((operation != NIXL_READ) && (operation != NIXL_WRITE)))  {
        std::cerr <<"Error in count or operation selection\n";
        return NIXL_XFER_ERR;
    }

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        std::cerr <<"Only support I/O between VRAM to file type\n";
        return NIXL_XFER_ERR;
    }

    if ((remote.getType() == DRAM_SEG) || (local.getType() == DRAM_SEG)) {
        std::cerr <<"Backend does not support DRAM to/from files\n";
        return NIXL_XFER_ERR;
    }

    nixlGdsIOBatch *batch_ios = new nixlGdsIOBatch(buf_cnt);

    for (size_t i = 0; i < buf_cnt; i++) {
        if (local.getType() == VRAM_SEG) {
            addr = (void *) local[i].addr;
            size = local[i].len;
            offset = (size_t) remote[i].addr;

            auto it = gds_file_map.find(remote[i].devId);
            if (it != gds_file_map.end()) {
                fh = it->second;
            } else {
                ret = NIXL_XFER_ERR;
                goto exit;
            }
        } else {
            addr		= (void *) remote[i].addr;
            size		= remote[i].len;
            offset		= (size_t) local[i].addr;

            auto it = gds_file_map.find(local[i].devId);
            if (it != gds_file_map.end()) {
                fh = it->second;
            } else {
                ret = NIXL_XFER_ERR;
                goto exit;
            }
        }

        CUfileOpcode_t op = (operation == NIXL_READ) ?
                            CUFILE_READ : CUFILE_WRITE;


        rc = batch_ios->addToBatch(fh.cu_fhandle, addr, size, offset, 0, op);
        if (rc != 0) {
            ret = NIXL_XFER_ERR;
            goto exit;
        }
    }
    rc = batch_ios->submitBatch(0);
    if (rc != 0) {
        ret = NIXL_XFER_ERR;
        goto exit;
    }
    handle = (nixlBackendReqH *)(batch_ios);

exit:
    return ret;
}

nixl_xfer_state_t nixlGdsEngine::checkXfer(nixlBackendReqH* handle)
{
    nixlGdsIOBatch *batch_ios = (nixlGdsIOBatch *)handle;

    return batch_ios->checkStatus();
}


void nixlGdsEngine::releaseReqH(nixlBackendReqH* handle)
{

    nixlGdsIOBatch *batch_ios = (nixlGdsIOBatch *)handle;
    delete batch_ios;

    return;
}

nixlGdsEngine::~nixlGdsEngine() {
    cuFileDriverClose();
    delete gds_utils;
}
