#include "nixl_adapter.h"
#include "utils/serdes/serdes.h"
#include <cassert>

nixlAdapter::nixlAdapter() {}

nixlAdapter::~nixlAdapter() {}

int nixlAdapter::initialize(const char *role, const char *backend)
{
    nixlAgentConfig config(true);
    agent = new nixlAgent(role, config);

    if (std::string(backend) == "UCX")
        ucx = agent->createBackend(&params);
    else
        return -1;

    return 0;
}

uintptr_t nixlAdapter::allocateManagedBuffer(size_t length)
{
    char* buffer = (char *)calloc(1, length);
    if (!buffer) {
        std::cerr << "Error allocating buffer\n";
        return -1;
    }

    buffer_alloc_list.insert(buffer);
    return (uintptr_t)buffer;
}

int nixlAdapter::freeManagedBuffer(uintptr_t buffer)
{
    char *buffer_addr = (char *)buffer;

    buffer_alloc_list.erase(buffer_addr);
    free(buffer_addr);
    return 0;
}

int nixlAdapter::writeBytesToBuffer(uintptr_t dst_address,
                                    char *src_ptr, size_t length)
{
    memcpy((void *)dst_address, (void *)src_ptr, length);
    return 0;
}

pybind11::bytes nixlAdapter::readBytesFromBuffer(uintptr_t src,
                                                 size_t length)
{
    return pybind11::bytes(
           static_cast<const char *>(reinterpret_cast<void *>(src)), length);
}

pybind11::bytes nixlAdapter::registerMemory(uintptr_t buffer_addr,
                                            size_t len,
                                            const char *type_name)
{
    nixlStringDesc      desc_buf;
    nixl_mem_t          type;
    std::string         str_desc;
    nixlSerDes          sd_obj;

    desc_buf.addr  = buffer_addr;
    desc_buf.len   = len;
    desc_buf.devId = 0;

    if (std::string(type_name) == "DRAM") {
        type = DRAM_SEG;
    } else if (std::string(type_name) == "VRAM") {
        type = VRAM_SEG;
    }

    nixl_reg_dlist_t descs(type);
    descs.addDesc(desc_buf);

    agent->registerMem(descs, ucx);
    descs.serialize(&sd_obj);
    str_desc = sd_obj.exportStr();

    return pybind11::bytes(str_desc);
}

int nixlAdapter::deregisterMemory(pybind11::bytes desc_list_blob)
{
    nixlSerDes  serdes_obj;
    std::string desc_list_str = static_cast<std::string>(desc_list_blob);

    serdes_obj.importStr(desc_list_str);
    nixl_reg_dlist_t rdescs(&serdes_obj);

    return agent->deregisterMem(rdescs, ucx);
}

int nixlAdapter::transferAndSync(pybind11::bytes src_desc_bytes,
                                 pybind11::bytes target_desc_bytes,
                                 const std::string target_name,
                                 const std::string op)
{
    nixl_xfer_op_t              operation;
    int                         ret = 0;
    nixlSerDes                  src_sd, tgt_sd;
    nixlXferReqH                *treq;
    int                         status = 0;

    std::string src_str_desc = static_cast<std::string>(src_desc_bytes);
    std::string target_str_desc = static_cast<std::string>(target_desc_bytes);

    src_sd.importStr(src_str_desc);
    tgt_sd.importStr(target_str_desc);

    nixl_xfer_dlist_t src_descs(&src_sd);
    nixl_xfer_dlist_t target_descs(&tgt_sd);

    if (op == "READ")
        operation = NIXL_READ;
    else
        operation = NIXL_WRITE;

    ret = agent->createXferReq(src_descs, target_descs, target_name, "",
                   operation, treq);
    if (ret != 0) {
        std::cerr << "Error creating transfer request for adapter\n";
    }
    ret = agent->postXferReq(treq);
    if (ret == NIXL_XFER_ERR) {
        std::cerr << "Error Posting request for adapter\n";
    return NIXL_XFER_ERR;
    }

    while (status != NIXL_XFER_DONE) {
           status = agent->getXferStatus(treq);
           assert(status != NIXL_XFER_ERR);
    }
    agent->invalidateXferReq(treq);

    return ret;
}

pybind11::bytes nixlAdapter::getNixlMD()
{
    return pybind11::bytes(agent->getLocalMD());
}

int nixlAdapter::loadNixlMD(pybind11::bytes remote_md_bytes)
{
    std::string remoteMD = static_cast<std::string>(remote_md_bytes);
    if (agent->loadRemoteMD(remoteMD) == "")
        return -1;
    else
        return 0;
}

int nixlAdapter::remoteProgress()
{
    //now protected, and should be unnecessary with progress thread
    //return ucx->progress();
    return 0;
}

PYBIND11_MODULE(nixlAdapter, n) {
    pybind11::class_<nixlAdapter>(n, "nixlAdapter")
                .def(pybind11::init<>())
                .def("initialize", &nixlAdapter::initialize)
                .def("allocateManagedBuffer", &nixlAdapter::allocateManagedBuffer)
                .def("freeManagedBuffer", &nixlAdapter::freeManagedBuffer)
                .def("transferAndSync", &nixlAdapter::transferAndSync)
                .def("writeBytesToBuffer", &nixlAdapter::writeBytesToBuffer)
                .def("readBytesFromBuffer", &nixlAdapter::readBytesFromBuffer)
                .def("registerMemory", &nixlAdapter::registerMemory)
                .def("deregisterMemory", &nixlAdapter::deregisterMemory)
                .def("getNixlMD", &nixlAdapter::getNixlMD)
                .def("loadNixlMD", &nixlAdapter::loadNixlMD)
                .def("remoteProgress", &nixlAdapter::remoteProgress);
}
