#ifndef __GDS_BACKEND_H
#define __GDS_BACKEND_H

#include <nixl.h>
#include <nixl_types.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <fcntl.h>
#include "gds_utils.h"

class nixlGdsMetadata : public nixlBackendMD {
public:
    gdsFileHandle  handle;
    gdsMemBuf      buf;
    nixl_mem_t     type;

    nixlGdsMetadata() : nixlBackendMD(true) { }
    ~nixlGdsMetadata() { }
};

class nixlGdsIOBatch :  public nixlBackendReqH {
private:
    unsigned int	    max_reqs;

    CUfileBatchHandle_t batch_handle;
    CUfileIOEvents_t    *io_batch_events;
    CUfileIOParams_t    *io_batch_params;
    CUfileError_t       init_err;
    nixl_xfer_state_t   current_status;
    unsigned int	entries_completed;
    unsigned int        batch_size;

public:
    nixlGdsIOBatch(int size);
    ~nixlGdsIOBatch();

    nixl_status_t	addToBatch(CUfileHandle_t fh,  void *buffer,
				   size_t size, size_t file_offset,
				   size_t ptr_offset, CUfileOpcode_t type);
    nixl_status_t       submitBatch(int flags);
    nixl_xfer_state_t   checkStatus();
};


class nixlGdsEngine : nixlBackendEngine {
    gdsUtil		        *gds_utils;
    std::map<int, gdsFileHandle> gds_file_map;

public:
    nixlGdsEngine(const nixlGdsInitParams* init_params);
    ~nixlGdsEngine();

    // File operations - target is the distributed FS
    // So no requirements to connect to target.
    // Just treat it locally.
    bool	 supportsNotif () const {
        return false;
    }
    bool	 supportsRemote  () const {
        return false;
    }
    bool	 supportsLocal   () const {
        return true;
    }
    bool	 supportsProgTh  () const {
        return false;
    }

protected:
    nixl_status_t connect(const std::string &remote_agent)
    {
        return NIXL_SUCCESS;
    }

    nixl_status_t disconnect(const std::string &remote_agent)
    {
        return NIXL_SUCCESS;
    }

    nixl_status_t loadLocalMD (nixlBackendMD* input,
                               nixlBackendMD* &output) {
        output = input;

        return NIXL_SUCCESS;
    }

    nixl_status_t unloadMD (nixlBackendMD* input) {
        return NIXL_SUCCESS;
    }
    nixl_status_t registerMem(const nixlStringDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD* &out);
    void deregisterMem (nixlBackendMD *meta);

    nixl_xfer_state_t postXfer (const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const nixl_xfer_op_t &op,
                                const std::string &remote_agent,
                                const std::string &notif_msg,
                                nixlBackendReqH* &handle);

    nixl_xfer_state_t checkXfer (nixlBackendReqH* handle);
    void releaseReqH(nixlBackendReqH* handle);
};
#endif
