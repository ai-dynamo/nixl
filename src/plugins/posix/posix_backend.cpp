/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <liburing.h>
#include "posix_backend.h"

#define NIXL_POSIX_RING_SIZE 1024
#define CQES_MAX_BATCH_SIZE 16

io_uring& nixlPosixEngine::uringGetInstance() {
    return this->uring;
}

int nixlPosixEngine::uringGetInitStatus() {
    if (this->uring_init_status == 1)
        this->uring_init_status = io_uring_init_params(NIXL_POSIX_RING_SIZE,
                                                       &this->uring,
                                                       &this->uring_params);

    return this->uring_init_status;
}

nixl_status_t nixlPosixEngine::checkCompletions() {
    struct io_uring_cqe *cqes[CQES_MAX_BATCH_SIZE];
    int num_ret_cqes;
    do {
        num_ret_cqes = io_uring_peek_batch_cqe(&this->uring, cqes, CQES_MAX_BATCH_SIZE);
        for (int i = 0; i < num_ret_cqes; i++) {
            if (cqes[i]->res < 0) {
                return NIXL_ERR_BACKEND;
            }
            nixlPosixBackendReqH *posix_handle = (nixlPosixBackendReqH *)cqes[i]->user_data;
            posix_handle->num_completed++;
        }
    } while (num_ret_cqes == CQES_MAX_BATCH_SIZE);
    return NIXL_SUCCESS;
}

nixlPosixEngine::nixlPosixEngine() {
    this->initErr = false;
    if (uringGetInitStatus() != 0) 
        this->initErr = true;
}

nixlPosixEngine::~nixlPosixEngine() {
}

//TODO: ask Vishwarath if this needs implementation. Dont see a need for registering files or memory for posix backend.
nixl_status_t nixlPosixEngine::registerMem(const nixlBlobDesc &mem,
                                           const nixl_mem_t &nixl_mem,
                                           nixlBackendMD* &out) {
    nixlPosixMetadata *md  = new nixlPosixMetadata();
    nixl_status_t status = NIXL_SUCCESS;
    switch (nixl_mem) {
        case FILE_SEG:
            this->posix_utils->fillHandle(mem.devId, mem.len,mem.metaInfo, md->handle);
            this->md_map[mem.devId] = md;
            out = (nixlBackendMD*) md;
            break;
        case DRAM_SEG:
            status = NIXL_ERR_NOT_SUPPORTED;
            break;
        default:
            status = NIXL_ERR_NOT_SUPPORTED;
    }
    return status;
}

nixl_status_t nixlPosixEngine::deregisterMem(nixlBackendMD *meta) {
    nixl_status_t status = NIXL_SUCCESS;
    nixlPosixMetadata *md = (nixlPosixMetadata *)meta;
    if (md->type == FILE_SEG) {
        auto it = this->md_map.find(md->handle.fd);
        if (it != this->md_map.end())
            // TODO: do i need to delete the md in the map
            // or is it for the user to do so?
            this->md_map.erase(it);
        else
            status = NIXL_ERR_NOT_FOUND;
    }
    return status;
}

nixl_status_t nixlPosixEngine::prepXfer(const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) {
    // TODO MAKE SURE IO_PREP_READ doesnt make read
    nixl_status_t status = NIXL_SUCCESS;
    // TOOD: Im going to assume that theres no need for the metadata map for posix backend
    // auto it = this->md_map.find(local[i].devId);
    // if (it == this->md_map.end()) {
    //     status = NIXL_ERR_NOT_FOUND;
    //     goto err_exit;
    // }
    nixlPosixBackendReqH  *posix_handle = new nixlPosixBackendReqH();
    struct io_uring_sqe *entry;
    unsigned int work_id;

    if (local.descCount() != remote.descCount()) {
        std::cerr <<"Error in count\n";
        status = NIXL_ERR_INVALID_PARAM;
        goto err_exit;
    }    

    // After verification this is a correct implementation, clean it up
    switch (operation) {
        case NIXL_READ: // Remote -> Local (file -> DRAM)
            for (int i = 0; i < local.descCount(); i++) {
                entry = io_uring_get_sqe(&this->uring);
                if (!entry) {
                    std::cerr <<"Error in getting sqe\n";
                    status = NIXL_ERR_BACKEND;
                    goto err_exit;
                }
                // add checks for validity of the local and remote fields
                io_uring_prep_read(entry, local.fd, remote[i].addr, local[i].len, local[i].addr);
                io_uring_sqe_set_data(entry, (void *)posix_handle);
            }
            break;
        case NIXL_WRITE: // Local -> Remote (DRAM -> file)
            for (int i = 0; i < local.descCount(); i++) {
                entry = io_uring_get_sqe(&this->uring);
                if (!entry) {
                    std::cerr <<"Error in getting sqe\n";
                    status = NIXL_ERR_BACKEND;
                    goto err_exit;
                }
                // add checks for validity of the local and remote fields
                io_uring_prep_write(entry, remote[i].fd, local[i].addr, remote[i].len, remote[i].addr);
                io_uring_sqe_set_data(entry, (void *)posix_handle);
            }
            break;
        default:
            std::cerr <<"Error in or operation selection\n";
            status = NIXL_ERR_INVALID_PARAM;
            goto err_exit;
    }

    handle->is_prepped = true;
    handle = posix_handle;
    return status;

err_exit:
    delete posix_handle;
    return status;
}

nixl_status_t nixlPosixEngine::postXfer(const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) {
    if (!handle->is_prepped)
        this->prepXfer(opeation, local, remote, remote_agent, handle, opt_args);
    
    io_uring_submit(&this->uring);
    return NIXL_SUCCESS;
}

nixl_status_t nixlPosixEngine::checkXfer(nixlBackendReqH* handle) {
    nixl_status_t status = this->registerCompletions();
    if (status != NIXL_SUCCESS)
        return status;
    // TODO: make more efficient (could make work_id a pointer to a struct that points to request handle and the index of the request
    // then we can just increase the count of different requests completed and also know the status of each request)
    nixlPosixBackendReqH *posix_handle = (nixlPosixBackendReqH *)handle;
    if (posix_handle->num_completed != posix_handle->num_entries)
        status = NIXL_IN_PROG;
        
    return status;
}

nixl_status_t nixlPosixEngine::releaseReqH(nixlBackendReqH* handle) {
    nixlPosixBackendReqH *posix_handle = (nixlPosixBackendReqH *)handle;
    delete posix_handle;
    return NIXL_SUCCESS;
} 