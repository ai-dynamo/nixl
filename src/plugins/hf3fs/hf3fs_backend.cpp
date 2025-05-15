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
#include <cassert>
#include <iostream>
#include <cctype>
#include <atomic>
#include <errno.h>
#include "hf3fs_backend.h"
#include "common/str_tools.h"
#include "common/status.h"
#include "common/nixl_log.h"


nixlHf3fsEngine::nixlHf3fsEngine (const nixlBackendInitParams* init_params)
    : nixlBackendEngine (init_params)
{
    hf3fs_utils = new hf3fsUtil();

    this->initErr = false;
    if (hf3fs_utils->openHf3fsDriver() == NIXL_ERR_BACKEND) {
        std::cerr << "Error opening HF3FS driver" << std::endl;
        this->initErr = true;
    }

    // Get mount point from parameters if available
    std::string mount_point = "/mnt/3fs/"; // default
    if (init_params && init_params->customParams && init_params->customParams->count("mount_point") > 0) {
        mount_point = init_params->customParams->at("mount_point");
    }

    char mount_point_cstr[256];
    auto ret = hf3fs_extract_mount_point(mount_point_cstr, 256, mount_point.c_str());
    if (ret < 0) {
        this->initErr = true;
    }     

    hf3fs_utils->mount_point = mount_point_cstr;
}


nixl_status_t nixlHf3fsEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    nixl_status_t status;
    int fd;
    int ret;
    nixlHf3fsMetadata *md = new nixlHf3fsMetadata();
    switch (nixl_mem) {
        case DRAM_SEG:
            md->type = DRAM_SEG;
            status = NIXL_SUCCESS;
            break;
        case FILE_SEG:
            fd = mem.devId; 
            ret = 0;
            status = hf3fs_utils->registerFileHandle(fd, &ret);
            if (status != NIXL_SUCCESS) {
                delete md;
                NIXL_LOG_AND_RETURN_IF_ERROR(status, absl::StrFormat("Error - failed to register file handle %d", fd));
            }
            md->handle.fd = fd;
            md->handle.size = mem.len;
            md->handle.metadata = mem.metaInfo;
            md->type = nixl_mem;
            md->handle.hf3fs_fd = ret;
            hf3fs_file_map[fd] = md->handle;
            break;
        case VRAM_SEG:
        default:
            NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, "Error - type not supported");
    }
  
    out = (nixlBackendMD*) md;
    return status;
}

nixl_status_t nixlHf3fsEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlHf3fsMetadata *md = (nixlHf3fsMetadata *)meta;
    if (md->type == FILE_SEG) {
        hf3fs_utils->deregisterFileHandle(md->handle.fd);
    } else if (md->type == DRAM_SEG) {
        return NIXL_SUCCESS;
    } else {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, "Error - type not supported");
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlHf3fsEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args)
{
    nixlHf3fsBackendReqH *hf3fs_handle;
    bool handle_created = false;
    void                *addr = NULL;
    size_t              size = 0;
    size_t              offset = 0;
    int                 buf_cnt  = local.descCount();
    int                 file_cnt = remote.descCount();
    nixl_status_t       nixl_err = NIXL_ERR_UNKNOWN;
    const char          *nixl_mesg = nullptr;

    // Determine which lists contain file/memory descriptors
    const nixl_meta_dlist_t* file_list = nullptr;
    const nixl_meta_dlist_t* mem_list = nullptr;
    if (local.getType() == FILE_SEG) {
        file_list = &local;
        mem_list = &remote;
    } else if (remote.getType() == FILE_SEG) {
        file_list = &remote;
        mem_list = &local;
    } else {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error: No file descriptors");
    }

    if ((buf_cnt != file_cnt) ||
            ((operation != NIXL_READ) && (operation != NIXL_WRITE)))  {
        nixl_err = NIXL_ERR_INVALID_PARAM;
        nixl_mesg =  "Error: No file descriptors";
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error: Count mismatch or invalid operation selection");
    }

    // Create a new backend request handle if one doesn't exist
    if (handle == nullptr) {
        hf3fs_handle = new nixlHf3fsBackendReqH();
        handle_created = true;
    } else {
        // TODO: could this ever be a valid case?
        hf3fs_handle = (nixlHf3fsBackendReqH *) handle;
    }

    bool is_read = (operation == NIXL_READ);

    auto status = hf3fs_utils->createIOR(&hf3fs_handle->ior, file_cnt, is_read);
    if (status != NIXL_SUCCESS) {
        if (handle_created) {
            delete hf3fs_handle;
        }
        NIXL_LOG_AND_RETURN_IF_ERROR(status, "Error: Failed to create IOR");
    }

    for (int i = 0; i < file_cnt; i++) {
        // Get file descriptor from the proper list
        int file_descriptor = (*file_list)[i].devId;
        addr = (void*) (*mem_list)[i].addr;
        size = (*mem_list)[i].len;
        offset = (size_t) (*file_list)[i].addr;  // Offset in file       

        nixlHf3fsIO *io = new nixlHf3fsIO();
        if (io == nullptr) {
            nixl_err = NIXL_ERR_BACKEND;
            nixl_mesg = "Error: Failed to create IO";
            goto cleanup_handle;
        }

        // Store original memory address for later use during READ operations
        io->orig_addr = addr;
        io->size = size;
        io->is_read = is_read;
        io->offset = offset;

        status = hf3fs_utils->createIOV(&io->iov, addr, size, size);
        if (status != NIXL_SUCCESS) {
            delete io;
            nixl_err = status;
            nixl_mesg = "Error: Failed to wrap memory as IOV";
            goto cleanup_handle;
        }

        // For WRITE operations, copy data from source buffer to IOV buffer
        // For READ operations, we don't need to copy data now - we'll copy after read completes
        if (!is_read) {
            auto mem_copy = memcpy(io->iov.base, addr, size);
            if (mem_copy == nullptr) {
                delete io;
                nixl_err = NIXL_ERR_BACKEND;
                nixl_mesg = "Error: Failed to copy memory";
                goto cleanup_handle;
            }
        }

        io->fd = file_descriptor;
        hf3fs_handle->io_list.push_back(io);
    }

    hf3fs_handle->status = NIXL_HF3FS_STATUS_PREPARED;
    handle = (nixlBackendReqH*) hf3fs_handle;
    return NIXL_SUCCESS;

cleanup_handle:
    // Clean up previously created IOs in the list
    for (auto prev_io : hf3fs_handle->io_list) {
        delete prev_io;
    }
    hf3fs_handle->io_list.clear();
    if (handle_created) {
        delete hf3fs_handle;
    }
    NIXL_LOG_AND_RETURN_IF_ERROR(nixl_err, nixl_mesg);
    return nixl_err;
}

nixl_status_t nixlHf3fsEngine::postXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args)
{
    // Handle null pointer case - should have been initialized in prepXfer
    if (handle == nullptr) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error - handle is null in postXfer");
    }

    nixlHf3fsBackendReqH *hf3fs_handle = (nixlHf3fsBackendReqH *) handle;
    nixl_status_t        status;

    for (auto it = hf3fs_handle->io_list.begin(); it != hf3fs_handle->io_list.end(); ++it) {
        nixlHf3fsIO* io = *it;
        status = hf3fs_utils->prepIO(&hf3fs_handle->ior, &io->iov, io->iov.base,
                                     io->offset, io->size, io->fd, io->is_read, io);
        if (status != NIXL_SUCCESS) {
            NIXL_LOG_AND_RETURN_IF_ERROR(status, "Error: Failed to prepare IO");
        }
    }

    status = hf3fs_utils->postIOR(&hf3fs_handle->ior);
    if (status != NIXL_SUCCESS) {
        NIXL_LOG_AND_RETURN_IF_ERROR(status, "Error: Failed to post IOR");
    }

    hf3fs_handle->status = NIXL_HF3FS_STATUS_POSTED;
    return NIXL_IN_PROG;
}

nixl_status_t nixlHf3fsEngine::checkXfer(nixlBackendReqH* handle)
{
    if (handle == nullptr) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error: handle is null in checkXfer");
    }

    nixlHf3fsBackendReqH *hf3fs_handle = (nixlHf3fsBackendReqH *) handle;
    
    // Check if IOR is initialized
    if (&hf3fs_handle->ior == nullptr) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error: IOR is not initialized in checkXfer");
    }

    // Use a timeout to avoid hanging indefinitely (e.g., 100ms)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ts.tv_nsec += 100 * 1000 * 1000; // 100 milliseconds
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000;
    }
    
    int num_ios = hf3fs_handle->io_list.size();
    int num_cqes = num_ios > 1024 ? 1024 : num_ios;
    hf3fs_cqe cqes[num_cqes];
    int ret = hf3fs_wait_for_ios(&hf3fs_handle->ior, cqes, num_cqes, 1, &ts);
    // TODO: handle ret == 0 as timeout
    if (ret <= 0) {
        // Check specifically for timeout (-ETIMEDOUT or -EAGAIN)
        if (ret == 0 || ret == -ETIMEDOUT || ret == -EAGAIN) {
            return NIXL_IN_PROG;  // Return in-progress to retry later
        }

        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, absl::StrFormat("Error: Failed to wait for IOs: %d (errno: %d - %s)", ret, ret, strerror(ret)));
    }

    std::cout << "wait IOS ret: " << ret << std::endl;

    // Check if we have any errors in the completed I/O operations
    // Process the return values and copy the data for read operations
    auto io_iter = hf3fs_handle->io_list.begin();
    for (int i = 0; i < ret && io_iter != hf3fs_handle->io_list.end(); i++, ++io_iter) {
        if (cqes[i].result < 0) {
            NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, absl::StrFormat("Error: I/O operation completed with error: %d", cqes[i].result));
        }

        nixlHf3fsIO* io = *io_iter;
        if (io->is_read) {
            auto mem_copy = memcpy(io->orig_addr, io->iov.base, io->size);
            if (mem_copy == nullptr) {
                NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, "Error: Failed to copy memory after read");
            }
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t nixlHf3fsEngine::releaseReqH(nixlBackendReqH* handle)
{   
    nixlHf3fsBackendReqH *hf3fs_handle = (nixlHf3fsBackendReqH *) handle;
    for (auto io : hf3fs_handle->io_list) {
        hf3fs_dereg_fd(io->fd);
        delete io;
    }

    hf3fs_utils->destroyIOR(&hf3fs_handle->ior);
    delete hf3fs_handle;
    return NIXL_SUCCESS;
}

nixlHf3fsEngine::~nixlHf3fsEngine() {
    hf3fs_utils->closeHf3fsDriver();
    delete hf3fs_utils;
}
