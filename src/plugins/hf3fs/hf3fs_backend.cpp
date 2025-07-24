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
#include <cctype>
#include <atomic>
#include <errno.h>
#include "hf3fs_backend.h"
#include "hf3fs_log.h"
#include "common/str_tools.h"
#include "common/nixl_log.h"
#include "file/file_utils.h"

#define NUM_CQES 1024

nixlHf3fsEngine::nixlHf3fsEngine (const nixlBackendInitParams* init_params)
    : nixlBackendEngine (init_params)
{
    hf3fs_utils = new hf3fsUtil();

    this->initErr = false;
    if (hf3fs_utils->openHf3fsDriver() == NIXL_ERR_BACKEND) {
        NIXL_ERROR << "Error opening HF3FS driver";
        this->initErr = true;
    }

    // Get mount point from parameters if available
    std::string mount_point = "/mnt/3fs/"; // default
    if (init_params &&
        init_params->customParams &&
        init_params->customParams->count("mount_point") > 0) {
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
    int ret;
    nixlHf3fsMetadata *md = new nixlHf3fsMetadata();

    switch (nixl_mem) {
        case DRAM_SEG:
            md->type = DRAM_SEG;
            status = NIXL_SUCCESS;
            break;
        case FILE_SEG: {
            // Check if we already have a file descriptor for this devId
            auto it = hf3fs_file_set.find(mem.devId);
            if (it != hf3fs_file_set.end()) {
                md->handle.fd = *it;
                md->handle.size = mem.len;
                md->handle.metadata = mem.metaInfo;
                md->type = nixl_mem;
                status = NIXL_SUCCESS;
                break;
            }

            ret = 0;
            status = hf3fs_utils->registerFileHandle(mem.devId, &ret);
            if (status != NIXL_SUCCESS) {
                delete md;
                HF3FS_LOG_RETURN(
                    status,
                    absl::StrFormat("Error - failed to register file handle %d", mem.devId));
            }
            md->handle.fd = mem.devId;
            md->handle.size = mem.len;
            md->handle.metadata = mem.metaInfo;
            md->type = nixl_mem;

            hf3fs_file_set.insert(mem.devId);
            break;
        }
        case VRAM_SEG:
        default:
            HF3FS_LOG_RETURN(NIXL_ERR_BACKEND, "Error - type not supported");
    }

    out = (nixlBackendMD*) md;
    return status;
}

nixl_status_t nixlHf3fsEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlHf3fsMetadata *md = (nixlHf3fsMetadata *)meta;
    if (md->type == FILE_SEG) {
        hf3fs_file_set.erase (md->handle.fd);
        hf3fs_utils->deregisterFileHandle(md->handle.fd);
        // No need to close fd since we're not opening files
    } else if (md->type == DRAM_SEG) {
        return NIXL_SUCCESS;
    } else {
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND, "Error - type not supported");
    }
    return NIXL_SUCCESS;
}

void nixlHf3fsEngine::cleanupIOList(nixlHf3fsBackendReqH *handle) const
{
    for (auto prev_io : handle->io_list) {
        hf3fs_utils->destroyIOV(&prev_io->iov);
        delete prev_io;
    }

    handle->io_list.clear();
}

void nixlHf3fsEngine::cleanupIOThread(nixlHf3fsBackendReqH *handle) const
{
    if (handle->io_status.thread != nullptr) {
        handle->io_status.stop_thread = true;
        handle->io_status.thread->join();

        delete handle->io_status.thread;
        handle->io_status.thread = nullptr;
        handle->io_status.error_status = NIXL_SUCCESS;
        handle->io_status.error_message = "";
        handle->io_status.stop_thread = false;
    }
}

nixl_status_t nixlHf3fsEngine::prepXfer (const nixl_xfer_op_t &operation,
                                         const nixl_meta_dlist_t &local,
                                         const nixl_meta_dlist_t &remote,
                                         const std::string &remote_agent,
                                         nixlBackendReqH* &handle,
                                         const nixl_opt_b_args_t* opt_args) const
{
    nixlHf3fsBackendReqH *hf3fs_handle;
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
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM, "Error: No file descriptors");
    }

    if ((buf_cnt != file_cnt) ||
        ((operation != NIXL_READ) && (operation != NIXL_WRITE)))  {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM,
            "Error: Count mismatch or invalid operation selection");
    }

    hf3fs_handle = new nixlHf3fsBackendReqH();

    bool is_read = (operation == NIXL_READ);

    auto status = hf3fs_utils->createIOR(&hf3fs_handle->ior, file_cnt, is_read);
    if (status != NIXL_SUCCESS) {
        delete hf3fs_handle;
        HF3FS_LOG_RETURN(status, "Error: Failed to create IOR");
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

    handle = (nixlBackendReqH*) hf3fs_handle;
    return NIXL_SUCCESS;

cleanup_handle:
    // Clean up previously created IOs in the list
    cleanupIOList(hf3fs_handle);
    delete hf3fs_handle;
    HF3FS_LOG_RETURN(nixl_err, nixl_mesg);
}

nixl_status_t nixlHf3fsEngine::postXfer (const nixl_xfer_op_t &operation,
                                         const nixl_meta_dlist_t &local,
                                         const nixl_meta_dlist_t &remote,
                                         const std::string &remote_agent,
                                         nixlBackendReqH* &handle,
                                         const nixl_opt_b_args_t* opt_args) const
{
    nixlHf3fsBackendReqH *hf3fs_handle = (nixlHf3fsBackendReqH *) handle;
    nixl_status_t        status;

    if (hf3fs_handle->io_list.empty()) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM, "Error: empty io list");
    }

    if (UINT_MAX - hf3fs_handle->num_ios < hf3fs_handle->io_list.size()) {
        HF3FS_LOG_RETURN(NIXL_ERR_NOT_ALLOWED, "Error: more than UINT_MAX ios");
    }

    for (auto it = hf3fs_handle->io_list.begin(); it != hf3fs_handle->io_list.end(); ++it) {
        nixlHf3fsIO* io = *it;
        status = hf3fs_utils->prepIO(&hf3fs_handle->ior, &io->iov, io->iov.base,
                                     io->offset, io->size, io->fd, io->is_read, io);
        if (status != NIXL_SUCCESS) {
            HF3FS_LOG_RETURN(status, "Error: Failed to prepare IO");
        }
    }

    status = hf3fs_utils->postIOR(&hf3fs_handle->ior);
    if (status != NIXL_SUCCESS) {
        HF3FS_LOG_RETURN(status, "Error: Failed to post IOR");
    }

    // postXfer may be called multiple times, so we need to check if the thread is already running
    if (hf3fs_handle->io_status.thread == nullptr) {
        hf3fs_handle->io_status.thread = new std::thread(waitForIOsThread, hf3fs_handle,
                                                         hf3fs_utils);
        if (hf3fs_handle->io_status.thread == nullptr) {
            HF3FS_LOG_RETURN(NIXL_ERR_BACKEND, "Error: Failed to create io thread");
        }
    }

    hf3fs_handle->num_ios += hf3fs_handle->io_list.size();

    return NIXL_IN_PROG;
}

void nixlHf3fsEngine::waitForIOsThread(void* handle, void *utils)
{
    nixlHf3fsBackendReqH* hf3fs_handle = (nixlHf3fsBackendReqH*)handle;
    hf3fsUtil* hf3fs_utils = (hf3fsUtil*)utils;
    nixlH3fsThreadStatus* io_status = &hf3fs_handle->io_status;
    hf3fs_cqe* cqes = new hf3fs_cqe[NUM_CQES];

    while (!io_status->stop_thread && io_status->error_status == NIXL_SUCCESS) {
        // Check if we've processed all IOs
        if (hf3fs_handle->completed_ios >= hf3fs_handle->num_ios) {
            // User may call postXfer multiple times, so we could not exit yet,
            // so we must wait for stop condition
            sched_yield();
            continue;
        }

        // Use a short timeout to allow thread to check stop condition
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        ts.tv_nsec += 10 * 1000 * 1000; // 10 milliseconds
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000;
        }

        int num_completed = 0;
        nixl_status_t status = hf3fs_utils->waitForIOs(&hf3fs_handle->ior, cqes, NUM_CQES, 1, &ts,
                                                       &num_completed);
        if (status != NIXL_SUCCESS) {
            io_status->error_status = status;
            io_status->error_message = "Error: Failed to wait for IOs";
            break;
        }

        if (num_completed > 0) {
            for (int i = 0; i < num_completed; i++) {
                if (cqes[i].result < 0) {
                    io_status->error_status = NIXL_ERR_BACKEND;
                    io_status->error_message = absl::StrFormat(
                        "Error: I/O operation completed with error: %d", cqes[i].result);
                    break;
                }

                nixlHf3fsIO* io = (nixlHf3fsIO*)cqes[i].userdata;

                if (io->is_read) {
                    auto mem_copy = memcpy(io->orig_addr, io->iov.base, io->size);
                    if (mem_copy == nullptr) {
                        io_status->error_status = NIXL_ERR_BACKEND;
                        io_status->error_message = "Error: Failed to copy memory after read";
                        break;
                    }
                }

                hf3fs_handle->completed_ios++;
            }
        }
    }

    delete[] cqes;
}

nixl_status_t nixlHf3fsEngine::checkXfer(nixlBackendReqH* handle) const
{
    if (handle == nullptr) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM, "Error: handle is null in checkXfer");
    }

    nixlHf3fsBackendReqH *hf3fs_handle = (nixlHf3fsBackendReqH *) handle;

    // Check if IOR is initialized
    if (&hf3fs_handle->ior == nullptr) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM,
            "Error: IOR is not initialized in checkXfer");
    }

    if (hf3fs_handle->io_status.thread == nullptr) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM,
            "Error: io thread is not initialized in checkXfer");
    }

    if (hf3fs_handle->io_status.error_status != NIXL_SUCCESS) {
        nixl_status_t error_status = hf3fs_handle->io_status.error_status;
        std::string error_message = hf3fs_handle->io_status.error_message;
        cleanupIOThread(hf3fs_handle);
        HF3FS_LOG_RETURN(error_status, error_message);
    }

    if (hf3fs_handle->completed_ios < hf3fs_handle->num_ios) {
        return NIXL_IN_PROG;
    }

    cleanupIOThread(hf3fs_handle);
    return NIXL_SUCCESS;
}

nixl_status_t nixlHf3fsEngine::releaseReqH(nixlBackendReqH* handle) const
{
    nixlHf3fsBackendReqH *hf3fs_handle = (nixlHf3fsBackendReqH *) handle;

    cleanupIOThread(hf3fs_handle);
    cleanupIOList(hf3fs_handle);
    hf3fs_utils->destroyIOR(&hf3fs_handle->ior);
    delete hf3fs_handle;
    return NIXL_SUCCESS;
}

nixlHf3fsEngine::~nixlHf3fsEngine() {
    hf3fs_utils->closeHf3fsDriver();
    delete hf3fs_utils;
}

nixl_status_t
nixlHf3fsEngine::queryMem(const nixl_reg_dlist_t &descs,
                          std::vector<nixl_query_resp_t> &resp) const {
    // Extract metadata from descriptors which are file names
    // Different plugins might customize parsing of metaInfo to get the file names
    std::vector<nixl_blob_t> metadata(descs.descCount());
    for (int i = 0; i < descs.descCount(); ++i)
        metadata[i] = descs[i].metaInfo;

    return nixl::queryFileInfoList(metadata, resp);
}
