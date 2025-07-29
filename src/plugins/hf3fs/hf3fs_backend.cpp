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
#include <sys/mman.h>
#include <sys/stat.h>
#include <filesystem>
#include "hf3fs_backend.h"
#include "hf3fs_log.h"
#include "common/str_tools.h"
#include "common/nixl_log.h"
#include "file/file_utils.h"

#define NUM_CQES 1024

long nixlHf3fsEngine::page_size = sysconf(_SC_PAGESIZE);

nixlHf3fsEngine::nixlHf3fsEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      mem_config(NIXL_HF3FS_MEM_CONFIG_AUTO) {
    hf3fs_utils = new hf3fsUtil();

    this->initErr = false;
    if (hf3fs_utils->openHf3fsDriver() == NIXL_ERR_BACKEND) {
        NIXL_ERROR << "Error opening HF3FS driver";
        this->initErr = true;
        return;
    }

    // Get mount point from parameters if available
    std::string mount_point = "/mnt/3fs/"; // default
    if (init_params && init_params->customParams) {
        if (init_params->customParams->count("mount_point") > 0) {
            mount_point = init_params->customParams->at("mount_point");
        }
        if (init_params->customParams->count("mem_config") > 0) {
            std::string mem_config_str = init_params->customParams->at("mem_config");
            if (mem_config_str == "dram") {
                mem_config = NIXL_HF3FS_MEM_CONFIG_DRAM;
            } else if (mem_config_str == "dram_zc") {
                mem_config = NIXL_HF3FS_MEM_CONFIG_DRAM_ZC;
            } else if (mem_config_str != "auto") {
                this->initErr = true;
                NIXL_ERROR << "Error: Invalid mem_config: " << mem_config_str;
                return;
            }
        }
    }

    char mount_point_cstr[256];
    auto ret = hf3fs_extract_mount_point(mount_point_cstr, 256, mount_point.c_str());
    if (ret < 0) {
        this->initErr = true;
        return;
    }

    hf3fs_utils->mount_point = mount_point_cstr;

    NIXL_DEBUG << "HF3FS: Page size: " << page_size;
}


nixl_status_t nixlHf3fsEngine::registerMem (const nixlBlobDesc &mem,
                                            const nixl_mem_t &nixl_mem,
                                            nixlBackendMD* &out)
{
    nixl_status_t status = NIXL_SUCCESS;

    switch (nixl_mem) {
    case DRAM_SEG: {
        // mmap requires the memory to be aligned to the page size
        // and the length to be a multiple of the page size
        if (mem_config == NIXL_HF3FS_MEM_CONFIG_AUTO ||
            mem_config == NIXL_HF3FS_MEM_CONFIG_DRAM_ZC) {
            if (page_size == 0 || (mem.addr % page_size == 0 && mem.len % page_size == 0)) {
                try {
                    nixlHf3fsDramZCMetadata *md =
                        new nixlHf3fsDramZCMetadata((uint8_t *)mem.addr, mem.len, *hf3fs_utils);
                    out = (nixlBackendMD *)md;
                    NIXL_DEBUG << "HF3FS: Registered shared memory(addr: " << std::hex << mem.addr
                               << ", len: " << mem.len << ")";
                    break;
                }
                catch (const nixlHf3fsShmException &e) {
                    NIXL_DEBUG << "HF3FS: Failed to register shared memory(addr: " << std::hex
                               << mem.addr << ", len: " << mem.len << "): " << e.what();
                    if (mem_config != NIXL_HF3FS_MEM_CONFIG_AUTO) {
                        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND,
                                         "Error: Failed to register shared memory");
                    }
                }
            } else if (mem_config == NIXL_HF3FS_MEM_CONFIG_DRAM_ZC) {
                HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM,
                                 "Error: DRAM_ZC requires page-aligned memory");
            }
        }
        out = new nixlHf3fsDramMetadata();
        NIXL_DEBUG << "HF3FS: Registered regular memory(addr: " << std::hex << mem.addr
                   << ", len: " << mem.len << ")";
        break;
    }
    case FILE_SEG: {
        int fd = mem.devId;

        // if the same file is reused - no need to re-register
        auto it = hf3fs_file_set.find(fd);
        if (it == hf3fs_file_set.end()) {
            int ret = 0;
            status = hf3fs_utils->registerFileHandle(fd, &ret);
            if (status != NIXL_SUCCESS) {
                HF3FS_LOG_RETURN(status,
                                 absl::StrFormat("Error - failed to register file handle %d", fd));
            }
            hf3fs_file_set.insert(fd);
        }

        nixlHf3fsFileMetadata *md = new nixlHf3fsFileMetadata();
        md->handle.fd = fd;
        md->handle.size = mem.len;
        md->handle.metadata = mem.metaInfo;
        out = (nixlBackendMD *)md;
        break;
    }
    default:
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND, "Error - type not supported");
    }

    return status;
}

nixl_status_t nixlHf3fsEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlHf3fsMetadata *md = (nixlHf3fsMetadata *)meta;
    if (md->type == NIXL_HF3FS_MEM_TYPE_FILE) {
        nixlHf3fsFileMetadata *file_md = (nixlHf3fsFileMetadata *)md;
        hf3fs_file_set.erase(file_md->handle.fd);
        hf3fs_utils->deregisterFileHandle(file_md->handle.fd);
    } else if (md->type != NIXL_HF3FS_MEM_TYPE_DRAM && md->type != NIXL_HF3FS_MEM_TYPE_DRAM_ZC) {
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND, "Error - invalid metadata type");
    }
    delete md;
    return NIXL_SUCCESS;
}

void nixlHf3fsEngine::cleanupIOList(nixlHf3fsBackendReqH *handle) const
{
    for (auto prev_io : handle->io_list) {
        if (prev_io->mem_type == NIXL_HF3FS_MEM_TYPE_DRAM) {
            hf3fs_utils->destroyIOV(&prev_io->iov);
        }
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
        auto mem_md = (nixlHf3fsMetadata *)(*mem_list)[i].metadataP;

        nixlHf3fsIO *io = new nixlHf3fsIO();
        if (io == nullptr) {
            nixl_err = NIXL_ERR_BACKEND;
            nixl_mesg = "Error: Failed to create IO";
            goto cleanup_handle;
        }

        if (mem_md->type == NIXL_HF3FS_MEM_TYPE_DRAM_ZC) {
            nixlHf3fsDramZCMetadata *shm_md = (nixlHf3fsDramZCMetadata *)mem_md;
            status = hf3fs_utils->wrapIOV(&io->iov,
                                          shm_md->mapped_addr,
                                          shm_md->mapped_size,
                                          size,
                                          shm_md->uuid.get_data().data());
            if (status != NIXL_SUCCESS) {
                delete io;
                nixl_err = status;
                nixl_mesg = "Error: Failed to wrap memory as IOV";
                goto cleanup_handle;
            }
        } else {
            status = hf3fs_utils->createIOV(&io->iov, size, size);
            if (status != NIXL_SUCCESS) {
                delete io;
                nixl_err = status;
                nixl_mesg = "Error: Failed to create IOV";
                goto cleanup_handle;
            }

            // For WRITE operations, copy data from source buffer to IOV buffer
            // For READ operations, we don't need to copy data now - we'll copy after read completes
            // TODO: Should the data copy in postXfer? User could still modify the data after
            // prepXfer
            if (!is_read) {
                memcpy(io->iov.base, addr, size);
            }
        }
        // Store original memory address for later use during READ operations
        io->addr = addr;
        io->size = size;
        io->is_read = is_read;
        io->offset = offset;
        io->mem_type = mem_md->type;

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
        void *addr = (io->mem_type == NIXL_HF3FS_MEM_TYPE_DRAM) ? io->iov.base : io->addr;

        status = hf3fs_utils->prepIO(
            &hf3fs_handle->ior, &io->iov, addr, io->offset, io->size, io->fd, io->is_read, io);
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
                if (io->is_read && io->mem_type == NIXL_HF3FS_MEM_TYPE_DRAM) {
                    memcpy(io->addr, io->iov.base, io->size);
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

nixlHf3fsDramZCMetadata::nixlHf3fsDramZCMetadata(uint8_t *addr, size_t len, hf3fsUtil &utils)
    : nixlHf3fsMetadata(NIXL_HF3FS_MEM_TYPE_DRAM_ZC) {

    // Create shared memory name using UUID (POSIX shared memory names start with /)
    shm_name = "/nixl_hf3fs." + uuid.to_string();

    // Create POSIX shared memory object
    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        NIXL_PERROR << "Failed to create POSIX shared memory";
        throw nixlHf3fsShmException("Failed to create POSIX shared memory: " +
                                    std::string(strerror(errno)));
    }

    // Set the size of the shared memory object
    if (ftruncate(shm_fd, len) == -1) {
        NIXL_PERROR << "Failed to set shared memory size";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        throw nixlHf3fsShmException("Failed to set shared memory size: " +
                                    std::string(strerror(errno)));
    }

    /**
     * The purpose of this is to keep the content in the memory.
     * mmap seems to populate the memory with the content of the shared memory fd,
     * without the pwrite here the memory will be all 0s after mmap.
     *
     * TODO: Is it a valid operation to write to the shared memory fd?
     */
    ssize_t written = pwrite(shm_fd, addr, len, 0);
    if (written < 0 || written != (ssize_t)len) {
        NIXL_PERROR << "Failed to write to shared memory";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        throw nixlHf3fsShmException("Failed to write to shared memory: " +
                                    std::string(strerror(errno)));
    }

    mapped_addr = mmap(addr, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, shm_fd, 0);
    if (mapped_addr == MAP_FAILED) {
        NIXL_PERROR << "Failed to map shared memory";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        throw nixlHf3fsShmException("Failed to map shared memory: " + std::string(strerror(errno)));
    }

    mapped_size = len;
    std::string shm_path = "/dev/shm" + shm_name;

    link_path = absl::StrFormat("%s/3fs-virt/iovs/%s", utils.mount_point, uuid.to_string());
    if (symlink(shm_path.c_str(), link_path.c_str()) == -1) {
        NIXL_PERROR << "Failed to create symlink";
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        throw nixlHf3fsShmException("Failed to create symlink " + std::string(strerror(errno)));
    }

    // Close the file descriptor as it's no longer needed after mmap
    close(shm_fd);

    NIXL_INFO << "Created POSIX shared memory: " << shm_name << " with size: " << len;
}

nixlHf3fsDramZCMetadata::~nixlHf3fsDramZCMetadata() {
    if (unlink(link_path.c_str()) && errno != ENOENT) {
        NIXL_PERROR << "Failed to remove symlink";
    }

    if (shm_unlink(shm_name.c_str()) == -1) {
        NIXL_PERROR << "Failed to unlink shared memory";
    }

    NIXL_INFO << "Cleaned up POSIX shared memory: " << shm_name;
}
