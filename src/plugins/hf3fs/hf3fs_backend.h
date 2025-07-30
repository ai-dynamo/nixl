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

#ifndef __HF3FS_BACKEND_H
#define __HF3FS_BACKEND_H

#include <nixl.h>
#include <nixl_types.h>
#include <unistd.h>
#include <fcntl.h>
#include "common/uuid_v4.h"
#include <list>
#include <unordered_set>
#include <thread>
#include "hf3fs_utils.h"
#include "backend/backend_engine.h"

class nixlHf3fsShmException : public std::runtime_error {
public:
    nixlHf3fsShmException(const std::string &message) : std::runtime_error(message) {}
};

enum nixl_hf3fs_mem_type {
    NIXL_HF3FS_MEM_TYPE_FILE = 0,
    NIXL_HF3FS_MEM_TYPE_DRAM = 1,
    NIXL_HF3FS_MEM_TYPE_DRAM_ZC = 2,
};

enum nixl_hf3fs_mem_config {
    NIXL_HF3FS_MEM_CONFIG_AUTO = 0,
    NIXL_HF3FS_MEM_CONFIG_DRAM = 1,
    NIXL_HF3FS_MEM_CONFIG_DRAM_ZC = 2,
};

class nixlHf3fsMetadata : public nixlBackendMD {
    public:
        nixl_hf3fs_mem_type type;

        nixlHf3fsMetadata(nixl_hf3fs_mem_type type) : nixlBackendMD(true), type(type) {}
};

class nixlHf3fsFileMetadata : public nixlHf3fsMetadata {
public:
    hf3fsFileHandle handle;

    nixlHf3fsFileMetadata() : nixlHf3fsMetadata(NIXL_HF3FS_MEM_TYPE_FILE) {}
};

class nixlHf3fsDramMetadata : public nixlHf3fsMetadata {
public:
    nixlHf3fsDramMetadata() : nixlHf3fsMetadata(NIXL_HF3FS_MEM_TYPE_DRAM) {}
};

class nixlHf3fsDramZCMetadata : public nixlHf3fsMetadata {
public:
    std::string shm_name;
    std::string link_path;
    void *mapped_addr;
    size_t mapped_size;
    nixl::UUIDv4 uuid;

    nixlHf3fsDramZCMetadata(uint8_t *addr, size_t len, hf3fsUtil &utils);
    ~nixlHf3fsDramZCMetadata();
};

class nixlHf3fsIO {
    public:
        hf3fs_iov iov;
        int fd = -1;
        void *addr = nullptr; // Start address to read from/write to
        size_t size = 0; // Size of the buffer
        bool is_read = false; // Whether this is a read operation
        size_t offset;    // Offset in the file
        nixl_hf3fs_mem_type mem_type;

        nixlHf3fsIO() = default;
};

class nixlH3fsThreadStatus {
    public:
        std::thread *thread = nullptr;
        nixl_status_t error_status = NIXL_SUCCESS;
        std::string error_message = "";
        bool stop_thread = false;

        nixlH3fsThreadStatus() = default;
};

class nixlHf3fsBackendReqH : public nixlBackendReqH {
    public:
        std::list<nixlHf3fsIO *> io_list;
        hf3fs_ior ior;
        uint32_t completed_ios = 0; // Number of completed IOs
        uint32_t num_ios = 0; // Number of submitted IOs
        nixlH3fsThreadStatus io_status;

        nixlHf3fsBackendReqH() = default;
};


class nixlHf3fsEngine : public nixlBackendEngine {
    private:
        hf3fsUtil *hf3fs_utils;
        std::unordered_set<int> hf3fs_file_set;
        nixl_hf3fs_mem_config mem_config;
        static long page_size;

        void cleanupIOList(nixlHf3fsBackendReqH *handle) const;
        void cleanupIOThread(nixlHf3fsBackendReqH *handle) const;
        static void waitForIOsThread(void* handle, void *utils);
    public:
        nixlHf3fsEngine(const nixlBackendInitParams* init_params);
        ~nixlHf3fsEngine();

        // File operations - target is the distributed FS
        // So no requirements to connect to target.
        // Just treat it locally.
        bool supportsNotif () const {
            return false;
        }
        bool supportsRemote  () const {
            return false;
        }
        bool supportsLocal   () const {
            return true;
        }
        bool supportsProgTh  () const {
            return false;
        }

        nixl_mem_list_t getSupportedMems () const {
            nixl_mem_list_t mems;
            mems.push_back(FILE_SEG);
            mems.push_back(DRAM_SEG);
            return mems;
        }

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
        nixl_status_t registerMem(const nixlBlobDesc &mem,
                                  const nixl_mem_t &nixl_mem,
                                  nixlBackendMD* &out);
        nixl_status_t deregisterMem (nixlBackendMD *meta);

        nixl_status_t prepXfer (const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH* &handle,
                                const nixl_opt_b_args_t* opt_args=nullptr) const;

        nixl_status_t postXfer (const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH* &handle,
                                const nixl_opt_b_args_t* opt_args=nullptr) const;

        nixl_status_t checkXfer (nixlBackendReqH* handle) const;
        nixl_status_t releaseReqH(nixlBackendReqH* handle) const;

        nixl_status_t
        queryMem(const nixl_reg_dlist_t &descs,
                 std::vector<nixl_query_resp_t> &resp) const override;
};
#endif
