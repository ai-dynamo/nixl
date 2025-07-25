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
#include <list>
#include <unordered_set>
#include <thread>
#include "hf3fs_utils.h"
#include "backend/backend_engine.h"

class nixlHf3fsMetadata : public nixlBackendMD {
    public:
        hf3fsFileHandle  handle;
        nixl_mem_t     type;

        nixlHf3fsMetadata() : nixlBackendMD(true) { }
        ~nixlHf3fsMetadata() { }
};

class nixlHf3fsIO {
    public:
        hf3fs_iov iov;
        int fd;
        void* orig_addr;  // Original memory address for copying after read
        size_t size;      // Size of the buffer
        bool is_read;     // Whether this is a read operation
        size_t offset;    // Offset in the file

        nixlHf3fsIO() : fd(-1), orig_addr(nullptr), size(0), is_read(false) {}
        ~nixlHf3fsIO() {}
};

class nixlH3fsThreadStatus {
    public:
        std::thread *thread;
        nixl_status_t error_status;
        std::string error_message;
        bool stop_thread;

        nixlH3fsThreadStatus() : thread(nullptr), error_status(NIXL_SUCCESS), error_message(""),
                                 stop_thread(false) {}
        ~nixlH3fsThreadStatus() {}
};

class nixlHf3fsBackendReqH : public nixlBackendReqH {
    public:
       std::list<nixlHf3fsIO *> io_list;
       hf3fs_ior ior;
       uint32_t completed_ios;  // Number of completed IOs
       uint32_t num_ios;        // Number of submitted IOs
       nixlH3fsThreadStatus io_status;

       nixlHf3fsBackendReqH() : completed_ios(0), num_ios(0) {}
       ~nixlHf3fsBackendReqH() {}
};


class nixlHf3fsEngine : public nixlBackendEngine {
    private:
        hf3fsUtil                      *hf3fs_utils;
        std::unordered_set<int> hf3fs_file_set;

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
