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

#ifndef POSIX_BACKEND_H
#define POSIX_BACKEND_H

#include <memory>
#include <string>
#include <vector>
#include <liburing.h>
#include "posix_utils.h"

class nixlPosixBackendReqH : public nixlBackendReqH {
    public:
        unsigned int num_entries;
        bool is_prepped = false;

        nixlPosixBackendReqH() : req_id(req_id++) {}
        ~nixlPosixBackendReqH() {
            for (struct io_uring_cqe* cqe_ptr : cqe) {
                if (cqe_ptr) {
                    io_uring_cqe_seen(&nixlPosixEngine::uring, cqe_ptr);
                }
            }
        }
};

class nixlPosixEngine : public nixlBackendEngine {
    private:
        static posixUtil *posix_utils;
        static int uring_init_status        = 1;
        static io_uring_params uring_params = {
            .cq_entries = NIXL_POSIX_RING_SIZE,
        };
        static io_uring uring;
        
        static io_uring &uringGetInstance();
        static int uringGetInitStatus();
    
        std::unordered_map<long unsigned int, struct io_uring_cqe **> cqe_map;
        void registerCompletions();

    public:
        nixlPosixEngine();
        ~nixlPosixEngine();

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
                                const nixl_opt_b_args_t* opt_args=nullptr);

        nixl_status_t postXfer (const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH* &handle,
                                const nixl_opt_b_args_t* opt_args=nullptr);

        nixl_status_t checkXfer (nixlBackendReqH* handle);
        
        nixl_status_t releaseReqH(nixlBackendReqH* handle);
    
};

#endif // POSIX_BACKEND_H 