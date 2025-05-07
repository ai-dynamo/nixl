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
#include "backend/backend_engine.h"

class uringQueue {
    private:
        io_uring uring;    // The io_uring instance for async I/O operations
        int num_entries;   // Total number of entries expected in this ring
        int num_completed; // Number of completed operations so far

        // Delete copy and move operations to prevent accidental copying of kernel resources
        uringQueue(const uringQueue&) = delete;
        uringQueue& operator=(const uringQueue&) = delete;
        uringQueue(uringQueue&&) = delete;
        uringQueue& operator=(uringQueue&&) = delete;

    public:
        uringQueue(int num_entries, io_uring_params params);
        ~uringQueue();
        nixl_status_t submit();
        nixl_status_t checkCompleted();
        struct io_uring_sqe *getSqe();

        enum class UringError {
            INIT,
        };
};

class nixlPosixBackendReqH : public nixlBackendReqH {
    private:
        using io_uring_prep_func_t = void (*)(struct io_uring_sqe *, int, void *, unsigned, __u64);

        const nixl_xfer_op_t         &operation;              // The transfer operation (read/write)
        const nixl_meta_dlist_t      &local;                  // Local memory descriptor list
        const int                    local_desc_count;        // Number of descriptors in the local memory list
        const nixl_meta_dlist_t      &remote;                 // Remote memory descriptor list
        const nixl_opt_b_args_t      *opt_args;               // Optional backend-specific arguments, currently unused
        int                          num_urings;              // Number of uringQueue instances needed for this request
        std::vector<std::unique_ptr<uringQueue>> uring;       // Vector of uringQueue instances
        short                        num_uring_params;        // Number of io_uring parameter sets (either 1 or 2)
        std::vector<io_uring_params> uring_params;            // Vector of io_uring parameters for uringQueue initialization
                                                              // (all but last urings use uring_params[1], last uring uses uring_params[0])
        io_uring_prep_func_t         io_uring_prep_func;      // Function pointer for preparing io_uring operations (io_uring_prep_read/write)
        bool                         is_prepped;              // Flag indicating if operations are prepared
        nixl_status_t                status;                  // Current status of the transfer operation

        void fillUringParams();                               // Initialize io_uring parameters
        void initUrings();                                    // Initialize io_uring instances

    public:
        nixlPosixBackendReqH(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const nixl_opt_b_args_t* opt_args=nullptr);
        ~nixlPosixBackendReqH();

        nixl_status_t postXfer();
        nixl_status_t prepXfer();
        nixl_status_t checkXfer();

        enum class OperationError {
            INVALID_OPERATION
        };
};

class nixlPosixEngine : public nixlBackendEngine {
    public:
        nixlPosixEngine(const nixlBackendInitParams* init_params): nixlBackendEngine (init_params) {}
        ~nixlPosixEngine() {};

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

        nixl_mem_list_t getSupportedMems() const;

        nixl_status_t connect(const std::string &remote_agent) {
            return NIXL_SUCCESS;
        }

        nixl_status_t disconnect(const std::string &remote_agent) {
            return NIXL_SUCCESS;
        }

        nixl_status_t loadLocalMD (nixlBackendMD* input, nixlBackendMD* &output) {
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
