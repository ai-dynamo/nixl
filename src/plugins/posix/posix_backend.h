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
#include "backend/backend_engine.h"
#include "async_queue.h"

// Forward declarations
class aioQueue;
class UringQueue;
class QueueFactory;

// Factory for creating IO queues
class IOQueueFactory {
public:
    static std::unique_ptr<nixlPosixQueue> createQueue(bool use_aio, int num_entries, bool is_read = false,
                                                      const void* params = nullptr);
};

class nixlPosixBackendReqH : public nixlBackendReqH {
private:
    const nixl_xfer_op_t         &operation;              // The transfer operation (read/write)
    const nixl_meta_dlist_t      &local;                  // Local memory descriptor list
    const nixl_meta_dlist_t      &remote;                 // Remote memory descriptor list
    const nixl_opt_b_args_t      *opt_args;               // Optional backend-specific arguments
    const nixl_b_params_t        *custom_params_;         // Custom backend parameters
    int                          queue_depth_;            // Queue depth for async I/O
    std::unique_ptr<nixlPosixQueue> queue;                // Async I/O queue instance
    bool                         is_prepped;              // Flag indicating if operations are prepared
    nixl_status_t                status;                  // Current status of the transfer operation
    bool                         use_aio_;                // Whether to use AIO instead of io_uring

    nixl_status_t initQueues(bool use_aio);               // Initialize async I/O queue

public:
    nixlPosixBackendReqH(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const nixl_opt_b_args_t* opt_args,
                         const nixl_b_params_t* custom_params);
    ~nixlPosixBackendReqH();

    nixl_status_t postXfer();
    nixl_status_t prepXfer();
    nixl_status_t checkXfer();
};

class nixlPosixEngine : public nixlBackendEngine {
private:
    bool use_aio;                             // Use AIO instead of io_uring
    const nixl_mem_list_t supported_mems = {  // supported memory types
        FILE_SEG,
        DRAM_SEG
    };

    bool validatePrepXferParams(const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                const std::string &local_agent);

public:
    nixlPosixEngine(const nixlBackendInitParams* init_params);
    virtual ~nixlPosixEngine() = default;

    // Initialize the engine after construction
    nixl_status_t init();

    bool supportsNotif   () const {
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

    nixl_status_t loadLocalMD(nixlBackendMD* input,
                              nixlBackendMD* &output) {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t unloadMD(nixlBackendMD* input) {
        return NIXL_SUCCESS;
    }

    nixl_status_t registerMem(const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD* &out) override;

    nixl_status_t deregisterMem(nixlBackendMD* meta);

    nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args=nullptr);

    nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args=nullptr);

    nixl_status_t checkXfer(nixlBackendReqH* handle);
    nixl_status_t releaseReqH(nixlBackendReqH* handle);
};

#endif // POSIX_BACKEND_H
