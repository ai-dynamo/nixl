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
#include <absl/strings/str_format.h>
#include "backend/backend_engine.h"
#include "posix_queue.h"

class nixlPosixBackendReqH : public nixlBackendReqH {
private:
    const nixl_xfer_op_t            &operation;      // The transfer operation (read/write)
    const nixl_meta_dlist_t         &local;          // Local memory descriptor list
    const nixl_meta_dlist_t         &remote;         // Remote memory descriptor list
    const nixl_opt_b_args_t         *opt_args;       // Optional backend-specific arguments
    const nixl_b_params_t           *custom_params_; // Custom backend parameters
    const int                       queue_depth_;    // Queue depth for async I/O
    std::unique_ptr<nixlPosixQueue> queue;           // Async I/O queue instance
    const nixlPosixQueue::queue_t   queue_type_;     // Type of queue used

    nixl_status_t initQueues();                      // Initialize async I/O queue

public:
    nixlPosixBackendReqH(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const nixl_opt_b_args_t* opt_args,
                         const nixl_b_params_t* custom_params);
    ~nixlPosixBackendReqH() {};

    nixl_status_t postXfer();
    nixl_status_t prepXfer();
    nixl_status_t checkXfer();

    // Exception classes
    class exception: public std::exception {
        private:
            const nixl_status_t code_;
        public:
            exception(const std::string& msg, nixl_status_t code)
                : std::exception(), code_(code) {}
            nixl_status_t code() const noexcept { return code_; }
    };
};

class nixlPosixEngine : public nixlBackendEngine {
private:
    const nixlPosixQueue::queue_t queue_type_;

public:
    nixlPosixEngine(const nixlBackendInitParams* init_params);
    virtual ~nixlPosixEngine() = default;

    bool supportsRemote() const override {
        return false;
    }

    bool supportsLocal() const override {
        return true;
    }

    bool supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t getSupportedMems() const override {
        return {FILE_SEG, DRAM_SEG};
    }

    nixl_status_t registerMem(const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD* &out) override;

    nixl_status_t deregisterMem(nixlBackendMD* meta) override;

    nixl_status_t connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t unloadMD(nixlBackendMD* input) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args=nullptr) const override;

    nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args=nullptr) const override;

    nixl_status_t checkXfer(nixlBackendReqH* handle) const override;
    nixl_status_t releaseReqH(nixlBackendReqH* handle) const override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t loadLocalMD(nixlBackendMD* input, nixlBackendMD* &output) override {
        output = input;
        return NIXL_SUCCESS;
    }
};

#endif // POSIX_BACKEND_H
