/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Serapheim Dimitropoulos, WekaIO Ltd.
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

#ifndef __GDS_MT_BACKEND_H
#define __GDS_MT_BACKEND_H

#include <nixl.h>
#include <nixl_types.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <fcntl.h>
#include <future>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <unordered_map>
#include <mutex>
#include "gds_mt_utils.h"
#include "backend/backend_engine.h"
#include "taskflow/taskflow.hpp"

class nixlGdsMtMetadata : public nixlBackendMD {
    public:
        nixlGdsMtMetadata() : nixlBackendMD(true) { }
        ~nixlGdsMtMetadata() { }

        std::shared_ptr<gdsMtFileHandle> handle;
        std::unique_ptr<gdsMtMemBuf> buf;
        nixl_mem_t type;
        int ref_cnt{1};  // Reference count for shared file handles
};

struct GdsMtTransferRequestH {
        GdsMtTransferRequestH(void* a, size_t s, size_t offset, CUfileHandle_t handle, CUfileOpcode_t operation) :
        addr{a}, size{s}, file_offset{offset}, fh{handle}, op{operation} {}

        void*           addr{nullptr};
        size_t          size{0};
        size_t          file_offset{0};
        CUfileHandle_t  fh{nullptr};
        CUfileOpcode_t  op{CUFILE_READ};
};

class nixlGdsMtBackendReqH : public nixlBackendReqH {
    public:
        // Ensure any running taskflow completes before destruction
        ~nixlGdsMtBackendReqH() {
            if (running_transfer.valid()) {
                running_transfer.wait();
            }
        }

        std::vector<GdsMtTransferRequestH> request_list;
        tf::Taskflow taskflow;
        std::future<void> running_transfer;

        // Only changes if an error actually occurs
        std::atomic<nixl_status_t> overall_status;
};

class nixlGdsMtEngine : public nixlBackendEngine {
    public:
        nixlGdsMtEngine(const nixlBackendInitParams* init_params);
        // Note: The destructor of the TaskFlow executor runs wait_for_all() to
        // wait for all submitted taskflows to complete and then notifies all worker
        // threads to stop and join these threads.
        // Note: The gds_mt_utils_ destructor automatically handles driver cleanup.
        ~nixlGdsMtEngine() = default;

        // Disable copy/move
        nixlGdsMtEngine(const nixlGdsMtEngine&) = delete;
        nixlGdsMtEngine& operator=(const nixlGdsMtEngine&) = delete;

        bool supportsNotif() const override {
            return false;
        }
        bool supportsRemote() const override {
            return false;
        }
        bool supportsLocal() const override {
            return true;
        }
        bool supportsProgTh() const override {
            return false;
        }

        nixl_mem_list_t getSupportedMems() const override {
            return {DRAM_SEG, VRAM_SEG, FILE_SEG};
        }

        nixl_status_t connect(const std::string &remote_agent) override {
            return NIXL_SUCCESS;
        }

        nixl_status_t disconnect(const std::string &remote_agent) override {
            return NIXL_SUCCESS;
        }

        nixl_status_t loadLocalMD(nixlBackendMD* input,
                                  nixlBackendMD* &output) override {
            output = input;
            return NIXL_SUCCESS;
        }

        nixl_status_t unloadMD(nixlBackendMD* input) override {
            return NIXL_SUCCESS;
        }
        nixl_status_t registerMem(const nixlBlobDesc &mem,
                                  const nixl_mem_t &nixl_mem,
                                  nixlBackendMD* &out) override;
        nixl_status_t deregisterMem(nixlBackendMD *meta) override;

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

    private:
        gdsMtUtil gds_mt_utils_;
        std::unordered_map<int, nixlGdsMtMetadata*> gds_mt_file_map_;
        size_t thread_count_;
        std::unique_ptr<tf::Executor> executor_;
};
#endif
