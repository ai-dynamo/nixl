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

#ifndef __GDS_BACKEND_H
#define __GDS_BACKEND_H

#include <nixl.h>
#include <nixl_types.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <fcntl.h>
#include <list>
#include <vector>
#include <mutex>
#include "gds_utils.h"
#include "backend/backend_engine.h"

class nixlGdsMetadata : public nixlBackendMD {
    public:
        gdsFileHandle handle;
        gdsMemBuf buf;
        nixl_mem_t type;

        nixlGdsMetadata() : nixlBackendMD(true) { }
        ~nixlGdsMetadata() { }
};

class GdsTransferRequestH {
    public:
        void*           addr;
        size_t          size;
        size_t          file_offset;
        CUfileHandle_t  fh;
        CUfileOpcode_t  op;

        // Default constructor
        GdsTransferRequestH() {
            addr = nullptr;
            size = 0;
            file_offset = 0;
            fh = nullptr;
            op = CUFILE_READ;
        }

        // Constructor with parameters
        GdsTransferRequestH(void* a, size_t s, size_t offset,
			    CUfileHandle_t handle, CUfileOpcode_t operation) {
            addr = a;
            size = s;
            file_offset = offset;
            fh = handle;
            op = operation;
        }
};

class nixlGdsBackendReqH : public nixlBackendReqH {
    public:
        std::vector<GdsTransferRequestH> request_list;
        std::vector<nixlGdsIOBatch*> batch_io_list;
        bool needs_prep;

        nixlGdsBackendReqH() {
            needs_prep = true;
        }
        ~nixlGdsBackendReqH() {
            for (auto* batch : batch_io_list) {
                delete batch;
            }
            batch_io_list.clear();
        }
};

class nixlGdsEngine : public nixlBackendEngine {
    private:
        gdsUtil *gds_utils;
        std::unordered_map<int, gdsFileHandle> gds_file_map;

        mutable std::mutex batch_pool_lock;
        mutable std::list<nixlGdsIOBatch*> batch_pool;
        unsigned int batch_pool_size;  // Renamed from pool_size
        unsigned int batch_limit;      // Added for configurable batch limit
        unsigned int max_request_size; // Added for configurable request size

        nixlGdsIOBatch* getBatchFromPool(unsigned int size) const;
        void returnBatchToPool(nixlGdsIOBatch* batch) const;
        nixl_status_t createAndSubmitBatch(const std::vector<GdsTransferRequestH>& requests,
                                           size_t start_idx, size_t batch_size,
                                           std::vector<nixlGdsIOBatch*>& batch_list) const;
        nixl_status_t createBatches(const nixl_xfer_op_t &operation,
                                   const nixl_meta_dlist_t &local,
                                   const nixl_meta_dlist_t &remote,
                                   nixlGdsBackendReqH* gds_handle);

    public:
        nixlGdsEngine(const nixlBackendInitParams* init_params);
        ~nixlGdsEngine();

        // File operations - target is the distributed FS
        // So no requirements to connect to target.
        // Just treat it locally.
        bool supportsNotif() const {
            return false;
        }
        bool supportsRemote() const {
            return false;
        }
        bool supportsLocal() const {
            return true;
        }

        nixl_mem_list_t getSupportedMems() const {
            nixl_mem_list_t mems;
            mems.push_back(DRAM_SEG);
            mems.push_back(VRAM_SEG);
            mems.push_back(FILE_SEG);
            return mems;
        }

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
                                 nixlBackendMD* &out);
        nixl_status_t deregisterMem(nixlBackendMD *meta);

        nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH* &handle,
                              const nixl_opt_b_args_t* opt_args=nullptr) const;

        nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH* &handle,
                              const nixl_opt_b_args_t* opt_args=nullptr) const;

        nixl_status_t checkXfer(nixlBackendReqH* handle) const;
        nixl_status_t releaseReqH(nixlBackendReqH* handle) const;

        nixl_status_t
        queryMem(const nixl_reg_dlist_t &descs,
                 std::vector<nixl_query_resp_t> &resp) const override;
};
#endif
