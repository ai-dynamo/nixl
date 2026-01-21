/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OBJ_BACKEND_H
#define OBJ_BACKEND_H

#include "obj_executor.h"
#include <string>
#include <memory>
#include <unordered_map>
#include "backend/backend_engine.h"

using put_object_callback_t = std::function<void(bool success)>;
using get_object_callback_t = std::function<void(bool success)>;

/**
 * Abstract interface for S3 client operations.
 * Provides async operations for PutObject and GetObject.
 */
class iS3Client {
public:
    virtual ~iS3Client() = default;

    /**
     * Set the executor for async operations.
     * @param executor The executor to use for async operations
     */
    virtual void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) = 0;

    /**
     * Asynchronously put an object to S3.
     * @param key The object key
     * @param data_ptr Pointer to the data to upload
     * @param data_len Length of the data in bytes
     * @param offset Offset within the object
     * @param callback Callback function to handle the result
     */
    virtual void
    putObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   put_object_callback_t callback) = 0;

    /**
     * Asynchronously get an object from S3.
     * @param key The object key
     * @param data_ptr Pointer to the buffer to store the downloaded data
     * @param data_len Maximum length of data to read
     * @param offset Offset within the object to start reading from
     * @param callback Callback function to handle the result
     */
    virtual void
    getObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) = 0;

    /**
     * Check if the object exists.
     * @param key The object key
     * @return true if the object exists, false otherwise
     */
    virtual bool
    checkObjectExists(std::string_view key) = 0;
};

class nixlObjEngine : public nixlBackendEngine {
public:
    nixlObjEngine(const nixlBackendInitParams *init_params);
    nixlObjEngine(const nixlBackendInitParams *init_params,
                  std::shared_ptr<iS3Client> s3_client,
                  std::shared_ptr<iS3Client> s3_client_crt = nullptr);
    virtual ~nixlObjEngine();

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

private:
    std::shared_ptr<asioThreadPoolExecutor> executor_;
    std::shared_ptr<iS3Client> s3Client_; // Standard S3 client for small objects
    std::shared_ptr<iS3Client> s3ClientCrt_; // S3 CRT client for large objects
    std::unordered_map<uint64_t, std::string> devIdToObjKey_;
    size_t crtMinLimit_; // Minimum size threshold to use CRT client
};

#endif // OBJ_BACKEND_H
