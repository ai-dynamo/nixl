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

#ifndef NIXL_SRC_PLUGINS_KV_KV_BACKEND_H
#define NIXL_SRC_PLUGINS_KV_KV_BACKEND_H

#include "backend/backend_engine.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Synchronous key-value store contract used by KV backend implementations.
 *
 * Implementations store byte ranges under string keys. The caller retains
 * ownership of input and output buffers; implementations must copy data during
 * the call and must not retain raw buffer pointers after returning.
 */
class iKVStore {
public:
    virtual ~iKVStore() = default;

    /**
     * @brief Store a byte range under a key.
     * @param key Backend key identifying the object.
     * @param data Pointer to the data to store. Must be valid for len bytes.
     * @param len Number of bytes to store.
     * @return NIXL_SUCCESS on success, otherwise an error status.
     */
    virtual nixl_status_t
    put(std::string_view key, const uint8_t *data, size_t len) = 0;

    /**
     * @brief Load a byte range from a key into a caller-owned buffer.
     * @param key Backend key identifying the object.
     * @param buffer Destination buffer. Must be valid for len bytes.
     * @param len Destination buffer capacity in bytes.
     * @param bytes_read Set to the number of bytes copied into buffer.
     * @return NIXL_SUCCESS on success, otherwise an error status.
     */
    virtual nixl_status_t
    get(std::string_view key, uint8_t *buffer, size_t len, size_t &bytes_read) const = 0;

    /**
     * @brief Check whether a key exists in the backing store.
     * @param key Backend key identifying the object.
     * @return true when the key exists, false otherwise.
     */
    virtual bool
    exists(std::string_view key) const = 0;
};

/**
 * @brief Storage-specific implementation contract behind nixlKVEngine.
 *
 * The public nixlKVEngine wrapper delegates backend operations to one owned
 * implementation object. Memory metadata returned from registerMem() remains
 * backend-owned in the sense that callers must release it through
 * deregisterMem(); callers must not delete it directly. Request handles
 * returned from prepXfer() remain valid until releaseReqH() is called exactly
 * once, even after postXfer() and checkXfer() complete.
 */
class nixlKVEngineImpl {
public:
    virtual ~nixlKVEngineImpl() = default;

    /**
     * @brief Return memory segment types supported by this implementation.
     */
    virtual nixl_mem_list_t
    getSupportedMems() const = 0;

    /**
     * @brief Register backend metadata for a memory or object descriptor.
     * @param mem Descriptor carrying address, length, device id, and optional key/path metadata.
     * @param nixl_mem Segment type being registered.
     * @param out Set to backend metadata on success, or nullptr when no metadata is required.
     * @return NIXL_SUCCESS on success, otherwise an error status.
     *
     * When out is non-null, the caller must pass it back to deregisterMem().
     */
    virtual nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) = 0;

    /**
     * @brief Release metadata previously returned by registerMem().
     * @param meta Metadata pointer returned by registerMem(), or nullptr.
     * @return NIXL_SUCCESS on success, otherwise an error status.
     */
    virtual nixl_status_t
    deregisterMem(nixlBackendMD *meta) = 0;

    /**
     * @brief Query backend visibility or availability for registered descriptors.
     * @param descs Descriptors to query.
     * @param resp Filled with one response per descriptor.
     * @return NIXL_SUCCESS when all queries complete, otherwise an error status.
     */
    virtual nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const = 0;

    /**
     * @brief Validate and prepare a transfer request handle.
     * @param operation Transfer operation, such as NIXL_READ or NIXL_WRITE.
     * @param local Local descriptors participating in the transfer.
     * @param remote Remote descriptors participating in the transfer.
     * @param remote_agent Remote agent name supplied by the caller.
     * @param local_agent Local agent name for validation.
     * @param handle Set to a new request handle on success; set to nullptr on failure.
     * @param opt_args Optional backend arguments.
     * @return NIXL_SUCCESS on successful preparation, otherwise an error status.
     *
     * The caller owns the returned handle lifecycle and must release it with
     * releaseReqH() exactly once.
     */
    virtual nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const = 0;

    /**
     * @brief Start work for a prepared transfer request.
     * @param operation Transfer operation matching prepXfer().
     * @param local Local descriptors participating in the transfer.
     * @param remote Remote descriptors participating in the transfer.
     * @param remote_agent Remote agent name supplied by the caller.
     * @param handle Request handle returned by prepXfer().
     * @param opt_args Optional backend arguments.
     * @return NIXL_IN_PROG for asynchronous progress, NIXL_SUCCESS when complete, or an error.
     *
     * The handle remains owned by the caller and must still be released with
     * releaseReqH().
     */
    virtual nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const = 0;

    /**
     * @brief Poll completion status for a prepared transfer request.
     * @param handle Request handle returned by prepXfer().
     * @return NIXL_IN_PROG while work remains, NIXL_SUCCESS on completion, or an error.
     */
    virtual nixl_status_t
    checkXfer(nixlBackendReqH *handle) const = 0;

    /**
     * @brief Destroy a request handle returned by prepXfer().
     * @param handle Request handle to release.
     * @return NIXL_SUCCESS on success, otherwise an error status.
     */
    virtual nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const = 0;
};

/**
 * @brief NIXL backend wrapper for key-value storage plugins.
 *
 * nixlKVEngine owns a concrete nixlKVEngineImpl through a unique_ptr and
 * delegates all storage-specific behavior to it. Concrete KV plugins derive
 * from this wrapper, pass their implementation to the protected constructor,
 * and inherit the NIXL backend interface glue.
 */
class nixlKVEngine : public nixlBackendEngine {
public:
    ~nixlKVEngine() override;

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
    getSupportedMems() const override;

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
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        if (input == nullptr) {
            output = nullptr;
            return NIXL_ERR_INVALID_PARAM;
        }
        output = input;
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

protected:
    nixlKVEngine(const nixlBackendInitParams *init_params, std::unique_ptr<nixlKVEngineImpl> impl);

private:
    std::unique_ptr<nixlKVEngineImpl> impl_;
};

#endif // NIXL_SRC_PLUGINS_KV_KV_BACKEND_H
