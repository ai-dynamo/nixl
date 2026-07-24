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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_RDMA_TOKEN_CLIENT_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_RDMA_TOKEN_CLIENT_H

#include <cuobjclient.h>

#include <memory>

/**
 * Abstract RDMA token client interface.
 *
 * Implementation:
 *  - CuObjRdmaTokenClient: thin wrapper around NVIDIA cuObjClient (DC transport)
 */
class iRdmaTokenClient {
public:
    /** Return true when the transport layer is ready for transfers. */
    [[nodiscard]] virtual bool
    isConnected() const = 0;

    /**
     * Register a memory region and obtain an RDMA descriptor (token).
     * @return CU_OBJ_SUCCESS on success.
     */
    [[nodiscard]] virtual cuObjErr_t
    cuMemObjGetDescriptor(void *ptr, size_t size) = 0;

    /**
     * Deregister a previously registered memory region.
     * @return CU_OBJ_SUCCESS on success.
     */
    [[nodiscard]] virtual cuObjErr_t
    cuMemObjPutDescriptor(void *ptr) = 0;

    /**
     * Initiate a GET (read) transfer, calls the user-supplied get callback
     * with the RDMA token so that the remote side can perform the data transfer.
     */
    [[nodiscard]] virtual ssize_t
    cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset = 0) = 0;

    /**
     * Initiate a PUT (write) transfer; calls the user-supplied put callback
     * with the RDMA token so that the remote side can perform the data transfer.
     */
    [[nodiscard]] virtual ssize_t
    cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset = 0) = 0;

    /**
     * Extract the user context pointer from cuObjClient's opaque handle.
     */
    [[nodiscard]] static void *
    getCtx(const void *handle) {
        if (!handle) {
            return nullptr;
        }
        return cuObjClient::getCtx(handle);
    }

protected:
    // Owned only through shared_ptr (never deleted via an iRdmaTokenClient*), so
    // a protected non-virtual destructor suffices.
    ~iRdmaTokenClient() = default;
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_RDMA_TOKEN_CLIENT_H
