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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CUOBJ_RDMA_TOKEN_CLIENT_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CUOBJ_RDMA_TOKEN_CLIENT_H

#include "rdma_token_client.h"

/**
 * Thin wrapper around NVIDIA cuObjClient (DC transport).
 */
class CuObjRdmaTokenClient : public iRdmaTokenClient {
public:
    explicit CuObjRdmaTokenClient(CUObjOps_t &ops);
    ~CuObjRdmaTokenClient() = default;

    bool
    isConnected() const override;
    cuObjErr_t
    cuMemObjGetDescriptor(void *ptr, size_t size) override;
    cuObjErr_t
    cuMemObjPutDescriptor(void *ptr) override;
    ssize_t
    cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset = 0) override;
    ssize_t
    cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset = 0) override;

private:
    std::shared_ptr<cuObjClient> inner_;
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CUOBJ_RDMA_TOKEN_CLIENT_H
