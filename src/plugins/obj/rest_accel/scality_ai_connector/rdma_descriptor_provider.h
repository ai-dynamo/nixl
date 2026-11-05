/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_RDMA_DESCRIPTOR_PROVIDER_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_RDMA_DESCRIPTOR_PROVIDER_H

#include <cstddef>
#include <string>

#include "nixl_types.h"

/**
 * Data-plane side of the Scality AI Connector: registers local buffers for RDMA
 * and describes a registered byte range as an RDMA descriptor. The RDMA
 * descriptor travels to the server in an HTTP header; the server then performs
 * the one-sided RDMA read or write on that range itself.
 */
class iRdmaDescriptorProvider {
public:
    virtual ~iRdmaDescriptorProvider() = default;

    /** True once the transport is ready to register memory. */
    virtual bool
    isConnected() const = 0;

    /**
     * Register [ptr, ptr + size) for remote RDMA access.
     * @param dev_id CUDA device ordinal of a VRAM buffer, or -1 for host memory.
     *               Implementations may use it as a NIC-affinity hint.
     */
    virtual nixl_status_t
    registerMemory(void *ptr, size_t size, int dev_id) = 0;

    /** Release the registration made by registerMemory() at ptr. */
    virtual nixl_status_t
    deregisterMemory(void *ptr) = 0;

    /**
     * RDMA descriptor for [ptr, ptr + size), which must lie within a single
     * registration.
     * @return the RDMA descriptor, or an empty string if no registration covers
     *         the range.
     */
    virtual std::string
    makeDescriptor(void *ptr, size_t size) = 0;
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_RDMA_DESCRIPTOR_PROVIDER_H
