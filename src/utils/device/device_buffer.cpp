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
#include "device/device_buffer.h"

#include <cuda_runtime.h>

#include "common/nixl_log.h"

nixl_status_t
nixlDeviceBufferAllocate(void **ptr, size_t size) noexcept {
    if (ptr == nullptr || size == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    *ptr = nullptr;
    if (cudaMalloc(ptr, size) != cudaSuccess) {
        *ptr = nullptr;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDeviceBufferCopyHostToDevice(void *dst, const void *src, size_t size) noexcept {
    if (dst == nullptr || src == nullptr || size == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDeviceBufferCopyDeviceToHost(void *dst, const void *src, size_t size) noexcept {
    if (dst == nullptr || src == nullptr || size == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

void
nixlDeviceBufferFree(void *ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
    if (cudaFree(ptr) != cudaSuccess) {
        NIXL_ERROR << "Failed to free device buffer";
    }
}
