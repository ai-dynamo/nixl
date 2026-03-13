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

#ifndef NIXL_TEST_GTEST_DEVICE_API_MEM_BUFFER_CUH
#define NIXL_TEST_GTEST_DEVICE_API_MEM_BUFFER_CUH

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <stdexcept>

namespace nixl::gpu {
class memBuffer {
public:
    explicit memBuffer(size_t size, const void *src = nullptr)
        : size_{size},
          ptr_{malloc(size), free} {
        if (src) {
            if (cudaMemcpy(ptr_.get(), src, size, cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("Failed to memcpy from host to device");
            }
        } else {
            if (cudaMemset(ptr_.get(), 0, size) != cudaSuccess) {
                throw std::runtime_error("Failed to memset memory");
            }
        }
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return size_;
    }

    [[nodiscard]] operator uintptr_t() const {
        return reinterpret_cast<uintptr_t>(ptr_.get());
    }

    [[nodiscard]] bool
    operator==(const void *rhs) const {
        const auto host_buffer = std::make_unique<std::byte[]>(size_);
        copyToHost(host_buffer.get());
        return std::memcmp(host_buffer.get(), rhs, size_) == 0;
    }

    [[nodiscard]] bool
    operator==(const memBuffer &rhs) const {
        const auto host_buffer = std::make_unique<std::byte[]>(size_);
        copyToHost(host_buffer.get());
        return rhs == host_buffer.get();
    }

private:
    [[nodiscard]] static void *
    malloc(size_t size) {
        void *ptr;
        if (cudaMalloc(&ptr, size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate memory");
        }
        return ptr;
    }

    static void
    free(void *ptr) {
        cudaFree(ptr);
    }

    void
    copyToHost(void *dst) const {
        if (cudaMemcpy(dst, ptr_.get(), size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("Failed to memcpy from device to host");
        }
    }

    size_t size_;
    std::unique_ptr<void, void (*)(void *)> ptr_;
};
} // namespace nixl::gpu
#endif // NIXL_TEST_GTEST_DEVICE_API_MEM_BUFFER_CUH
