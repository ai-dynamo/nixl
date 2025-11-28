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

#ifndef NIXL_DEVICE_API_TEST_MEM_BUFFER_H
#define NIXL_DEVICE_API_TEST_MEM_BUFFER_H

#include <cuda_runtime.h>
#include <nixl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

class MemBuffer {
public:
    MemBuffer(size_t size, nixl_mem_t mem_type)
        : ptr_(allocate(size, mem_type), [mem_type](void *ptr) { release(ptr, mem_type); }),
          size_(size) {}

    [[nodiscard]] uintptr_t
    toUintptr() const {
        return reinterpret_cast<uintptr_t>(get());
    }

    [[nodiscard]] void *
    get() const {
        return ptr_.get();
    }

    [[nodiscard]] const void *
    getConst() const {
        return ptr_.get();
    }

    // Explicit conversion operators to prevent implicit conversion bugs
    [[nodiscard]] explicit
    operator uintptr_t() const {
        return toUintptr();
    }

    [[nodiscard]] explicit
    operator void *() const {
        return get();
    }

    [[nodiscard]] explicit
    operator const void *() const {
        return getConst();
    }

    [[nodiscard]] size_t
    getSize() const {
        return size_;
    }

private:
    [[nodiscard]] static void *
    allocate(size_t size, nixl_mem_t) {
        void *ptr = nullptr;
        const cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to allocate CUDA memory: ") +
                                     cudaGetErrorString(err));
        }
        return ptr;
    }

    static void
    release(void *ptr, nixl_mem_t) noexcept {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    std::shared_ptr<void> ptr_;
    size_t size_;
};

#endif // NIXL_DEVICE_API_TEST_MEM_BUFFER_H
