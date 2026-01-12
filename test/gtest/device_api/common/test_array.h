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

#ifndef NIXL_DEVICE_API_TEST_ARRAY_H
#define NIXL_DEVICE_API_TEST_ARRAY_H

#include <cuda_runtime.h>
#include <nixl.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

template<typename elementType> class testArray {
public:
    explicit testArray(size_t count, nixl_mem_t mem_type = VRAM_SEG)
        : count_{count},
          mem_type_{mem_type},
          ptr_{malloc(count, mem_type), deleter{mem_type}} {}

    void
    copyFromHost(const elementType *host_data, size_t count) {
        copy(ptr_.get(), host_data, count, cudaMemcpyHostToDevice);
    }

    void
    copyFromHost(const std::vector<elementType> &host_vector) {
        copyFromHost(host_vector.data(), host_vector.size());
    }

    void
    copyToHost(elementType *host_data, size_t count) const {
        copy(host_data, ptr_.get(), count, cudaMemcpyDeviceToHost);
    }

    [[nodiscard]] elementType *
    get() const noexcept {
        return ptr_.get();
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return count_;
    }

    [[nodiscard]] nixl_mem_t
    memType() const noexcept {
        return mem_type_;
    }

private:
    struct deleter {
        nixl_mem_t mem_type;

        void
        operator()(elementType *ptr) const {
            if (ptr == nullptr) {
                return;
            }

            switch (mem_type) {
            case VRAM_SEG:
                cudaFree(ptr);
                break;
            case DRAM_SEG:
                ::operator delete(ptr);
                break;
            default:
                cudaFree(ptr);
                break;
            }
        }
    };

    size_t count_;
    nixl_mem_t mem_type_;
    std::unique_ptr<elementType, deleter> ptr_;

    [[nodiscard]] static elementType *
    malloc(size_t count, nixl_mem_t mem_type) {
        elementType *ptr = nullptr;
        const size_t bytes = count * sizeof(elementType);
        cudaError_t err;

        switch (mem_type) {
        case VRAM_SEG:
            err = cudaMalloc(&ptr, bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("testArray: cudaMalloc failed: ") +
                                         cudaGetErrorString(err));
            }
            break;

        case DRAM_SEG:
            ptr = static_cast<elementType *>(::operator new(bytes, std::nothrow));
            if (ptr == nullptr) {
                throw std::runtime_error("testArray: operator new failed");
            }
            return ptr;

        default:
            throw std::runtime_error(std::string("testArray: unsupported memory type: ") +
                                     std::to_string(mem_type));
        }

        return ptr;
    }

    void
    copy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) const {
        if (count > count_) {
            throw std::out_of_range("testArray: copy count exceeds array size");
        }
        if (mem_type_ == DRAM_SEG) {
            std::memcpy(dst, src, count * sizeof(elementType));
            return;
        }

        const cudaError_t err = cudaMemcpy(dst, src, count * sizeof(elementType), kind);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("testArray: cudaMemcpy failed: ") +
                                     cudaGetErrorString(err));
        }
    }
};

#endif // NIXL_DEVICE_API_TEST_ARRAY_H
