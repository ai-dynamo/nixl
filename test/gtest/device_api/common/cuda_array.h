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

#ifndef NIXL_DEVICE_API_TEST_CUDA_ARRAY_H
#define NIXL_DEVICE_API_TEST_CUDA_ARRAY_H

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

template<typename T> class CudaArray {
public:
    explicit CudaArray(size_t count) : count_(count) {
        const cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CudaArray: cudaMalloc failed: ") +
                                     cudaGetErrorString(err));
        }
    }

    ~CudaArray() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    CudaArray(const CudaArray &) = delete;
    CudaArray &
    operator=(const CudaArray &) = delete;

    void
    copyFromHost(const T *host_data, size_t count) {
        if (count > count_) {
            throw std::out_of_range("CudaArray: copy count exceeds array size");
        }
        const cudaError_t err =
            cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CudaArray: cudaMemcpy from host failed: ") +
                                     cudaGetErrorString(err));
        }
    }

    void
    copyFromHost(const std::vector<T> &host_vector) {
        copyFromHost(host_vector.data(), host_vector.size());
    }

    void
    copyToHost(T *host_data, size_t count) const {
        if (count > count_) {
            throw std::out_of_range("CudaArray: copy count exceeds array size");
        }
        const cudaError_t err =
            cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CudaArray: cudaMemcpy to host failed: ") +
                                     cudaGetErrorString(err));
        }
    }

    [[nodiscard]] T *
    get() const noexcept {
        return ptr_;
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return count_;
    }

private:
    T *ptr_;
    size_t count_;
};

#endif // NIXL_DEVICE_API_TEST_CUDA_ARRAY_H
