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

#ifndef NIXL_TEST_GTEST_DEVICE_API_CUDA_PTR_CUH
#define NIXL_TEST_GTEST_DEVICE_API_CUDA_PTR_CUH

#include <cuda_runtime.h>

#include <stdexcept>
#include <memory>

namespace nixl::gpu {
template<typename T> class cudaPtr {
public:
    cudaPtr(bool managed = false) : ptr_(calloc(managed), deleter{}) {}

    [[nodiscard]] T *
    get() const noexcept {
        return ptr_.get();
    }

    [[nodiscard]] T
    operator*() const {
        T value;
        if (cudaMemcpy(&value, ptr_.get(), sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("Failed to memcpy from device to host");
        }
        return value;
    }

private:
    struct deleter {
        void
        operator()(T *ptr) const {
            cudaFree(ptr);
        }
    };

    [[nodiscard]] static T *
    calloc(bool managed) {
        T *ptr = nullptr;
        cudaError_t status;
        if (managed) {
            status = cudaMallocManaged(&ptr, sizeof(T));
        } else {
            status = cudaMalloc(&ptr, sizeof(T));
        }
        if (status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate memory");
        }
        if (cudaMemset(ptr, 0, sizeof(T)) != cudaSuccess) {
            throw std::runtime_error("Failed to memset memory");
        }
        return ptr;
    }

    std::unique_ptr<T, deleter> ptr_;
};

template<typename T>
bool
operator==(const cudaPtr<T> &lhs, const cudaPtr<T> &rhs) {
    return *lhs == *rhs;
}

template<typename T>
bool
operator==(const cudaPtr<T> &lhs, const T &rhs) {
    return *lhs == rhs;
}
} // namespace nixl::gpu
#endif // NIXL_TEST_GTEST_DEVICE_API_CUDA_PTR_CUH
