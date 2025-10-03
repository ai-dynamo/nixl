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

#ifndef _CUDA_PTR_CUH
#define _CUDA_PTR_CUH

#include <cuda_runtime.h>

template<typename T> class CudaPtr {
public:
    explicit CudaPtr(T **ptr) : ptr_(ptr) {
        cudaMalloc(reinterpret_cast<void **>(ptr_), sizeof(T));
        cudaMemset(*ptr_, 0, sizeof(T));
    }

    ~CudaPtr() {
        if (ptr_ && *ptr_) {
            cudaFree(*ptr_);
            *ptr_ = nullptr;
        }
    }

    CudaPtr(const CudaPtr &) = delete;
    CudaPtr &
    operator=(const CudaPtr &) = delete;

    CudaPtr(CudaPtr &&other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    CudaPtr &
    operator=(CudaPtr &&other) noexcept {
        if (this != &other) {
            if (ptr_ && *ptr_) {
                cudaFree(*ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T *
    get() const {
        return ptr_ ? *ptr_ : nullptr;
    }

private:
    T **ptr_;
};

#endif // _CUDA_PTR_CUH
