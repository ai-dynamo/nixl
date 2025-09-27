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

#ifndef _MEM_BUFFER_CUH
#define _MEM_BUFFER_CUH

#include <memory>
#include <cuda_runtime.h>
#include "nixl.h"

class MemBuffer : std::shared_ptr<void> {
public:
    MemBuffer(size_t size, nixl_mem_t mem_type)
        : std::shared_ptr<void>(allocate(size, mem_type),
                                [mem_type](void *ptr) { release(ptr, mem_type); }),
          size(size) {}

    operator uintptr_t() const {
        return reinterpret_cast<uintptr_t>(get());
    }

    operator void *() const {
        return get();
    }

    operator const void *() const {
        return get();
    }

    size_t
    getSize() const {
        return size;
    }

private:
    static void *
    allocate(size_t size, nixl_mem_t mem_type) {
        void *ptr;
        return cudaSuccess == cudaMalloc(&ptr, size) ? ptr : nullptr;
    }

    static void
    release(void *ptr, nixl_mem_t mem_type) {
        cudaFree(ptr);
    }

    size_t size;
};

#endif // _MEM_BUFFER_CUH
