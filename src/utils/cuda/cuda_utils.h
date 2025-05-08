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

/* forward declaration for the internal CUDA data structure */

#include <nixl.h>
#include <memory>

/****************************************
 * Pointer Context
*****************************************/

class nixlCudaMemCtx {
public:
    enum memory_t {
        MEM_NONE,
        MEM_HOST,
        MEM_DEV,
        MEM_VMM_HOST,
        MEM_VMM_DEV,
    } ;
protected:
    memory_t memType;
    uint64_t _devId;
public:
    nixlCudaMemCtx() : memType(MEM_NONE), _devId(0)
    {  }

    virtual ~nixlCudaMemCtx() = default;

    memory_t getMemType() {
        return memType;
    }

    uint64_t getDevId() {
        return _devId;
    }

    virtual nixl_status_t set() {
        // no-op for non-CUDA case
        return NIXL_SUCCESS;
    }

    virtual nixl_status_t enableAddr(const void *address, uint64_t chkDevId) {
        return NIXL_SUCCESS;
    }

    static bool vramIsSupported();
    static std::unique_ptr<nixlCudaMemCtx> nixlCudaMemCtxInit();
};
