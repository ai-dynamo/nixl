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

class nixlCudaPtrCtx {
public:
    enum memory_t {
        MEM_NONE,
        MEM_HOST,
        MEM_DEV,
        MEM_VMM_HOST,
        MEM_VMM_DEV,
        MEM_INVALID,
    } ;

protected:
    void *address;
    memory_t mem_type;
    uint64_t devId;

    /* To be used in derived classes */
    inline virtual bool
    internalCmp(const nixlCudaPtrCtx &rhs) {
        return true;
    }

public:
    nixlCudaPtrCtx(void *addr) : address(addr),
                                 mem_type(MEM_HOST),
                                 devId(0)
    { /* Empty body */ }

    virtual ~nixlCudaPtrCtx() = default;

    memory_t getMemType() {
        return mem_type;
    }

    uint64_t getDevId() {
        return devId;
    }

    virtual nixl_status_t setMemCtx() {
        // no-op for non-CUDA case
        return NIXL_SUCCESS;
    }

    virtual nixl_status_t unsetMemCtx() {
        // no-op for non-CUDA case
        return NIXL_SUCCESS;
    }

    bool operator==(const nixlCudaPtrCtx &rhs) {
        return (mem_type == rhs.mem_type) &&
                internalCmp(rhs);
    }


    static bool vramIsSupported();
    static std::unique_ptr<nixlCudaPtrCtx> nixlCudaPtrCtxInit(void *address);
};
