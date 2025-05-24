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

#include "cuda_utils.h"
#include <nixl_log.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cuda.h>

#endif

/****************************************
 * Helper routines
*****************************************/

#ifdef HAVE_CUDA

namespace {

enum memory_t {
    MEM_NONE,
    MEM_HOST,
    MEM_DEV,
    MEM_VMM_DEV,
};


class regularCtx {
protected:
    CUcontext m_context{nullptr};
public:

    regularCtx() = default;
    regularCtx(CUcontext c) : m_context(c)
    { }

    virtual ~regularCtx() = default;

    [[nodiscard]] nixl_status_t
    set() {
        if (nullptr == m_context) {
            return NIXL_ERR_NOT_FOUND;
        }

        CUresult result = cuCtxSetCurrent(m_context);
        if (CUDA_SUCCESS != result) {
            NIXL_ERROR << "cuCtxSetCurrent() failed. result = "
                    << result;
            return NIXL_ERR_UNKNOWN;
        }
        return NIXL_SUCCESS;
    }

    [[nodiscard]]
    virtual nixl_status_t
    pushIfNeed() {
        CUcontext context;
            const auto res = cuCtxGetCurrent(&context);
        if (res != CUDA_SUCCESS || context != nullptr) {
            return NIXL_SUCCESS;
        }

        if (m_context == nullptr) {
            return NIXL_ERR_NOT_FOUND;
        }

        return (CUDA_SUCCESS == cuCtxPushCurrent(m_context)) ?
                NIXL_IN_PROG : NIXL_ERR_NOT_POSTED ;
    }

    [[nodiscard]]
    nixl_status_t
    pop() {
        return (CUDA_SUCCESS == cuCtxPopCurrent(nullptr)) ?
            NIXL_SUCCESS : NIXL_ERR_UNKNOWN;
    }
};

class primaryCtx : public regularCtx{
    int m_ordinal;
    CUdevice m_device{CU_DEVICE_INVALID};

public:

    primaryCtx(int ordinal) : m_ordinal(ordinal)
    { }

    ~primaryCtx() override {
        if (m_context != nullptr) {
            cuDevicePrimaryCtxRelease(m_device);
        }
    }

    [[nodiscard]] nixl_status_t
    retain()
    {
        CUdevice device;

        auto result = cuDeviceGet(&device, m_ordinal);
        if (result != CUDA_SUCCESS) {
            NIXL_ERROR << "cuDeviceGet() failed. result = " << result;
            return NIXL_ERR_UNKNOWN;
        }

        unsigned int flags;
        int active;
        result = cuDevicePrimaryCtxGetState(device, &flags, &active);
        if (result != CUDA_SUCCESS) {
            NIXL_ERROR << "cuDevicePrimaryCtxGetState() failed. result = "
                    << result;
            return NIXL_ERR_UNKNOWN;
        }

        if (!active) {
            NIXL_ERROR << "No active context found for CUDA device " << m_ordinal;
            return NIXL_ERR_INVALID_PARAM;
        }

        result = cuDevicePrimaryCtxRetain(&m_context, device);
        if (result != CUDA_SUCCESS) {
            NIXL_ERROR << "cuDevicePrimaryCtxRetain() failed. result = "
                    << result;
            return NIXL_ERR_UNKNOWN;
        }

        return NIXL_SUCCESS;
    }

    [[nodiscard]]
    virtual nixl_status_t
    pushIfNeed() override {
        CUcontext context;
            const auto res = cuCtxGetCurrent(&context);
        if (res != CUDA_SUCCESS || context != nullptr) {
            return NIXL_SUCCESS;
        }

        if (m_context == nullptr) {
            return NIXL_ERR_NOT_FOUND;
        }

        return (CUDA_SUCCESS == cuCtxPushCurrent(m_context)) ?
                NIXL_IN_PROG : NIXL_ERR_NOT_POSTED ;
    }
};

[[nodiscard]] nixl_status_t
_queryVmmPtr(const void *address, memory_t &type, int &id)
{
    nixl_status_t ret = NIXL_ERR_NOT_FOUND;

#if HAVE_CUMEMRETAINALLOCATIONHANDLE
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle alloc_handle;

    /* Check if memory is allocated using VMM API and see if host memory needs
    * to be treated as pinned device memory */
    auto result = cuMemRetainAllocationHandle(&alloc_handle, (void*)address);
    if (result != CUDA_SUCCESS) {
        NIXL_TRACE << "cuMemRetainAllocationHandle() failed. result = "
                << result;
        return NIXL_ERR_NOT_FOUND;
    }
    // TODO: set the call to cuMemRelease when leaving the scope to avoid GOTO

    result = cuMemGetAllocationPropertiesFromHandle(&prop, alloc_handle);
    if (result != CUDA_SUCCESS) {
        NIXL_DEBUG << "cuMemGetAllocationPropertiesFromHandle() failed. result = "
                << result;
        ret = NIXL_ERR_UNKNOWN;
        goto err;
    }

    id = prop.location.id;
    switch (prop.location.type) {
    case CU_MEM_LOCATION_TYPE_DEVICE:
        type = MEM_VMM_DEV;
        ret = NIXL_SUCCESS;
        break;
    default:
        NIXL_DEBUG << "Unsupported VMM memory type: " << prop.location.type;
        ret = NIXL_ERR_INVALID_PARAM;
        goto err;
    }

err:
    result = cuMemRelease(alloc_handle);
    if (CUDA_SUCCESS != result) {
        NIXL_DEBUG << "cuMemRelease() failed. result = "
                << result;
        if (NIXL_SUCCESS == ret) {
            ret = NIXL_ERR_UNKNOWN;
        }
    }

#endif
    return ret;
}

[[nodiscard]] nixl_status_t
_queryCudaPtr(const void *address, memory_t &type, int &outOrdinal, CUcontext &newCtx)
{
    constexpr int numAttrs = 4;
    CUpointer_attribute attr_type[numAttrs];
    void *attr_data[numAttrs];
    CUmemorytype cudaMemType = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
    CUresult result;
    int ordinal;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &cudaMemType;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &ordinal;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &newCtx;

    result = cuPointerGetAttributes(numAttrs, attr_type, attr_data, (CUdeviceptr)address);
    if (CUDA_SUCCESS != result) {
        NIXL_ERROR << "cuPointerGetAttributes() failed. result = "
                << result;
        return NIXL_ERR_NOT_FOUND;
    }

    outOrdinal = ordinal;
    switch(cudaMemType) {
    case CU_MEMORYTYPE_DEVICE:
        type = MEM_DEV;
        break;
    case CU_MEMORYTYPE_HOST:
        type = MEM_HOST;
    case CU_MEMORYTYPE_ARRAY:
        NIXL_ERROR << "CU_MEMORYTYPE_ARRAY memory type is not supported";
        return NIXL_ERR_INVALID_PARAM;
    default:
        NIXL_TRACE << "Unknown CUDA memory type" << cudaMemType;
        return NIXL_ERR_NOT_FOUND;
    }

    if (is_managed) {
        NIXL_ERROR << "CUDA managed memory is not supported";
        return NIXL_ERR_INVALID_PARAM;
    }

    return NIXL_SUCCESS;
}

} // end of anonymous namespace

namespace nixl::cuda {

/****************************************
 * CUDA memory context class
*****************************************/

class memCtxImpl : public memCtx {
    static constexpr int defaultCudaDeviceOrdinal = 0;
    int ordinal { -1 };
    std::unique_ptr<regularCtx> ctx;

public:

    memCtxImpl() {
        // Create default context for push/pop operations
        // Do not retain it unless needed
        ctx = std::make_unique<primaryCtx>(defaultCudaDeviceOrdinal);
    }

    // TODO: can it be default?
    ~memCtxImpl() override
    { }

    [[nodiscard]]
    nixl_status_t initFromAddr(const void *address, uint64_t chkDevId) override;

    [[nodiscard]]
    nixl_status_t set() override {
        if (0 > ordinal) {
            // Not context was set - use empry op
            // Alternatively we can set primary device context
            return NIXL_SUCCESS;
        }
        return ctx->set();
    }

    [[nodiscard]]
    nixl_status_t pushIfNeed() override {
        return ctx->pushIfNeed();
    }

    [[nodiscard]]
    nixl_status_t pop() override {
        return ctx->pop();
    }

};

/****************************************
 * CUDA memCtx class implementation
*****************************************/

nixl_status_t
memCtxImpl::initFromAddr(const void *address, uint64_t chkDevId)
{
    memory_t addrMemType = MEM_NONE;
    CUcontext newCtx;
    int newOrdinal;

    nixl_status_t status = _queryVmmPtr(address, addrMemType, newOrdinal);
    if (status == NIXL_ERR_NOT_FOUND) {
        status = _queryCudaPtr(address, addrMemType, newOrdinal, newCtx);
        if (status == NIXL_ERR_NOT_FOUND) {
            addrMemType = MEM_HOST;
            status = NIXL_SUCCESS;
        } else if (NIXL_SUCCESS != status) {
            NIXL_ERROR << "CUDA Query failed with status = "
                        << status;
            // TODO use nixlEnumStrings::statusStr(status); once circ dep between libnixl & utils is resolved
        }
    } else if (NIXL_SUCCESS != status) {
        NIXL_ERROR << "VMM Query failed with status = "
                << status;
        // TODO use nixlEnumStrings::statusStr(status); once circ dep between libnixl & utils is resolved
    }

    if (status != NIXL_SUCCESS) {
        return status;
    }

    if (MEM_HOST == addrMemType) {
        // Host memory doesn't have any context
        return NIXL_SUCCESS;
    }

    if (static_cast<uint64_t>(newOrdinal) != chkDevId) {
        NIXL_DEBUG << "Mismatch between the expected and actual CUDA device id";
        NIXL_DEBUG << "Expect: " << chkDevId << ", have: " << newOrdinal;
        return NIXL_ERR_MISMATCH;
    }

    // The context was already set
    if (0 <= ordinal) {
        // UCX up to 1.18 only supports one device context per
        // UCP context. Enforce that!
        if (ordinal != newOrdinal) {
            status = NIXL_ERR_MISMATCH;
        }
        return status;
    }


    // Initialize the context
    switch(addrMemType) {
    case MEM_VMM_DEV:
    case MEM_DEV: {
        // Try using Primary context whenever possible
        auto ctxP = std::make_unique<primaryCtx>(newOrdinal);
        status = ctxP->retain();
        if (NIXL_SUCCESS == status) {
            ctx = std::move(ctxP);
        } else if (MEM_DEV == addrMemType) {
            ctx = std::make_unique<regularCtx>(newCtx);
        } else {
            return status;
        }
        status = NIXL_IN_PROG;
        ordinal = newOrdinal;
        break;
    }
    case MEM_NONE:
    case MEM_HOST:
        NIXL_ERROR << "Unknown issue - memType is invalid: " <<  addrMemType;
        status = NIXL_ERR_INVALID_PARAM;
        break;
    }
    return status;
}

#endif

std::shared_ptr<memCtx>
makeMemCtx()
{
    // Environment fixup
    if (getenv("NIXL_DISABLE_CUDA_ADDR_WA")) {
        // If the workarounf is disabled - return the dummy class
        NIXL_INFO << "WARNING: disabling CUDA address workaround";
        return std::make_shared<memCtx>();
    } else {
#ifdef HAVE_CUDA
        return std::make_shared<memCtxImpl>();
#else
     return std::make_shared<memCtx>();
#endif
    }
}

uint32_t numDevices()
{
    int n_vram_dev = 0;
#ifdef HAVE_CUDA
    const auto result = cudaGetDeviceCount(&n_vram_dev);
    if (result != cudaSuccess) {
        NIXL_ERROR << "cudaGetDeviceCount failed: result = " << result;
    }
#endif
    return n_vram_dev;
}

} // nixl::cuda
