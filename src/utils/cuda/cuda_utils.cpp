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


// TODO: remove:
// #define HAVE_CUDA 1
// #define  HAVE_CUMEMRETAINALLOCATIONHANDLE 1
// #define HAVE_DECL_CU_MEM_LOCATION_TYPE_HOST 1


/****************************************
 * CUDA nixlCudaPtr class
*****************************************/

#ifdef HAVE_CUDA

class nixlCudaMemCtxImpl : public nixlCudaMemCtx {
private:
    CUcontext ctx;

    nixl_status_t queryVmm(const void *address, memory_t &type, uint64_t &id);
    nixl_status_t retainVmmCudaCtx(uint64_t id, CUcontext &newCtx) const;
    nixl_status_t releaseVmmCudaCtx(uint64_t id) const;
    nixl_status_t queryCuda(const void *address, memory_t &type, uint64_t &id,
                            CUcontext &newCtx);

public:

    nixlCudaMemCtxImpl() : nixlCudaMemCtx()
    {    }

    ~nixlCudaMemCtxImpl() override {
        if (MEM_VMM_DEV == memType) {
            nixl_status_t status = releaseVmmCudaCtx(_devId);
            if (NIXL_SUCCESS != status) {
                NIXL_ERROR << "Failed to release CUDA context";
            }
        }
    }

    nixl_status_t enableAddr(const void *address, uint64_t chkDevId) override;

    nixl_status_t set() override;
};

#endif

/****************************************
 * Static nixlCudaPtr functions
*****************************************/

#ifdef HAVE_CUDA

#define NIXL_CUDA_PTR_CTX_CLASS nixlCudaMemCtxImpl
#define NIXL_CUDA_PTR_CTX_VRAM_SUPPORT true

#else

#define NIXL_CUDA_PTR_CTX_CLASS nixlCudaMemCtx
#define NIXL_CUDA_PTR_CTX_VRAM_SUPPORT false

#endif

bool nixlCudaMemCtx::vramIsSupported()
{
    return NIXL_CUDA_PTR_CTX_VRAM_SUPPORT;
}

std::unique_ptr<nixlCudaMemCtx>
nixlCudaMemCtx::nixlCudaMemCtxInit()
{
    // Environment fixup
    if (getenv("NIXL_DISABLE_CUDA_ADDR_WA")) {
        // If the workarounf is disabled - return the dummy class
        NIXL_INFO << "WARNING: disabling CUDA address workaround";
        return std::make_unique<nixlCudaMemCtx>();
    } else {
        return std::make_unique<NIXL_CUDA_PTR_CTX_CLASS>();
    }
}


/****************************************
 * CUDA nixlCudaPtr class implementation
*****************************************/

#ifdef HAVE_CUDA

nixl_status_t
nixlCudaMemCtxImpl::queryVmm(const void *address, memory_t &type, uint64_t &id)
{
    nixl_status_t ret = NIXL_ERR_NOT_FOUND;

#if HAVE_CUMEMRETAINALLOCATIONHANDLE
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle alloc_handle;
    CUresult result;

    /* Check if memory is allocated using VMM API and see if host memory needs
     * to be treated as pinned device memory */
    result = cuMemRetainAllocationHandle(&alloc_handle, (void*)address);
    if (result != CUDA_SUCCESS) {
        NIXL_DEBUG << "cuMemRetainAllocationHandle() failed. result = "
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

    id = (CUdevice)prop.location.id;
    switch (prop.location.type) {
    case CU_MEM_LOCATION_TYPE_DEVICE:
        type = MEM_VMM_DEV;
        ret = NIXL_SUCCESS;
        break;
#if HAVE_DECL_CU_MEM_LOCATION_TYPE_HOST
    case CU_MEM_LOCATION_TYPE_HOST:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
        // Currently not supported
        //type = MEM_VMM_HOST;
#endif
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

nixl_status_t
nixlCudaMemCtxImpl::retainVmmCudaCtx(uint64_t id, CUcontext &newCtx) const
{
    unsigned int flags;
    int active;

    CUresult result = cuDevicePrimaryCtxGetState(id, &flags, &active);
    if (result != CUDA_SUCCESS) {
        NIXL_ERROR << "cuDevicePrimaryCtxGetState() failed. result = "
                   << result;
        return NIXL_ERR_UNKNOWN;
    }

    if (!active) {
        NIXL_ERROR << "No active context found for CUDA device " << id;
        return NIXL_ERR_INVALID_PARAM;
    }

    result = cuDevicePrimaryCtxRetain(&newCtx, id);
    if (result != CUDA_SUCCESS) {
        NIXL_ERROR << "cuDevicePrimaryCtxRetain() failed. result = "
                   << result;
        return NIXL_ERR_UNKNOWN;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlCudaMemCtxImpl::releaseVmmCudaCtx(uint64_t id) const
{
    CUresult result = cuDevicePrimaryCtxRelease(id);
    if (result != CUDA_SUCCESS) {
        NIXL_ERROR << "cuDevicePrimaryCtxRelease() failed. result = "
                   << result;
    }
    return (CUDA_SUCCESS == result) ? NIXL_SUCCESS : NIXL_ERR_UNKNOWN;
}

nixl_status_t
nixlCudaMemCtxImpl::queryCuda(const void *address, memory_t &type, uint64_t &id, CUcontext &newCtx)
{
    CUmemorytype cudaMemType = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];
    CUresult result;
    CUdevice cuDevId;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &cudaMemType;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &cuDevId;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &newCtx;

    result = cuPointerGetAttributes(NUM_ATTRS, attr_type, attr_data, (CUdeviceptr)address);
    if (CUDA_SUCCESS != result) {
        NIXL_ERROR << "cuPointerGetAttributes() failed. result = "
                << result;
        return NIXL_ERR_NOT_FOUND;
    }

    id = (uint64_t)cuDevId;
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
        NIXL_ERROR << "Unknown CUDA memory type" << cudaMemType;
        return NIXL_ERR_NOT_FOUND;
    }

    if (is_managed) {
        NIXL_ERROR << "CUDA managed memory is not supported";
        return NIXL_ERR_INVALID_PARAM;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlCudaMemCtxImpl::enableAddr(const void *address, uint64_t chkDevId)
{
    nixl_status_t status;
    memory_t addrMemType = MEM_NONE;
    memory_t tmpMemType;
    CUcontext newCtx;
    uint64_t newDevId;

    status = queryVmm(address, tmpMemType, newDevId);
    if (NIXL_SUCCESS == status) {
        addrMemType = tmpMemType;
    } else if (status == NIXL_ERR_NOT_FOUND) {
        status = queryCuda(address, tmpMemType, newDevId, newCtx);
        if (NIXL_SUCCESS == status) {
            addrMemType = tmpMemType;
        } else if (status == NIXL_ERR_NOT_FOUND) {
            addrMemType = MEM_HOST;
            newDevId =
            status = NIXL_SUCCESS;
        } else {
            NIXL_ERROR << "CUDA Query failed with status = "
                       << status;
            // TODO use nixlEnumStrings::statusStr(status); once circ dep between libnixl & utils is resolved
        }
    } else {
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

    if (newDevId != chkDevId) {
        NIXL_DEBUG << "Mismatch between the expected and actual CUDA device id";
        NIXL_DEBUG << "Expect: " << chkDevId << ", have: " << newDevId;
        return NIXL_ERR_MISMATCH;
    }

    if (MEM_NONE == memType) {
        // Initialize the context
        switch(addrMemType) {
        case MEM_VMM_DEV:
            status = retainVmmCudaCtx(newDevId, newCtx);
            if (NIXL_SUCCESS != status) {
                return NIXL_ERR_UNKNOWN;
            }
            /* fall through */
        case MEM_DEV:
            ctx = newCtx;
            _devId = newDevId;
            status = NIXL_IN_PROG;
            // All set successfully =>  safe to set memType
            memType = addrMemType;
            break;
        default:
            NIXL_ERROR << "Unknown issue - memType is invalid: " <<  addrMemType;
            return NIXL_ERR_INVALID_PARAM;
        }
        return status;
    } else {
        // UCX up to 1.18 only supports one device context per
        // UCP context. Enforce that!
        if (_devId != newDevId) {
            status = NIXL_ERR_MISMATCH;
        }
        return status;
    }
}

nixl_status_t
nixlCudaMemCtxImpl::set()
{
    CUresult result;

    switch (memType) {
    case MEM_NONE:
    case MEM_HOST:
        // Nothing to do
        return NIXL_SUCCESS;
    case MEM_DEV:
    case MEM_VMM_DEV: {
        result = cuCtxSetCurrent(ctx);
        if (CUDA_SUCCESS != result) {
            NIXL_ERROR << "cuCtxSetCurrent() failed. result = "
                       << result;
            return NIXL_ERR_UNKNOWN;
        }
        return NIXL_SUCCESS;
    }
    case MEM_VMM_HOST:
        NIXL_ERROR << "Host VMM mappings are not supported";
        // fall through
    default:
        // TODO error log
        return NIXL_ERR_INVALID_PARAM;
    }
}

#endif
