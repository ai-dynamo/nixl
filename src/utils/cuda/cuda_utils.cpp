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

#include <cuda_utils.h>

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

class nixlCudaPtrImpl : public nixlCudaPtrCtx {
private:
    CUcontext ctx;
    bool wasSet;

    nixl_status_t initVmm(void *address);
    nixl_status_t initCuda(void *address);


    /* To be used in derived classes */
    inline virtual bool
    internalCmp(const nixlCudaPtrCtx &_rhs) override {
        const nixlCudaPtrImpl &rhs = *(const nixlCudaPtrImpl *)&_rhs;
        switch (mem_type) {
        case MEM_HOST:
            return true;
        case MEM_DEV:
        case MEM_VMM_DEV:
            return (devId == rhs.devId) &&
                   (ctx == rhs.ctx);
            break;
        case MEM_VMM_HOST:
            // TODO: check what is required
        default:
            // TODO error log
            return false;
        }
    }

    nixl_status_t checkVmm(void *address);
    nixl_status_t checkCuda(void *address);
    nixl_status_t unsetMemCtx() override;

public:

    nixlCudaPtrImpl(void *addr) : nixlCudaPtrCtx (addr)
    {
        nixl_status_t status;

        // Test VMM allocations first
        status = checkVmm(address);
        if (status == NIXL_SUCCESS) {
            // This is VMM allocation, the class is initialized
            return;
        }

        if (status != NIXL_ERR_NOT_FOUND) {
            // Unexpected error
            // TODO: throw status;
            mem_type = MEM_INVALID;
        }

        // Continue with CUDA and Host allocations
        status = checkCuda(address);
        if (status == NIXL_SUCCESS) {
            // Everything is initialized
            return;
        }
        if (status != NIXL_ERR_NOT_FOUND) {
            // Unexpected error
            // TODO: throw status;
            mem_type = MEM_INVALID;
        }

        mem_type = MEM_HOST;
    }

    ~nixlCudaPtrImpl() override {
        if (wasSet) {
            unsetMemCtx();
        }
    }



    nixl_status_t setMemCtx() override;
};

#endif



/****************************************
 * Static nixlCudaPtr functions
*****************************************/

#ifdef HAVE_CUDA

#define NIXL_CUDA_PTR_CTX_CLASS nixlCudaPtrImpl
#define NIXL_CUDA_PTR_CTX_VRAM_SUPPORT true

#else

#define NIXL_CUDA_PTR_CTX_CLASS nixlCudaPtrCtx
#define NIXL_CUDA_PTR_CTX_VRAM_SUPPORT false

#endif

bool nixlCudaPtrCtx::vramIsSupported()
{

    bool ret = NIXL_CUDA_PTR_CTX_VRAM_SUPPORT;
    return ret;
}

std::unique_ptr<nixlCudaPtrCtx>
nixlCudaPtrCtx::nixlCudaPtrCtxInit(void *address)
{
    std::unique_ptr<NIXL_CUDA_PTR_CTX_CLASS> ptr;
    ptr = std::make_unique<NIXL_CUDA_PTR_CTX_CLASS>(address);

    return ptr;
}


/****************************************
 * CUDA nixlCudaPtr class implementation
*****************************************/

#ifdef HAVE_CUDA

nixl_status_t
nixlCudaPtrImpl::checkVmm(void *address)
{
    nixl_status_t ret = NIXL_SUCCESS;

#if HAVE_CUMEMRETAINALLOCATIONHANDLE
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle alloc_handle;
    CUresult result;

    /* Check if memory is allocated using VMM API and see if host memory needs
     * to be treated as pinned device memory */
    result = cuMemRetainAllocationHandle(&alloc_handle, (void*)address);
    if (result != CUDA_SUCCESS) {
        return NIXL_ERR_NOT_FOUND;
    }
    // TODO: set the call to cuMemRelease when leaving the scope to avoid GOTO

    result = cuMemGetAllocationPropertiesFromHandle(&prop, alloc_handle);
    if (result != CUDA_SUCCESS) {
        // TODO: log error
        ret = NIXL_ERR_UNKNOWN;
        goto err;
    }

    devId = (CUdevice)prop.location.id;
    switch (prop.location.type) {
#if HAVE_DECL_CU_MEM_LOCATION_TYPE_HOST
    case CU_MEM_LOCATION_TYPE_HOST:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
        /* Do we need to set context in this case? */
        mem_type = MEM_VMM_HOST;
        break;
#endif
    case CU_MEM_LOCATION_TYPE_DEVICE:
        mem_type = MEM_VMM_DEV;
        break;
    default:
        // This is VMM memory, but its invalid
        ret = NIXL_ERR_INVALID_PARAM;
        goto err;
    }

err:
    result = cuMemRelease(alloc_handle);
    if (CUDA_SUCCESS != result) {
        // TODO: log error
        if (NIXL_SUCCESS == ret) {
            ret = NIXL_ERR_UNKNOWN;
        }
    }
#endif
    return ret;
}


nixl_status_t
nixlCudaPtrImpl::checkCuda(void *address)
{
    CUmemorytype cuda_mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &cuda_mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &devId;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);
    if (CUDA_SUCCESS != result) {
        return NIXL_ERR_NOT_FOUND;
    }

    switch(cuda_mem_type) {
    case CU_MEMORYTYPE_DEVICE:
        mem_type = MEM_DEV;
        break;
    case CU_MEMORYTYPE_HOST:
        mem_type = MEM_HOST;
    case CU_MEMORYTYPE_ARRAY:
        // TODO: how should this case be processed?
        return NIXL_ERR_INVALID_PARAM;
    default:
        return NIXL_ERR_NOT_FOUND;
    }

    // TODO: what to do if the memory "is_managed"?

    return NIXL_SUCCESS;
}

nixl_status_t
nixlCudaPtrImpl::setMemCtx()
{
    CUresult result;

    switch (mem_type) {
    case MEM_HOST:
        return NIXL_SUCCESS;
    case MEM_DEV: {
        result = cuCtxSetCurrent(ctx);
        if (CUDA_SUCCESS != result) {
            // TODO: something like NIXL_ERR_CMD_FAILED
            // would be more appropriate
            return NIXL_ERR_NOT_SUPPORTED;
        }
        wasSet = true;
        return NIXL_SUCCESS;
    }
    case MEM_VMM_DEV: {
#if 0
        unsigned int flags;
        int active;

        result = cuDevicePrimaryCtxGetState(devId, &flags, &active);
        if (result != CUDA_SUCCESS) {
            // TODO: log error
            return NIXL_ERR_UNKNOWN;
        }

        if (!active) {
            // TODO: Not supported at the moment. In most cases it is set
            // FIXME: Allocate a new context?
            return NIXL_ERR_INVALID_PARAM;
        }

        result = cuDevicePrimaryCtxRetain(&ctx, devId);
        if (result != CUDA_SUCCESS) {
            // TODO: log error
            return NIXL_ERR_UNKNOWN;
        }
        return NIXL_SUCCESS;
#endif
        // Fall thru
    }
    case MEM_VMM_HOST:
        // TODO: Not supported at the moment
    default:
        // TODO error log
        return NIXL_ERR_INVALID_PARAM;
    }
}

nixl_status_t
nixlCudaPtrImpl::unsetMemCtx()
{
    switch (mem_type) {
    case MEM_HOST:
    case MEM_DEV:
        return NIXL_SUCCESS;
    case MEM_VMM_DEV: {
#if 0
        CUresult result;
        result = cuDevicePrimaryCtxRelease(devId);
        if (result != CUDA_SUCCESS) {
            // TODO: log error
            return NIXL_ERR_UNKNOWN;
        }
        return NIXL_SUCCESS;
#endif
    }
    case MEM_VMM_HOST:
        // TODO: Not supported at the moment
    default:
        // TODO error log
        return NIXL_ERR_INVALID_PARAM;
    }
}

#endif
