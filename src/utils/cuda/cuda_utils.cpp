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

namespace nixlCuda {

/****************************************
 * CUDA nixlCudaPtr class
*****************************************/

#ifdef HAVE_CUDA

    class memCtxImpl : public memCtx {
        int devId;
        CUcontext ctx;
        memory_t memType;

        [[nodiscard]]
        nixl_status_t queryVmm(const void *address, memory_t &type, int &id);

        [[nodiscard]]
        nixl_status_t retainVmmCudaCtx(int id, CUcontext &newCtx) const;

        void releaseVmmCudaCtx(int id) const;

        [[nodiscard]]
        nixl_status_t queryCuda(const void *address, memory_t &type, int &id,
                                CUcontext &newCtx);

    public:

        memCtxImpl() : memCtx(), memType(MEM_NONE) { };
        ~memCtxImpl() override {
            if (MEM_VMM_DEV == memType) {
                releaseVmmCudaCtx(devId);
            }
        }

        [[nodiscard]]
        memory_t getMemType() const override {
            return memType;
        }

        [[nodiscard]]
        nixl_status_t enableAddr(const void *address, uint64_t chkDevId) override;

        [[nodiscard]]
        nixl_status_t set() override;
    };

#endif

    std::unique_ptr<memCtx>
    makeMemCtx()
    {
        // Environment fixup
        if (getenv("NIXL_DISABLE_CUDA_ADDR_WA")) {
            // If the workarounf is disabled - return the dummy class
            NIXL_INFO << "WARNING: disabling CUDA address workaround";
            return std::make_unique<memCtx>();
        } else {
#ifdef HAVE_CUDA
            return std::make_unique<memCtxImpl>();
#else
         return std::make_unique<memCtx>();
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

    /****************************************
     * CUDA memCtx class implementation
    *****************************************/

#ifdef HAVE_CUDA

    nixl_status_t
    memCtxImpl::queryVmm(const void *address, memory_t &type, int &id)
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

    nixl_status_t
    memCtxImpl::retainVmmCudaCtx(int id, CUcontext &newCtx) const
    {
        CUdevice device;

        auto result = cuDeviceGet(&device, id);
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
            NIXL_ERROR << "No active context found for CUDA device " << id;
            return NIXL_ERR_INVALID_PARAM;
        }

        result = cuDevicePrimaryCtxRetain(&newCtx, device);
        if (result != CUDA_SUCCESS) {
            NIXL_ERROR << "cuDevicePrimaryCtxRetain() failed. result = "
                    << result;
            return NIXL_ERR_UNKNOWN;
        }

        return NIXL_SUCCESS;
    }

    void
    memCtxImpl::releaseVmmCudaCtx(int id) const
    {
        CUdevice device;

        auto result = cuDeviceGet(&device, id);
        if (result != CUDA_SUCCESS) {
            NIXL_ERROR << "cuDeviceGet() failed. result = " << result;
            return;
        }

        result = cuDevicePrimaryCtxRelease(device);
        if (result != CUDA_SUCCESS) {
            NIXL_ERROR << "cuDevicePrimaryCtxRelease() failed. result = "
                    << result;
        }
    }

    nixl_status_t
    memCtxImpl::queryCuda(const void *address, memory_t &type, int &id, CUcontext &newCtx)
    {
        constexpr int numAttrs = 4;
        CUpointer_attribute attr_type[numAttrs];
        void *attr_data[numAttrs];
        CUmemorytype cudaMemType = CU_MEMORYTYPE_HOST;
        uint32_t is_managed = 0;
        CUresult result;
        int devOrdinal;

        attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
        attr_data[0] = &cudaMemType;
        attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
        attr_data[1] = &is_managed;
        attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
        attr_data[2] = &devOrdinal;
        attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
        attr_data[3] = &newCtx;

        result = cuPointerGetAttributes(numAttrs, attr_type, attr_data, (CUdeviceptr)address);
        if (CUDA_SUCCESS != result) {
            NIXL_ERROR << "cuPointerGetAttributes() failed. result = "
                    << result;
            return NIXL_ERR_NOT_FOUND;
        }

        id = devOrdinal;
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

    nixl_status_t
    memCtxImpl::enableAddr(const void *address, uint64_t chkDevId)
    {
        memory_t addrMemType = MEM_NONE;
        CUcontext newCtx;
        int newDevId;

        nixl_status_t status = queryVmm(address, addrMemType, newDevId);
        if (status == NIXL_ERR_NOT_FOUND) {
            status = queryCuda(address, addrMemType, newDevId, newCtx);
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

        if (static_cast<uint64_t>(newDevId) != chkDevId) {
            NIXL_DEBUG << "Mismatch between the expected and actual CUDA device id";
            NIXL_DEBUG << "Expect: " << chkDevId << ", have: " << newDevId;
            return NIXL_ERR_MISMATCH;
        }

        // The context was already set
        if (MEM_NONE != memType) {
            // UCX up to 1.18 only supports one device context per
            // UCP context. Enforce that!
            if (devId != newDevId) {
                status = NIXL_ERR_MISMATCH;
            }
            return status;
        }

        // Initialize the context
        switch(addrMemType) {
        case MEM_VMM_DEV:
            status = retainVmmCudaCtx(newDevId, newCtx);
            if (NIXL_SUCCESS != status) {
                return NIXL_ERR_UNKNOWN;
            }
            [[fallthrough]];
        case MEM_DEV:
            ctx = newCtx;
            devId = newDevId;
            status = NIXL_IN_PROG;
            // All set successfully =>  safe to set memType
            memType = addrMemType;
            break;
        default:
            NIXL_ERROR << "Unknown issue - memType is invalid: " <<  addrMemType;
            return NIXL_ERR_INVALID_PARAM;
        }
        return status;

    }

    nixl_status_t
    memCtxImpl::set()
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
        default:
            // TODO error log
            return NIXL_ERR_INVALID_PARAM;
        }
    }

#endif

}
