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
#include <iostream>
#include <cassert>
#include <map>

#include <cuda_runtime.h>
#include <cuda.h>

#include <cuda/cuda_utils.h>
#include <nixl_log.h>

using namespace std;

#define CHECK_CUDA_ERROR(result, message)                                           \
    do {                                                                            \
        if (result != cudaSuccess) {                                                \
            std::cerr << "CUDA: " << message << " (Error code: " << result          \
                      << " - " << cudaGetErrorString(result) << ")" << std::endl;   \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while(0)

#define CHECK_CUDA_DRIVER_ERROR(result, message)                                    \
    do {                                                                            \
        if (result != CUDA_SUCCESS) {                                               \
            const char *error_str;                                                  \
            cuGetErrorString(result, &error_str);                                   \
            std::cerr << "CUDA Driver: " << message << " (Error code: "             \
                      << result << " - " << error_str << ")" << std::endl;          \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while(0)

void allocateCUDA(int dev_id, size_t len, void* &addr, CUcontext *context)
{
    CUcontext contextTmp;
    CHECK_CUDA_ERROR(cudaSetDevice(dev_id), "Failed to cudaSetDevice()");
    CHECK_CUDA_ERROR(cudaMalloc(&addr, len), "Failed to allocate CUDA buffer");
    CHECK_CUDA_DRIVER_ERROR(cuCtxGetCurrent(context), "Failed to query current context");

    CHECK_CUDA_DRIVER_ERROR(cuCtxPopCurrent(nullptr), "Failed to pop the context");
    CHECK_CUDA_DRIVER_ERROR(cuCtxGetCurrent(&contextTmp), "Failed to query current context");
    assert(contextTmp == nullptr);
}

void releaseCUDA(int dev_id, void* addr)
{
    CHECK_CUDA_ERROR(cudaSetDevice(dev_id), "Failed to set device");
    CHECK_CUDA_ERROR(cudaFree(addr), "Failed to allocate CUDA buffer 0");
}

#ifdef HAVE_CUDA_VMM

#define ROUND_UP(value, granularity) ((((value) + (granularity) - 1) / (granularity)) * (granularity))

namespace {
    size_t __attribute__((unused)) padded_size = 0;
    std::map<void*, CUmemGenericAllocationHandle> handles;
}

void allocateVMM(int dev_id, size_t len, void* &_addr, CUcontext *context)
{
    CUdeviceptr addr = 0;
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle handle;

    CHECK_CUDA_ERROR(cudaSetDevice(dev_id), "Failed to set device");
    CHECK_CUDA_DRIVER_ERROR(cuCtxGetCurrent(context), "Failed to query current context");

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.location.id = dev_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    // Get the allocation granularity
    if (!padded_size) {
        CHECK_CUDA_DRIVER_ERROR(cuMemGetAllocationGranularity(&granularity, &prop,
                                                             CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                                "Failed to get allocation granularity");
        cout << "Granularity: " << granularity << std::endl;
        padded_size = ROUND_UP(len, granularity);
    }

    CHECK_CUDA_DRIVER_ERROR(cuMemCreate(&handle, padded_size, &prop, 0),
                            "Failed to create allocation");

    // Reserve the memory address
    CHECK_CUDA_DRIVER_ERROR(cuMemAddressReserve(&addr, padded_size,
                                                granularity, 0, 0),
                            "Failed to reserve address");
    handles[(void*)addr] = handle;
    // Map the memory
    CHECK_CUDA_DRIVER_ERROR(cuMemMap(addr, padded_size, 0, handle, 0),
                            "Failed to map memory");

    cout << "Address: " << std::hex << std::showbase << addr
              << " Buffer size: " << std::dec << len
              << " Padded size: " << std::dec << padded_size << std::endl;

    // Set the memory access rights
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA_DRIVER_ERROR(cuMemSetAccess(addr, padded_size, &access, 1),
                            "Failed to set access");

    _addr = (void*)addr;

    // Release the context
    CUcontext contextTmp;
    CHECK_CUDA_DRIVER_ERROR(cuCtxPopCurrent(nullptr), "Failed to pop the context");
    CHECK_CUDA_DRIVER_ERROR(cuCtxGetCurrent(&contextTmp), "Failed to query current context");
    assert(contextTmp == nullptr);
}

void releaseVMM(int dev_id, size_t len, void* addr)
{
    CHECK_CUDA_ERROR(cudaSetDevice(dev_id), "Failed to set device");
    CHECK_CUDA_DRIVER_ERROR(cuMemUnmap((CUdeviceptr)addr, padded_size),
                            "Failed to unmap memory");

    assert(handles.find(addr) != handles.end());
    CHECK_CUDA_DRIVER_ERROR(cuMemRelease(handles[addr]),
                            "Failed to release memory");
    CHECK_CUDA_DRIVER_ERROR(cuMemAddressFree((CUdeviceptr)addr, padded_size),
                            "Failed to free reserved address");
}

#endif


int main()
{
    size_t len = 1024;

    /* Discover environment */
    int ngpus;
    cudaGetDeviceCount(&ngpus);

    if (!ngpus) {
        cout << "No GPGPU devices detected, nothing to test!" << endl;
        return 0;
    }

    /* Test regular CUDA malloc */
    {
        cout << endl << "*************************" << endl;
        cout << "      Test malloc'd memory" << endl;

        void *address = malloc(len);
        assert(address);
        std::shared_ptr<nixl::cuda::memCtx> ctx = nixl::cuda::makeMemCtx();
        assert(NIXL_SUCCESS == ctx->initFromAddr(address, 0));
        assert(NIXL_SUCCESS == ctx->set());
        cout << "      >>>> PASSED! <<<<<<<" << endl;
        free(address);
        cout << "*************************" << endl;
    }

    /* Test regular CUDA malloc */
    {
        cout << endl << "*************************" << endl;
        cout << "      Test CUDA malloc'd memory" << endl;

        void *address;
        CUcontext context, context1;
        allocateCUDA(0, len, address, &context);

        assert(context1 == nullptr);
        std::shared_ptr<nixl::cuda::memCtx> ctx = nixl::cuda::makeMemCtx();
        assert(NIXL_IN_PROG == ctx->initFromAddr(address, 0));
        assert(NIXL_SUCCESS == ctx->set());
        CHECK_CUDA_DRIVER_ERROR(cuCtxGetCurrent(&context1), "Failed to query current context");
        assert(context == context1);
        cout << "      >>>> PASSED! <<<<<<<" << endl;
        releaseCUDA(0, address);
        cout << "*************************" << endl;
    }


    /* Test regular CUDA malloc address mismatch */
    if (ngpus > 1) {
        cout << endl << "*************************" << endl;
        cout << "      Test CUDA malloc'd memory: device mismatch" << endl;

        std::shared_ptr<nixl::cuda::memCtx> ctx = nixl::cuda::makeMemCtx();

        void *address;
        CUcontext context;
        allocateCUDA(0, len, address, &context);
        assert(NIXL_IN_PROG == ctx->initFromAddr(address, 0));
        assert(NIXL_SUCCESS == ctx->set());

        void *address2;
        allocateCUDA(1, len, address2, &context);
        assert(NIXL_ERR_MISMATCH == ctx->initFromAddr(address2, 0));
        assert(NIXL_ERR_MISMATCH == ctx->initFromAddr(address2, 1));

        cout << "      >>>> PASSED! <<<<<<<" << endl;

        releaseCUDA(0, address);
        releaseCUDA(1, address2);
        cout << "*************************" << endl;
    }

#ifdef HAVE_CUDA_VMM
    /* Test VMM allocation */
    {
        cout << endl << "*************************" << endl;
        cout << "      Test VMM mapped memory" << endl;
        std::shared_ptr<nixl::cuda::memCtx> ctx = nixl::cuda::makeMemCtx();

        void *address;
        CUcontext context, context1;
        allocateVMM(0, len, address, &context);

        assert(NIXL_IN_PROG == ctx->initFromAddr(address, 0));
        assert(NIXL_SUCCESS == ctx->set());
        CHECK_CUDA_DRIVER_ERROR(cuCtxGetCurrent(&context1), "Failed to query current context");
        assert(context == context1);

        // CUDA malloc'd memory is OK as long as on the same dev
        void *address2;
        allocateCUDA(0, len, address2, &context);
        assert(NIXL_SUCCESS == ctx->initFromAddr(address2, 0));

        cout << "      >>>> PASSED! <<<<<<<" << endl;
        releaseVMM(0, len, address);
        releaseCUDA(0, address2);
        cout << "*************************" << endl;
    }

     /* Test regular CUDA malloc address mismatch */
     if (ngpus > 1) {
        cout << endl << "*************************" << endl;
        cout << "      Test VMM mapped memory: device MISMATCH" << endl;
        std::shared_ptr<nixl::cuda::memCtx> ctx = nixl::cuda::makeMemCtx();

        void *address;
        CUcontext context;
        allocateVMM(0, len, address, &context);
        assert(NIXL_IN_PROG == ctx->initFromAddr(address, 0));
        assert(NIXL_SUCCESS == ctx->set());

        // VMM memory on a different device is a mismatch
        void *address2;
        allocateVMM(1, len, address2, &context);
        assert(NIXL_ERR_MISMATCH == ctx->initFromAddr(address2, 0));
        assert(NIXL_ERR_MISMATCH == ctx->initFromAddr(address2, 0));

        // CUDA malloc memory on a different device is a mismatch
        void *address3;
        allocateCUDA(1, len, address3, &context);
        assert(NIXL_ERR_MISMATCH == ctx->initFromAddr(address3, 0));
        assert(NIXL_ERR_MISMATCH == ctx->initFromAddr(address3, 1));

        cout << "      >>>> PASSED! <<<<<<<" << endl;
        releaseVMM(0, len, address);
        releaseVMM(1, len, address2);
        releaseCUDA(1, address3);
        cout << "*************************" << endl;

     }

#endif

}
