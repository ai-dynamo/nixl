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

void cudaQueryAddr(void *address, bool &is_dev,
                         CUdevice &dev, CUcontext &ctx)
{
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    CHECK_CUDA_DRIVER_ERROR(cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address),
                            "Failed to cuPointerGetAttributes");

    is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);
}

void allocateCUDA(int dev_id, size_t len, void* &addr)
{
    bool is_dev;
    CUdevice dev;
    CUcontext ctx;

    CHECK_CUDA_ERROR(cudaMalloc(&addr, len), "Failed to allocate CUDA buffer 0");
    cudaQueryAddr(addr, is_dev, dev, ctx);
    cout << "CUDA addr: " << std::hex << addr << " dev=" << std::dec << dev
              << " ctx=" << std::hex << ctx << std::dec << std::endl;
}

void releaseCUDA(int dev_id, void* addr)
{
    CHECK_CUDA_ERROR(cudaSetDevice(dev_id), "Failed to set device");
    CHECK_CUDA_ERROR(cudaFree(addr), "Failed to allocate CUDA buffer 0");
}

#ifdef HAVE_CUDA_VMM

#define ROUND_UP(value, granularity) ((((value) + (granularity) - 1) / (granularity)) * (granularity))

namespace {
    static size_t __attribute__((unused)) padded_size = 0;
    static CUmemGenericAllocationHandle __attribute__((unused)) handle;
}

void allocateVMM(int dev_id, size_t len, void* &_addr)
{
    CUdeviceptr addr = 0;
    size_t granularity = 0;
    CUmemAllocationProp prop = {};

    CHECK_CUDA_ERROR(cudaSetDevice(dev_id), "Failed to set device");

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.location.id = dev_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;

    // Get the allocation granularity
    CHECK_CUDA_DRIVER_ERROR(cuMemGetAllocationGranularity(&granularity, &prop,
                                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                            "Failed to get allocation granularity");
    cout << "Granularity: " << granularity << std::endl;

    padded_size = ROUND_UP(len, granularity);
    CHECK_CUDA_DRIVER_ERROR(cuMemCreate(&handle, padded_size, &prop, 0),
                            "Failed to create allocation");

    // Reserve the memory address
    CHECK_CUDA_DRIVER_ERROR(cuMemAddressReserve(&addr, padded_size,
                                                granularity, 0, 0),
                            "Failed to reserve address");

    // Map the memory
    CHECK_CUDA_DRIVER_ERROR(cuMemMap(addr, padded_size, 0, handle, 0),
                            "Failed to map memory");

    cout << "Address: " << std::hex << std::showbase << addr
              << " Buffer size: " << std::dec << len
              << " Padded size: " << std::dec << padded_size << std::endl;

    // // Set the memory access rights
    // CUmemAccessDesc access = {};
    // access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // access.location.id = dev_id;
    // access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // CHECK_CUDA_DRIVER_ERROR(cuMemSetAccess(addr, len, &access, 1),
    //                         "Failed to set access");

    _addr = (void*)addr;
}

void releaseVMM(int dev_id, size_t len, void* addr)
{
    CHECK_CUDA_ERROR(cudaSetDevice(dev_id), "Failed to set device");
    CHECK_CUDA_DRIVER_ERROR(cuMemUnmap((CUdeviceptr)addr, padded_size),
                            "Failed to unmap memory");
    CHECK_CUDA_DRIVER_ERROR(cuMemRelease(handle),
                            "Failed to release memory");
    CHECK_CUDA_DRIVER_ERROR(cuMemAddressFree((CUdeviceptr)addr, padded_size),
                            "Failed to free reserved address");
}

#endif


int main()
{
    void *address;
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

        address = malloc(len);
        assert(address);
        std::unique_ptr<nixlCudaPtrCtx> ctx =
                nixlCudaPtrCtx::nixlCudaPtrCtxInit(address);
        assert(ctx->getMemType() == nixlCudaPtrCtx::MEM_HOST);
        cout << " >>>> PASSED! <<<<<<<" << endl;
        free(address);
        cout << "*************************" << endl;
    }

    /* Test regular CUDA malloc */
    {
        cout << endl << "*************************" << endl;
        cout << "      Test CUDA malloc'd memory" << endl;

        allocateCUDA(0, len, address);
        std::unique_ptr<nixlCudaPtrCtx> ctx =
                nixlCudaPtrCtx::nixlCudaPtrCtxInit(address);
        assert(ctx->getMemType() == nixlCudaPtrCtx::MEM_DEV);
        cout << " >>>> PASSED! <<<<<<<" << endl;
        releaseCUDA(0, address);
        cout << "*************************" << endl;
    }

#ifdef HAVE_CUDA_VMM
    /* Test regular CUDA malloc */
    {

        cout << endl << "*************************" << endl;
        cout << "      Test VMM mapped memory" << endl;

        allocateVMM(0, len, address);
        std::unique_ptr<nixlCudaPtrCtx> ctx =
                nixlCudaPtrCtx::nixlCudaPtrCtxInit(address);
        assert(ctx->getMemType() == nixlCudaPtrCtx::MEM_VMM_DEV);
        cout << " >>>> PASSED! <<<<<<<" << endl;
        releaseVMM(0, len, address);
        cout << "*************************" << endl;
    }
#endif

}
