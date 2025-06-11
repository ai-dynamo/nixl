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
#ifndef __MEMORY_HANDLER_H
#define __MEMORY_HANDLER_H

template<nixl_mem_t MemType> struct MemoryHandler {
    static void *
    allocate(size_t len, int devId = 0) {
        std::cerr << "Unsupported memory type!" << std::endl;
        assert(0);
    }

    static void
    deallocate(void *addr, int devId = 0) {
        std::cerr << "Unsupported memory type!" << std::endl;
        assert(0);
    }

    static void
    set(void *addr, char byte, size_t size, int devId = 0) {
        std::cerr << "Unsupported memory type!" << std::endl;
        assert(0);
    }

    static void *
    getPtr(void *addr, size_t len) {
        std::cerr << "Unsupported memory type!" << std::endl;
        assert(0);
    }

    static void
    releasePtr(void *addr) {
        std::cerr << "Unsupported memory type!" << std::endl;
        assert(0);
    }
};

template<> struct MemoryHandler<DRAM_SEG> {
    static void *
    allocate(size_t len, int devId = 0) {
        return new char[len];
    }

    static void
    deallocate(void *addr, int devId = 0) {
        delete[] static_cast<char *>(addr);
    }

    static void
    set(void *addr, char byte, size_t size, int devId = 0) {
        memset(addr, byte, size);
    }

    static void *
    getPtr(void *addr, size_t len) {
        return addr;
    }

    static void
    releasePtr(void *addr) {
        // No-op for host memory
    }
};

#ifdef HAVE_CUDA
template<> struct MemoryHandler<VRAM_SEG> {
    static void *
    allocate(size_t len, int devId = 0) {
        CUcontext ctx;
        CUdevice dev;
        bool is_dev;
        void *ptr;
        checkCudaError(cudaSetDevice(devId), "Failed to set device");
        checkCudaError(cudaMalloc(&ptr, len), "Failed to allocate CUDA buffer");
        if (cudaQueryAddr(ptr, is_dev, dev, ctx)) {
            std::cerr << "Failed to query CUDA addr: " << std::hex << ptr << " dev=" << std::dec
                      << dev << " ctx=" << std::hex << ctx << std::dec << std::endl;
            assert(0);
        }

        return ptr;
    }

    static void
    deallocate(void *addr, int devId = 0) {
        checkCudaError(cudaSetDevice(devId), "Failed to set device");
        checkCudaError(cudaFree(addr), "Failed to free CUDA buffer");
    }

    static void
    set(void *addr, char byte, size_t size, int devId = 0) {
        checkCudaError(cudaSetDevice(devId), "Failed to set device");
        checkCudaError(cudaMemset(addr, byte, size), "Failed to memset");
    }

    static void *
    getPtr(void *addr, size_t len) {
        void *ptr = new char[len];
        checkCudaError(cudaMemcpy(ptr, addr, len, cudaMemcpyDeviceToHost), "Failed to memcpy");
        return ptr;
    }

    static void
    releasePtr(void *addr) {
        delete[] static_cast<char *>(addr);
    }
};
#endif // HAVE_CUDA
#endif // __MEMORY_HANDLER_H
