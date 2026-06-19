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
#include <pybind11/pybind11.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda.h>
#include "cuda/vmm.h"
#endif

namespace py = pybind11;

// NB: for storage IO with O_DIRECT, allocated memory must match device block size
#define IO_SIZE 4096

//JUST FOR TESTING
uintptr_t malloc_passthru(int size) {
    return (uintptr_t)aligned_alloc(IO_SIZE, size);
}

//JUST FOR TESTING
void free_passthru(uintptr_t buf) {
    free((void*) buf);
}

//JUST FOR TESTING
void ba_buf(uintptr_t addr, int size) {
    uint8_t* buf = (uint8_t*) addr;
    for(int i = 0; i<size; i++) buf[i] = 0xba;
}

//JUST FOR TESTING
void verify_transfer(uintptr_t addr1, uintptr_t addr2, int size) {
    for(int i = 0; i<size; i++) assert(((uint8_t*) addr1)[i] == ((uint8_t*) addr2)[i]);
}

#ifdef HAVE_CUDA
namespace {

std::mutex cuda_alloc_mu;
std::unordered_map<uintptr_t, std::unique_ptr<nixl::cuda::fabric_vmm_region>> cuda_allocs;

void check_cu(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        const char* msg = nullptr;
        if (cuGetErrorString(status, &msg) != CUDA_SUCCESS || msg == nullptr) {
            msg = "unknown CUDA driver error";
        }
        throw std::runtime_error(std::string(operation) + " failed: " + msg);
    }
}

nixl::cuda::fabric_vmm_region* find_cuda_alloc(uintptr_t addr) {
    auto it = cuda_allocs.find(addr);
    if (it == cuda_allocs.end()) {
        throw std::invalid_argument("unknown CUDA fabric allocation");
    }
    return it->second.get();
}

} // namespace

bool has_cuda_support() {
    return true;
}

int cuda_device_count() {
    if (cuInit(0) != CUDA_SUCCESS) {
        return 0;
    }
    int count = 0;
    return cuDeviceGetCount(&count) == CUDA_SUCCESS ? count : 0;
}

bool cuda_fabric_supported(int device_id = 0) {
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    return nixl::cuda::is_fabric_vmm_supported(device_id);
}

uintptr_t cuda_fabric_malloc(std::size_t size, int device_id = 0, bool require_fabric = true) {
    check_cu(cuInit(0), "cuInit");
    auto region = std::make_unique<nixl::cuda::fabric_vmm_region>(size, device_id, require_fabric);
    auto addr = reinterpret_cast<uintptr_t>(region->ptr());
    std::lock_guard<std::mutex> lock(cuda_alloc_mu);
    cuda_allocs.emplace(addr, std::move(region));
    return addr;
}

void cuda_fabric_free(uintptr_t addr) {
    std::lock_guard<std::mutex> lock(cuda_alloc_mu);
    auto it = cuda_allocs.find(addr);
    if (it == cuda_allocs.end()) {
        throw std::invalid_argument("unknown CUDA fabric allocation");
    }
    cuda_allocs.erase(it);
}

void cuda_memset(uintptr_t addr, int value, std::size_t size) {
    {
        std::lock_guard<std::mutex> lock(cuda_alloc_mu);
        find_cuda_alloc(addr)->set_current_context();
    }
    check_cu(cuMemsetD8(static_cast<CUdeviceptr>(addr), static_cast<unsigned char>(value), size), "cuMemsetD8");
}

void cuda_verify_transfer(uintptr_t addr1, uintptr_t addr2, std::size_t size) {
    std::vector<uint8_t> src(size);
    std::vector<uint8_t> dst(size);
    {
        std::lock_guard<std::mutex> lock(cuda_alloc_mu);
        find_cuda_alloc(addr1)->set_current_context();
        check_cu(cuMemcpyDtoH(src.data(), static_cast<CUdeviceptr>(addr1), size), "cuMemcpyDtoH(src)");
        find_cuda_alloc(addr2)->set_current_context();
        check_cu(cuMemcpyDtoH(dst.data(), static_cast<CUdeviceptr>(addr2), size), "cuMemcpyDtoH(dst)");
    }
    if (src != dst) {
        throw std::runtime_error("CUDA buffers differ");
    }
}
#else
bool has_cuda_support() {
    return false;
}

int cuda_device_count() {
    return 0;
}

bool cuda_fabric_supported(int device_id = 0) {
    (void)device_id;
    return false;
}

uintptr_t cuda_fabric_malloc(std::size_t size, int device_id = 0, bool require_fabric = true) {
    (void)size;
    (void)device_id;
    (void)require_fabric;
    throw std::runtime_error("NIXL was built without CUDA support");
}

void cuda_fabric_free(uintptr_t addr) {
    (void)addr;
    throw std::runtime_error("NIXL was built without CUDA support");
}

void cuda_memset(uintptr_t addr, int value, std::size_t size) {
    (void)addr;
    (void)value;
    (void)size;
    throw std::runtime_error("NIXL was built without CUDA support");
}

void cuda_verify_transfer(uintptr_t addr1, uintptr_t addr2, std::size_t size) {
    (void)addr1;
    (void)addr2;
    (void)size;
    throw std::runtime_error("NIXL was built without CUDA support");
}
#endif

bool is_cuda_fabric_vmm_supported(int device_id = 0) {
    return cuda_fabric_supported(device_id);
}

uintptr_t cuda_fabric_vmm_alloc(std::size_t size, int device_id = 0, bool require_fabric = true) {
    return cuda_fabric_malloc(size, device_id, require_fabric);
}

void cuda_fabric_vmm_free(uintptr_t addr) {
    cuda_fabric_free(addr);
}

PYBIND11_MODULE(_utils, m) {
    m.def("malloc_passthru", &malloc_passthru);
    m.def("free_passthru", &free_passthru);
    m.def("ba_buf", &ba_buf);
    m.def("verify_transfer", &verify_transfer);
    m.def("has_cuda_support", &has_cuda_support);
    m.def("cuda_device_count", &cuda_device_count);
    m.def("cuda_fabric_supported", &cuda_fabric_supported, py::arg("device_id") = 0);
    m.def("cuda_fabric_malloc", &cuda_fabric_malloc, py::arg("size"), py::arg("device_id") = 0, py::arg("require_fabric") = true);
    m.def("cuda_fabric_free", &cuda_fabric_free);
    m.def("is_cuda_fabric_vmm_supported", &is_cuda_fabric_vmm_supported, py::arg("device_id") = 0);
    m.def("cuda_fabric_vmm_alloc", &cuda_fabric_vmm_alloc, py::arg("size"), py::arg("device_id") = 0, py::arg("require_fabric") = true);
    m.def("cuda_fabric_vmm_free", &cuda_fabric_vmm_free);
    m.def("cuda_memset", &cuda_memset);
    m.def("cuda_verify_transfer", &cuda_verify_transfer);
}
