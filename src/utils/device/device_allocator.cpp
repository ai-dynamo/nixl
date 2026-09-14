/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "device/device_allocator.h"

#include <dlfcn.h>

#include <filesystem>

#include "common/nixl_log.h"

namespace {

constexpr const char *kCudaAllocatorLibrary = "libnixl_device_allocator_cuda.so";
constexpr const char *kCudaAllocatorFactory = "nixlCreateCudaDeviceAllocator";

class nixlUnsupportedDeviceAllocator final : public nixlDeviceAllocator {
public:
    nixl_status_t
    doAllocDeviceMem(void **, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    void
    doFreeDeviceMem(void *) noexcept override {}

    nixl_status_t
    doAllocMappedHostMem(void **, void **, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    void
    doFreeMappedHostMem(void *) noexcept override {}

    nixl_status_t
    copyHostToDevice(void *, const void *, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    copyDeviceToHost(void *, const void *, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    memsetDeviceMem(void *, int, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    synchronize() noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    getActiveDevice(int &) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    setActiveDevice(int) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }
};

using CudaAllocatorFactory = nixlDeviceAllocator *(*)() noexcept;

nixlDeviceAllocator *
loadCudaAllocator() noexcept {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&nixlGetDeviceAllocator), &info) == 0 ||
        info.dli_fname == nullptr) {
        return nullptr;
    }

    // The frontend and optional CUDA implementation are installed side by side.
    const auto library_path =
        std::filesystem::path(info.dli_fname).parent_path() / kCudaAllocatorLibrary;
    void *handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (handle == nullptr) {
        NIXL_INFO << "Failed to load CUDA device allocator from " << library_path << ": "
                  << dlerror();
        return nullptr;
    }

    dlerror(); // Clear any error left by an earlier dynamic-loader call.
    auto factory = reinterpret_cast<CudaAllocatorFactory>(dlsym(handle, kCudaAllocatorFactory));
    if (factory == nullptr) {
        NIXL_ERROR << "Failed to find " << kCudaAllocatorFactory << " in " << library_path << ": "
                   << dlerror();
        dlclose(handle);
        return nullptr;
    }

    nixlDeviceAllocator *allocator = factory();
    if (allocator == nullptr) {
        dlclose(handle);
    }
    // Keep the library loaded on success because the allocator and its vtable live in it.
    return allocator;
}

} // namespace

nixlDeviceAllocator &
nixlGetDeviceAllocator() noexcept {
    static nixlUnsupportedDeviceAllocator unsupported;
    static nixlDeviceAllocator *allocator = []() noexcept {
        nixlDeviceAllocator *cuda_allocator = loadCudaAllocator();
        return cuda_allocator == nullptr ? &unsupported : cuda_allocator;
    }();
    return *allocator;
}
