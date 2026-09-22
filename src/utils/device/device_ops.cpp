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
#include "device/device_ops.h"

#include <dlfcn.h>
#include <filesystem>

#include "common/nixl_log.h"

namespace {

constexpr const char *kCudaDeviceOpsLibrary = "libnixl_device_ops_cuda.so";
constexpr const char *kCudaDeviceOpsFactory = "nixlCreateCudaDeviceOps";

class nullDeviceOps final : public nixl::deviceOps {
public:
    nixl_status_t
    doAllocDeviceMem(void *&, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    void
    doFreeDeviceMem(void *) noexcept override {}

    nixl_status_t
    doAllocMappedHostMem(void *&, void *&, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    void
    doFreeMappedHostMem(void *) noexcept override {}

    nixl_status_t
    copyHostToDevice(void *, const void *, size_t size) noexcept override {
        return size == 0 ? NIXL_ERR_INVALID_PARAM : NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    copyDeviceToHost(void *, const void *, size_t size) noexcept override {
        return size == 0 ? NIXL_ERR_INVALID_PARAM : NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    memsetDeviceMem(void *, int, size_t size) noexcept override {
        return size == 0 ? NIXL_ERR_INVALID_PARAM : NIXL_ERR_NOT_SUPPORTED;
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

using CudaDeviceOpsFactory = nixl::deviceOps *(*)() noexcept;

nixl::deviceOps *
loadCudaDeviceOps() noexcept {
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&nixl::getDeviceOps), &info) == 0 ||
        info.dli_fname == nullptr) {
        NIXL_ERROR << "Failed to locate the device operations frontend library";
        return nullptr;
    }

    // The frontend and optional CUDA implementation are installed side by side.
    const auto library_path =
        std::filesystem::path(info.dli_fname).parent_path() / kCudaDeviceOpsLibrary;
    void *handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (handle == nullptr) {
        const char *error = dlerror();
        std::error_code ec;
        if (std::filesystem::exists(library_path, ec) || ec) {
            NIXL_WARN << "Failed to load CUDA device operations from " << library_path << ": "
                      << error;
        } else {
            NIXL_INFO << "CUDA device operations are unavailable at " << library_path << ": "
                      << error;
        }
        return nullptr;
    }

    dlerror(); // Clear any error left by an earlier dynamic-loader call.
    auto factory = reinterpret_cast<CudaDeviceOpsFactory>(dlsym(handle, kCudaDeviceOpsFactory));
    if (factory == nullptr) {
        NIXL_WARN << "Failed to find " << kCudaDeviceOpsFactory << " in " << library_path << ": "
                  << dlerror();
        dlclose(handle);
        return nullptr;
    }

    nixl::deviceOps *ops = factory();
    if (ops == nullptr) {
        dlclose(handle);
    } else {
        NIXL_INFO << "Loaded CUDA device operations from " << library_path;
    }
    // Keep the library loaded on success because the implementation and its vtable live in it.
    return ops;
}

} // namespace

namespace nixl {

deviceOps &
getDeviceOps() noexcept {
    static nullDeviceOps null_ops;
    static deviceOps *ops = []() noexcept {
        deviceOps *cuda_ops = loadCudaDeviceOps();
        return cuda_ops == nullptr ? &null_ops : cuda_ops;
    }();
    return *ops;
}

} // namespace nixl
