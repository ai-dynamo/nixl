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

nixl_status_t
deviceOps::allocDeviceMem(size_t size, deviceMem &out) noexcept {
    if (size == 0) {
        NIXL_ERROR << "Device allocation requires nonzero size";
        return NIXL_ERR_INVALID_PARAM;
    }
    void *ptr;
    const nixl_status_t status = doAllocDeviceMem(ptr, size);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    out = deviceMem(ptr, deviceMemDeleter{this});
    return NIXL_SUCCESS;
}

nixl_status_t
deviceOps::allocMappedHostMem(size_t size, mappedHostMem &out) noexcept {
    if (size == 0) {
        NIXL_ERROR << "Mapped host allocation requires nonzero size";
        return NIXL_ERR_INVALID_PARAM;
    }
    void *host_ptr;
    void *dev_ptr;
    const nixl_status_t status = doAllocMappedHostMem(host_ptr, dev_ptr, size);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    out = mappedHostMem(this, host_ptr, dev_ptr);
    return NIXL_SUCCESS;
}

deviceOps *
getDeviceOps() noexcept {
    static deviceOps *ops = loadCudaDeviceOps();
    return ops;
}

} // namespace nixl
