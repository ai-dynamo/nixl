/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>
#include <dlfcn.h>

#include <nixl_ep_v2/common/exception.cuh>

namespace nixl_ep_v2 {

// Lazy loading all driver symbols
static void* get_driver_handle() {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        EP_HOST_ASSERT(handle != nullptr and "Failed to load CUDA driver `libcuda.so.1`");
    }
    return handle;
}

// Macro to define wrapper functions named `lazy_cu{API name}`
#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name) \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&name); \
    static FuncType func = nullptr; \
    if (func == nullptr) { \
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), #name)); \
        EP_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API"); \
    } \
    return func(std::forward<decltype(args)>(args)...); \
}

DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleLoad);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleUnload);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleGetFunction);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemSetAccess);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemRetainAllocationHandle);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemGetAddressRange_v2);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemAddressReserve);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemAddressFree);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemMap);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemUnmap);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemCreate);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemRelease);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuCtxGetDevice);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemImportFromShareableHandle);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemExportToShareableHandle);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemGetAllocationGranularity);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuDeviceGet);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuDeviceGetAttribute);

}  // namespace nixl_ep_v2
