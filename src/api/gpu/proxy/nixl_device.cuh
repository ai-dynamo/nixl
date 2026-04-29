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

// Proxy-backend device API facade.  Self-contained: no backend-selection
// macro required.  Include this header (or place src/api/gpu/proxy on the
// include path so that <nixl_device.cuh> resolves here) when building CUDA
// translation units that submit work through the CPU proxy runtime.

#ifndef NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_CUH
#define NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_CUH

#include "nixl_device_impl.cuh"

namespace nixl::gpu {
namespace selected_impl = proxy_impl;
}

#include "../common/nixl_device_wrappers.cuh"

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_CUH
