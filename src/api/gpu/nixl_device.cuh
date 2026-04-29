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

// Generic backend-agnostic entry point.  Selects an implementation at
// compile time via NIXL_GPU_DEVICE_BACKEND_UCX or
// NIXL_GPU_DEVICE_BACKEND_PROXY; exactly one must be defined.
//
// When the backend is already known at the include site prefer the
// backend-specific facade (ucx/nixl_device.cuh or proxy/nixl_device.cuh)
// which requires no macro.

#ifndef NIXL_SRC_API_GPU_NIXL_DEVICE_CUH
#define NIXL_SRC_API_GPU_NIXL_DEVICE_CUH

#include "common/nixl_device_api.cuh"
#include "common/nixl_device_wrappers.cuh"

#endif // NIXL_SRC_API_GPU_NIXL_DEVICE_CUH
