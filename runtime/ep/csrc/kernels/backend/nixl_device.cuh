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

#include <cuda_runtime.h>
#include <nixl_ep_v2/common/nixl_transport.cuh>

namespace nixl_ep_v2::elastic::nixl {

// Fill `context.p2p_ptrs` from `context.remote_mvh`: `nixlGetPtr` (a device function) for every connected peer (a pointer
// iff UCX gave the peer its CUDA IPC lane), the local base for self, nullptr otherwise; the context travels by value (a staged view, a scratch table)
cudaError_t cache_peer_pointers(const NIXLDeviceContext& context, const int* connected, cudaStream_t stream);

}  // namespace nixl_ep_v2::elastic::nixl
