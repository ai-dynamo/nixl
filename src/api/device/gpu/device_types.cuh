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
#ifndef NIXL_SRC_API_DEVICE_GPU_DEVICE_TYPES_CUH
#define NIXL_SRC_API_DEVICE_GPU_DEVICE_TYPES_CUH

#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

namespace nixl::gpu {

struct xferStatusH {
    alignas(16) unsigned char storage[64] = {};
};

constexpr size_t xfer_status_payload_size = 64;

static_assert(xfer_status_payload_size == sizeof(xferStatusH));

enum class level_t : uint64_t { THREAD = 0, WARP = 1, BLOCK = 2, GRID = 3 };

namespace flags {
    constexpr uint64_t defer = 1;
} // namespace flags

struct memViewElem {
    nixlMemViewH mvh;
    size_t index; /**< Index in the memory view */
    size_t offset; /**< Offset within the buffer */
};

} // namespace nixl::gpu

using nixlGpuXferStatusH = nixl::gpu::xferStatusH;
using nixl_gpu_level_t = nixl::gpu::level_t;
using nixlMemViewElem = nixl::gpu::memViewElem;
constexpr size_t nixl_gpu_xfer_status_payload_size = nixl::gpu::xfer_status_payload_size;

namespace nixl_gpu_flags {
constexpr uint64_t defer = nixl::gpu::flags::defer;
} // namespace nixl_gpu_flags

#endif // NIXL_SRC_API_DEVICE_GPU_DEVICE_TYPES_CUH
