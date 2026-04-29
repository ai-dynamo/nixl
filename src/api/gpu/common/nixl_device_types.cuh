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
#ifndef NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_TYPES_CUH
#define NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_TYPES_CUH

#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

struct nixlGpuXferStatusH {
    alignas(16) unsigned char storage[64] = {};
};

enum class nixl_gpu_level_t : uint64_t { THREAD = 0, WARP = 1, BLOCK = 2, GRID = 3 };

namespace nixl_gpu_flags {
constexpr uint64_t defer = 1;
} // namespace nixl_gpu_flags

struct nixlMemViewElem {
    nixlMemViewH mvh;
    size_t index; /**< Index in the memory view */
    size_t offset; /**< Offset within the buffer */
};

#endif // NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_TYPES_CUH
