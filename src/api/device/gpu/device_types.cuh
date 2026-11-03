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
/**
 * @file device_types.cuh
 * @brief Public types for the GPU Device API
 */

#ifndef NIXL_SRC_API_DEVICE_GPU_DEVICE_TYPES_CUH
#define NIXL_SRC_API_DEVICE_GPU_DEVICE_TYPES_CUH

#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

namespace nixl::gpu {

/**
 * Which implementation executes a prepared memory view, chosen per view by
 * the backend that prepared it, not per build. Zero is deliberately not a
 * valid mode, so an untagged handle reads as invalid rather than as the first
 * backend.
 *
 * DYNAMIC, the device API's default mode argument, reads the mode from the
 * view. A fixed mode must match the views (otherwise undefined; asserted in
 * debug builds), and its statuses must be polled with the same mode.
 */
enum class exec_mode_t : uint8_t {
    UCX_DIRECT = 1,
    PROXY = 2,
    DYNAMIC = 0xff,
};

/**
 * @brief Opaque handle for a GPU transfer request.
 *
 * Pass an optional pointer to put or atomicAdd, then poll with
 * @ref nixl::gpu::getXferStatus until the request completes.
 */
struct xferStatusH {
    alignas(16) unsigned char storage[60] = {}; /**< Backend-owned request payload */
    exec_mode_t mode{}; /**< Mode that accepted the request; set by the dispatch layer */
};

constexpr size_t xfer_status_payload_size = sizeof(xferStatusH::storage);

static_assert(sizeof(xferStatusH) == 64);

/**
 * @enum  level_t
 * @brief An enumeration of different levels for GPU transfer requests.
 */
enum class level_t : uint64_t { THREAD = 0, WARP = 1, BLOCK = 2, GRID = 3 };

/**
 * What a prepared memory-view handle actually points at: the mode, plus the
 * backend's own handle. Device code unwraps it on the dispatch path.
 */
struct memViewWrapper {
    exec_mode_t execution_mode;
    nixlMemViewH backend_memview;
};

namespace flags {
    /** @brief Defer the operation instead of flushing immediately. */
    constexpr uint64_t defer = 1;
} // namespace flags

/**
 * @brief A single element in a prepared memory view.
 */
struct memViewElem {
    nixlMemViewH mvh; /**< Memory view handle */
    size_t index; /**< Index in the memory view */
    size_t offset; /**< Offset within the buffer */
};

} // namespace nixl::gpu

/**
 * @brief Opaque handle for a GPU transfer request.
 * @see nixl::gpu::xferStatusH
 */
using nixlGpuXferStatusH = nixl::gpu::xferStatusH;
/**
 * @brief An enumeration of different levels for GPU transfer requests.
 * @see nixl::gpu::level_t
 */
using nixl_gpu_level_t = nixl::gpu::level_t;
/**
 * @brief A single element in a prepared memory view.
 * @see nixl::gpu::memViewElem
 */
using nixlMemViewElem = nixl::gpu::memViewElem;
using nixl_device_exec_mode_t = nixl::gpu::exec_mode_t;
using nixlDeviceMemViewWrapper = nixl::gpu::memViewWrapper;
constexpr size_t nixl_gpu_xfer_status_payload_size = nixl::gpu::xfer_status_payload_size;

namespace nixl_gpu_flags {
/** @brief Defer the operation instead of flushing immediately. */
constexpr uint64_t defer = nixl::gpu::flags::defer;
} // namespace nixl_gpu_flags

#endif // NIXL_SRC_API_DEVICE_GPU_DEVICE_TYPES_CUH
