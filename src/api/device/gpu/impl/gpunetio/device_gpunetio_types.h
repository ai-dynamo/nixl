/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_TYPES_H
#define NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_TYPES_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Keep this header CUDA- and DOCA-header-free: host preparation and CUDA
// consumers share these POD layouts.  The complete type is required only by
// device_gpunetio_compat.cuh.
struct doca_gpu_dev_verbs_qp;

namespace nixl::gpu::impl::gpunetio {

constexpr uint32_t gpunetio_device_view_abi_version = 1;
constexpr uint32_t gpunetio_device_view_role_local = 1;
constexpr uint32_t gpunetio_device_view_role_remote = 2;
constexpr uint32_t gpunetio_local_peer_slot = ~uint32_t{0};
constexpr uint32_t gpunetio_remote_peer_slot = 0;

constexpr uint32_t gpunetio_lane_phase_idle = 0;
constexpr uint32_t gpunetio_lane_phase_posting = 1;
constexpr uint32_t gpunetio_lane_phase_posted = 2;
constexpr uint32_t gpunetio_lane_phase_failed = 3;

enum class GpunetioLanePhase : uint32_t {
    IDLE = gpunetio_lane_phase_idle,
    POSTING = gpunetio_lane_phase_posting,
    POSTED = gpunetio_lane_phase_posted,
    FAILED = gpunetio_lane_phase_failed,
};

struct alignas(16) GpunetioDeviceLane {
    doca_gpu_dev_verbs_qp *qp;
    alignas(8) uint64_t generation;
    alignas(8) uint64_t active_ticket;
    alignas(8) uint64_t last_success_generation;
    alignas(4) uint32_t phase;
    alignas(4) uint32_t failed;
};

// The host allocates this once per native engine.  Every native local and
// remote view for that engine carries the same context pointer.
struct GpunetioDeviceContext {
    GpunetioDeviceLane *lane;
};

struct GpunetioViewElem {
    uint64_t base;
    uint64_t length;
    uint32_t key;
    uint32_t peer_slot;
    uint32_t valid;
    uint32_t reserved;
};

struct GpunetioDeviceView {
    uint32_t abi_version;
    uint32_t role;
    uint64_t count;
    uint64_t context_cookie;
    uint32_t execution_gpu;
    uint32_t reserved;
    const GpunetioViewElem *elems;
    GpunetioDeviceContext *context;
};

struct GpunetioStatusPayload {
    GpunetioDeviceLane *lane;
    uint64_t generation;
    uint64_t ticket;
    int32_t cached_status;
    uint32_t initialized;
};

static_assert(sizeof(GpunetioViewElem) == 32);
static_assert(sizeof(GpunetioStatusPayload) == 32);
static_assert(sizeof(GpunetioStatusPayload) <= 60);
static_assert(alignof(GpunetioDeviceLane) >= 16);
static_assert(std::is_standard_layout_v<GpunetioDeviceLane>);
static_assert(std::is_trivial_v<GpunetioDeviceLane>);
static_assert(std::is_standard_layout_v<GpunetioDeviceContext>);
static_assert(std::is_trivial_v<GpunetioDeviceContext>);
static_assert(std::is_standard_layout_v<GpunetioDeviceView>);
static_assert(std::is_trivial_v<GpunetioDeviceView>);

} // namespace nixl::gpu::impl::gpunetio

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_TYPES_H
