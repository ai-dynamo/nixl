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

constexpr uint32_t gpunetio_device_view_abi_version = 3;
constexpr uint32_t gpunetio_device_view_role_local = 1;
constexpr uint32_t gpunetio_device_view_role_remote = 2;
constexpr uint32_t gpunetio_local_peer_slot = ~uint32_t{0};
constexpr uint32_t gpunetio_remote_peer_slot = 0;

constexpr uint32_t gpunetio_lane_phase_idle = 0;
constexpr uint32_t gpunetio_lane_phase_posting = 1;
constexpr uint32_t gpunetio_lane_phase_posted = 2;
constexpr uint32_t gpunetio_lane_phase_failed = 3;
constexpr uint32_t gpunetio_lane_phase_scalar = 4;

struct alignas(16) GpunetioDeviceLane {
    doca_gpu_dev_verbs_qp *qp;
    alignas(8) uint64_t generation;
    // One past the last accepted standard-operation ticket.
    alignas(8) uint64_t active_ticket;
    // First ticket whose completion has not been checked successfully.
    alignas(8) uint64_t next_retire;
    alignas(4) uint32_t phase;
    alignas(4) uint32_t failed;
    // Standard GET queue capacity, stored by the first GET whose QP checks pass and 0 until
    // then. The host never sets it. It takes what was tail padding, which hosts zero.
    alignas(4) uint32_t capacity;
};

// Each prepared context retains a selected lane for the batch extension.
// Standard operations select from the shared channel array at call time.
struct GpunetioDeviceContext {
    GpunetioDeviceLane *lane;
    GpunetioDeviceLane *channels;
    uint32_t channel_count;
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
    uint64_t ticket;
    int32_t cached_status;
    uint32_t initialized;
};

static_assert(sizeof(GpunetioViewElem) == 32);
static_assert(sizeof(GpunetioStatusPayload) == 24);
static_assert(sizeof(GpunetioStatusPayload) <= 60);
static_assert(sizeof(GpunetioDeviceLane) == 48);
static_assert(offsetof(GpunetioDeviceLane, capacity) == 40);
static_assert(alignof(GpunetioDeviceLane) >= 16);
static_assert(std::is_standard_layout_v<GpunetioDeviceLane>);
static_assert(std::is_trivial_v<GpunetioDeviceLane>);
static_assert(std::is_standard_layout_v<GpunetioDeviceContext>);
static_assert(std::is_trivial_v<GpunetioDeviceContext>);
static_assert(std::is_standard_layout_v<GpunetioDeviceView>);
static_assert(std::is_trivial_v<GpunetioDeviceView>);

} // namespace nixl::gpu::impl::gpunetio

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_TYPES_H
