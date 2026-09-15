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

#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_PROTOCOL_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_PROTOCOL_H

#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

// The trailing-array accessor below has to be callable from device code, but
// this header must not include a CUDA header to say so.
#ifdef __CUDACC__
#define NIXL_PROXY_PROTO_FN __host__ __device__ inline
#else
#define NIXL_PROXY_PROTO_FN inline
#endif

/** Increment when the wire layout becomes incompatible. */
inline constexpr uint32_t kProxyProtocolVersion = 1;

enum class nixl_proxy_opcode_t : uint8_t {
    PUT = 0,
    ATOMIC_ADD = 1,
};

enum class nixl_proxy_control_state_t : uint32_t {
    RUNNING = 0,
    SHUTDOWN = 1,
};

struct nixlProxyDeviceContextData;

/** A prepared memory view as device code sees it; `direct_ptr_count` pointers follow it. */
struct nixlProxyDeviceMemView {
    uint32_t proxy_memview_id = 0;
    uint32_t direct_ptr_count = 0;
    const nixlProxyDeviceContextData *context = nullptr;
};

NIXL_PROXY_PROTO_FN void **
nixlProxyDeviceMemViewDirectPtrs(nixlProxyDeviceMemView *view) {
    return reinterpret_cast<void **>(view + 1);
}

NIXL_PROXY_PROTO_FN void *const *
nixlProxyDeviceMemViewDirectPtrs(const nixlProxyDeviceMemView *view) {
    return reinterpret_cast<void *const *>(view + 1);
}

/** Bytes to allocate for a view carrying `count` direct pointers. */
NIXL_PROXY_PROTO_FN size_t
nixlProxyDeviceMemViewBytes(size_t count) {
    return sizeof(nixlProxyDeviceMemView) + count * sizeof(void *);
}

/** A GPU-submitted PUT or atomic-add operation for the CPU proxy to execute. */
struct alignas(64) nixlProxySubmission {
    uint64_t op_idx = 0;
    uint64_t value = 0;
    uint64_t src_offset = 0;
    uint64_t dst_offset = 0;
    uint64_t size = 0;
    nixl_proxy_opcode_t opcode = nixl_proxy_opcode_t::PUT;
    uint8_t flags = 0;
    uint16_t channel_id = 0;
    uint32_t reserved = 0;
    uint32_t src_index = 0;
    uint32_t dst_index = 0;
    uint32_t src_proxy_memview_id = 0;
    uint32_t dst_proxy_memview_id = 0;
};

struct nixlProxyWorkRing {
    /** Mapped host records: GPU writes via device alias; CPU worker reads host alias. */
    nixlProxySubmission *records = nullptr;
    /** Device-resident producer index; only the GPU updates it. */
    uint64_t *producer_idx = nullptr;
    /** Authoritative consumer index; CPU publishes through GDRCopy or mapped host memory. */
    uint64_t *consumer_idx = nullptr;
    /** Device-resident cached consumer index; GPU refreshes from consumer_idx only when full. */
    uint64_t *consumer_idx_cache = nullptr;
    /** The depth of the work ring. */
    uint32_t depth = 0;
};

struct alignas(16) nixlProxyCompletionSlot {
    uint64_t completed_idx = 0;
    /** Status of the op at completed_idx; an error stays latched. */
    nixl_status_t completion_status = NIXL_IN_PROG;
};

struct nixlProxyChannelView {
    nixlProxyWorkRing *work_ring = nullptr;
    /** Mapped pinned host memory (device alias); host writes via host pointer with atomics. */
    nixlProxyCompletionSlot *completion_slot = nullptr;
};

struct nixlProxyDeviceContextData {
    nixlProxyChannelView *channels = nullptr;
    uint32_t max_peers = 0;
    uint32_t num_channels = 0;
    uint64_t *shutdown_word = nullptr;
    /** Host protocol version, checked by device code for compatibility. */
    uint32_t protocol_version = kProxyProtocolVersion;
};

static_assert(sizeof(nixlProxySubmission) == 64, "nixlProxySubmission must be 64 bytes");
static_assert(offsetof(nixlProxySubmission, op_idx) == 0,
              "op_idx must be the first word because it publishes record readiness");
static_assert(alignof(nixlProxySubmission) == 64, "nixlProxySubmission must be cache-line aligned");

static_assert(sizeof(nixlProxyDeviceMemView) == 16, "nixlProxyDeviceMemView layout changed");
static_assert(offsetof(nixlProxyDeviceMemView, proxy_memview_id) == 0,
              "nixlProxyDeviceMemView layout changed");
static_assert(sizeof(nixlProxyDeviceMemView) % alignof(void *) == 0,
              "the trailing direct-pointer run must start aligned");

static_assert(sizeof(nixlProxyWorkRing) == 40, "nixlProxyWorkRing layout changed");
static_assert(sizeof(nixlProxyCompletionSlot) == 16, "nixlProxyCompletionSlot layout changed");
static_assert(offsetof(nixlProxyCompletionSlot, completed_idx) == 0,
              "nixlProxyCompletionSlot layout changed");
static_assert(sizeof(nixlProxyChannelView) == 16, "nixlProxyChannelView layout changed");
static_assert(sizeof(nixlProxyDeviceContextData) == 32,
              "nixlProxyDeviceContextData layout changed");
static_assert(offsetof(nixlProxyDeviceContextData, channels) == 0,
              "nixlProxyDeviceContextData layout changed");

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_PROTOCOL_H
