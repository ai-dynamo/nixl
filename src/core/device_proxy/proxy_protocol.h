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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H
#define NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <nixl_types.h>

enum class nixl_proxy_opcode_t : uint8_t {
    PUT = 0,
    ATOMIC_ADD = 1,
};

enum class nixl_proxy_control_state_t : uint32_t {
    RUNNING = 0,
    SHUTDOWN = 1,
};

constexpr uint16_t NIXL_PROXY_MEM_LIST_NAMESPACE = 0x8000;
constexpr uint16_t NIXL_PROXY_MEM_LIST_VERSION_V1 =
    NIXL_PROXY_MEM_LIST_NAMESPACE | 1;

using nixlProxyMemViewId = uint32_t;

enum class nixlProxyMemViewKind : uint8_t {
    LOCAL = 1,
    REMOTE = 2,
};

struct nixlProxyDeviceMemViewElem {
    void *direct_ptr;
};

/**
 * GPU-resident proxy memory-list handle.
 *
 * The version/length prefix deliberately matches both UCX device memory-list
 * representations. Generic Device API dispatch may therefore read only the
 * leading version before deciding which complete representation is present.
 */
struct nixlProxyDeviceMemView {
    uint16_t version;
    uint16_t reserved0;
    uint32_t length;

    nixlProxyMemViewId proxy_memview_id;
    nixlProxyMemViewKind kind;
    uint8_t reserved1[3];

    // Must remain the final member.
    nixlProxyDeviceMemViewElem mem_elements[0];
};

static_assert(std::is_standard_layout_v<nixlProxyDeviceMemView>);
static_assert(offsetof(nixlProxyDeviceMemView, version) == 0);
static_assert(offsetof(nixlProxyDeviceMemView, length) == 4);
static_assert(offsetof(nixlProxyDeviceMemView, mem_elements) % alignof(void *) == 0);

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

static_assert(std::is_standard_layout_v<nixlProxySubmission>,
              "submission layout must be predictable across CPU and GPU code");
static_assert(std::is_aggregate_v<nixlProxySubmission>,
              "submission must support designated aggregate initialization");
static_assert(std::is_trivially_copyable_v<nixlProxySubmission>,
              "for the compiler to optimize copies");
static_assert(std::has_unique_object_representations_v<nixlProxySubmission>,
              "submission must not contain implicit padding");
static_assert(sizeof(nixlProxySubmission) == 64, "submission must be 64 bytes");
static_assert(alignof(nixlProxySubmission) == 64, "submission must be 64 bytes aligned");
static_assert(offsetof(nixlProxySubmission, op_idx) == 0,
              "op_idx must be the first word because it publishes record readiness");

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
    nixl_status_t next_status = NIXL_IN_PROG;
};

/**
 * Device-visible view of one ring: channel_id's ring for destination peer_index.
 * Storage is a row-major [num_channels][peer_capacity] matrix; the SW ordering
 * entity is the channel, which owns one ring per dest peer.
 */
struct nixlProxyChannelView {
    nixlProxyWorkRing *work_ring = nullptr;
    /** Mapped pinned host memory (device alias); host writes via host pointer with atomics. */
    nixlProxyCompletionSlot *completion_slot = nullptr;
    uint32_t peer_index = 0;
    /** Logical channel id (ordering entity); ring is the (channel_id, peer_index) cell. */
    uint32_t channel_id = 0;
};

struct nixlProxyDeviceContextData {
    /**
     * Row-major [num_channels][peer_capacity] matrix of ring views.
     * Index as channel_id * peer_capacity + peer_index: channel owns one ring per dest.
     */
    nixlProxyChannelView *channels = nullptr;
    uint32_t peer_capacity = 0;
    /** Number of logical channels; each has one ring per destination peer slot. */
    uint32_t num_channels = 0;
    /** Runtime-wide shutdown signal shared by every peer and channel. */
    uint64_t *shutdown_word = nullptr;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H
