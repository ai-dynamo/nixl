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

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

// The accessors below have to be callable from device code, but this header
// must not include a CUDA header to say so.
#ifdef __CUDACC__
#define NIXL_PROXY_PROTO_FN __host__ __device__ inline
#else
#define NIXL_PROXY_PROTO_FN inline
#endif

namespace nixl {

enum class proxy_opcode_t : uint8_t {
    PUT = 0,
    ATOMIC_ADD = 1,
};

enum class proxy_control_state_t : uint32_t {
    RUNNING = 0,
    SHUTDOWN = 1,
};

struct proxyDeviceContextData;

/**
 * A prepared memory view as device code sees it; `direct_ptr_count` pointers follow it,
 * then one peer per remote descriptor.
 */
struct proxyDeviceMemView {
    uint64_t host_view = 0;
    const proxyDeviceContextData *context = nullptr;
    uint32_t direct_ptr_count = 0;
    uint32_t peer_count = 0;
};

NIXL_PROXY_PROTO_FN void **
proxyDeviceMemViewDirectPtrs(proxyDeviceMemView *view) {
    return reinterpret_cast<void **>(view + 1);
}

NIXL_PROXY_PROTO_FN void *const *
proxyDeviceMemViewDirectPtrs(const proxyDeviceMemView *view) {
    return reinterpret_cast<void *const *>(view + 1);
}

NIXL_PROXY_PROTO_FN uint32_t
proxyDeviceMemViewPeer(const proxyDeviceMemView *view, size_t index) {
    assert(index < view->peer_count);
    return reinterpret_cast<const uint32_t *>(proxyDeviceMemViewDirectPtrs(view) +
                                              view->direct_ptr_count)[index];
}

/** Bytes to allocate for a view carrying `ptr_count` direct pointers and `peer_count` peers. */
NIXL_PROXY_PROTO_FN size_t
proxyDeviceMemViewBytes(size_t ptr_count, size_t peer_count) {
    return sizeof(proxyDeviceMemView) + ptr_count * sizeof(void *) + peer_count * sizeof(uint32_t);
}

/**
 * A GPU-submitted PUT or atomic-add operation for the CPU proxy to execute.
 *
 * What each field means, by opcode:
 *
 *   | field      | PUT                     | ATOMIC_ADD          |
 *   |------------|-------------------------|---------------------|
 *   | operand    | source offset           | value to add        |
 *   | dst_offset | destination offset      | counter offset      |
 *   | size       | bytes                   | unused              |
 *   | src_view   | source view token       | unused              |
 *   | src_index  | source descriptor       | unused              |
 *   | dst_view   | destination view token  | counter view token  |
 *   | dst_index  | destination descriptor  | counter descriptor  |
 *
 * The ring the command is written into implies its channel and peer.
 * For ATOMIC_ADD the size field is unused and the host uses 8 bytes.
 */
struct alignas(64) proxyCommand {
    uint64_t op_idx = 0;
    uint64_t operand = 0; // PUT: source offset; ATOMIC_ADD: value.
    uint64_t dst_offset = 0;
    uint64_t size = 0;
    uint64_t src_view = 0;
    uint64_t dst_view = 0;
    uint32_t src_index = 0;
    uint32_t dst_index = 0;
    proxy_opcode_t opcode = proxy_opcode_t::PUT;
    /** Reserved. Keeps the command 64 bytes. */
    uint8_t reserved[7] = {};

    /** PUT: byte offset into the source descriptor. */
    NIXL_PROXY_PROTO_FN uint64_t
    sourceOffset() const {
        return operand;
    }

    /** ATOMIC_ADD: the value to add to the counter. */
    NIXL_PROXY_PROTO_FN uint64_t
    atomicValue() const {
        return operand;
    }
};

struct proxyWorkRing {
    /** Mapped host commands: GPU writes via device alias; CPU worker reads host alias. */
    proxyCommand *commands = nullptr;
    /** Device-resident producer index; only the GPU updates it. */
    uint64_t *producer_idx = nullptr;
    /** Authoritative consumer index. The CPU publishes it through host-published device memory. */
    uint64_t *consumer_idx = nullptr;
    /** Device-resident cached consumer index; GPU refreshes from consumer_idx only when full. */
    uint64_t *consumer_idx_cache = nullptr;
    /** The depth of the work ring. */
    uint32_t depth = 0;
};

/**
 * The host writes completion_status, failed_idx, then completed_idx; a poller reads completed_idx
 * before failed_idx, so it reads completion_status only behind the failure that wrote it.
 */
struct alignas(16) proxyCompletionSlot {
    uint64_t completed_idx = 0;
    /** op_idx of the first failed op, or 0 while none has failed. */
    uint64_t failed_idx = 0;
    /** Status of the op at completed_idx; an error stays latched. */
    nixl_status_t completion_status = NIXL_IN_PROG;
};

struct proxyRingDesc {
    proxyWorkRing *work_ring = nullptr;
    /** Mapped pinned host memory (device alias); host writes via host pointer with atomics. */
    proxyCompletionSlot *completion_slot = nullptr;
};

struct proxyDeviceContextData {
    proxyRingDesc *rings = nullptr;
    uint32_t max_peers = 0;
    uint32_t num_channels = 0;
    uint64_t *shutdown_word = nullptr;
};

static_assert(sizeof(proxyCommand) == 64, "proxyCommand must be 64 bytes");
static_assert(offsetof(proxyCommand, op_idx) == 0,
              "op_idx must be the first word because it publishes command readiness");
static_assert(alignof(proxyCommand) == 64, "proxyCommand must be cache-line aligned");

static_assert(sizeof(proxyDeviceMemView) == 24, "proxyDeviceMemView size changed");
static_assert(offsetof(proxyDeviceMemView, host_view) == 0, "proxyDeviceMemView layout changed");
static_assert(sizeof(proxyDeviceMemView) % alignof(void *) == 0,
              "the trailing direct-pointer run must start aligned");

static_assert(sizeof(proxyWorkRing) == 40, "proxyWorkRing size changed");
static_assert(sizeof(proxyCompletionSlot) == 32, "proxyCompletionSlot size changed");
static_assert(offsetof(proxyCompletionSlot, completed_idx) == 0,
              "proxyCompletionSlot layout changed");
static_assert(offsetof(proxyCompletionSlot, failed_idx) == 8, "proxyCompletionSlot layout changed");
static_assert(sizeof(nixl_status_t) == 4, "nixl_status_t must be 4 bytes in a completion slot");
static_assert(sizeof(proxyRingDesc) == 16, "proxyRingDesc size changed");
static_assert(sizeof(proxyDeviceContextData) == 24, "proxyDeviceContextData size changed");
static_assert(offsetof(proxyDeviceContextData, rings) == 0,
              "proxyDeviceContextData layout changed");

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_PROTOCOL_H
