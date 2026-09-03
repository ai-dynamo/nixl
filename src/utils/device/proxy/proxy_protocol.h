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

/*
 * The wire contract between GPU device code and the host-side CPU proxy.
 *
 * This header is installed and compiled into user device code, so the host
 * runtime and a user's kernels can be built at different times against
 * different versions of it. The rules that keep that safe:
 *
 *   - Frozen except additively. Never reorder or retype an existing field.
 *     New data goes in `flags`, in `reserved`, or behind a new opcode. Every
 *     shared struct carries a size assert below; if one fires, the change
 *     under it is an ABI break, not a compile error to be silenced.
 *
 *   - Plain C++ only. No CUDA includes and no std::atomic members. Device
 *     code reaches these fields through cuda::atomic views and host code
 *     through __atomic builtins; making a member atomic would change the
 *     layout and break device compilation.
 *
 *   - Nothing backend-specific ever enters the ring. A submission names
 *     memory views by id and offset; the backend's own handles are resolved
 *     host-side, after the record is dequeued.
 *
 * Bump kProxyProtocolVersion whenever an additive change still needs old
 * device code to be rejected rather than silently misread.
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

/** Incompatible-layout counter; see the evolution rules above. */
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

/**
 * A prepared memory view as device code sees it. Allocated with a trailing
 * run of `direct_ptr_count` pointers, so the object is larger than
 * sizeof(nixlProxyDeviceMemView); reach the run through directPtrs().
 */
struct nixlProxyDeviceMemView {
    uint32_t proxy_memview_id = 0;
    uint32_t direct_ptr_count = 0;
    const nixlProxyDeviceContextData *context = nullptr;
};

/*
 * A zero-length array member would be a GNU extension in an installed public
 * header, which pedantic and MSVC consumers reject. The run starts at the
 * first correctly aligned byte past the struct, which is its own end: the
 * struct's alignment is that of a pointer and its size is a multiple of it.
 */
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
    nixl_status_t next_status = NIXL_IN_PROG;
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
    /**
     * kProxyProtocolVersion as the host runtime that published this context
     * was built with. Appended rather than placed first, because appending
     * into existing tail padding is the only change to this struct that is
     * not a reorder. Device code compares it once, when it acquires a memory
     * view, so a kernel built against a different protocol fails loudly
     * instead of misreading the rings.
     */
    uint32_t protocol_version = kProxyProtocolVersion;
};

/*
 * Layout locks. A submission is the record the GPU publishes into the ring,
 * so its size and the offset of its readiness word are the two things a
 * mismatched build would corrupt silently; the rest are pinned so that an
 * accidental field reorder shows up here instead of in a wrong transfer.
 */
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
