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
#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_PROXY_DEVICE_PROXY_RING_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_PROXY_DEVICE_PROXY_RING_CUH

#include <cuda/atomic>

#include "../../device_types.cuh"
#include "../level_utils.cuh"
#include <device/proxy/proxy_protocol.h>

namespace nixl::gpu::impl::proxy {

// Overlay struct written into xferStatusH::storage by enqueue()
// and read back by pollXferStatus().  Must fit within the 64-byte opaque blob.
struct ProxyXferStatus {
    nixlProxyCompletionSlot *slot; // device pointer to the channel's nixlProxyCompletionSlot
    uint64_t op_idx;
};

static_assert(sizeof(ProxyXferStatus) <= xfer_status_payload_size,
              "ProxyXferStatus must fit in the transfer-status payload");

__device__ __forceinline__ uint64_t
proxyHostViewFromHandle(nixlMemViewH mvh) {
    return static_cast<const nixlProxyDeviceMemView *>(mvh)->host_view;
}

struct ProxyDeviceContext;

__device__ __forceinline__ const ProxyDeviceContext *
proxyContextFromMemView(const nixlProxyDeviceMemView *memview) {
    if (memview == nullptr || memview->context == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<const ProxyDeviceContext *>(memview->context);
}

struct ProxyDeviceContext : nixlProxyDeviceContextData {

    __device__ __forceinline__ size_t
    ringIndex(uint32_t peer_index, uint32_t channel_id) const {
        return static_cast<size_t>(channel_id) * max_peers + peer_index;
    }

    __device__ inline nixl_status_t
    enqueue(nixlProxyCommand command,
            uint32_t channel_id,
            xferStatusH *xfer_status = nullptr) const {
        if (command.peerIndex() >= max_peers || num_channels == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        channel_id %= num_channels;

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> shut(*shutdown_word);
        if (shut.load(cuda::memory_order_relaxed) ==
            static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN)) {
            return NIXL_ERR_BACKEND;
        }

        nixlProxyRingDesc &ring_desc = rings[ringIndex(command.peerIndex(), channel_id)];
        if (ring_desc.work_ring == nullptr || ring_desc.completion_slot == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }
        nixlProxyWorkRing *ring = ring_desc.work_ring;

        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> producer_idx(*ring->producer_idx);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> cons(*ring->consumer_idx);

        // Atomically claim a unique slot in the ring.
        const uint64_t ticket = producer_idx.fetch_add(1, cuda::memory_order_relaxed);

        // Fast path: use the device cache. Refresh from host only if the ring
        // appears full, since mapped-host loads are much slower than HBM loads.
        uint64_t cached_consumer_idx = *ring->consumer_idx_cache;
        while (ticket - cached_consumer_idx >= ring->depth) {
            cached_consumer_idx = cons.load(cuda::memory_order_acquire);
            *ring->consumer_idx_cache = cached_consumer_idx;

            if (shut.load(cuda::memory_order_relaxed) ==
                static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN)) {
                return NIXL_ERR_BACKEND;
            }
        }

        const uint64_t submission_op_idx = ticket + 1;
        const uint32_t slot = static_cast<uint32_t>(ticket % ring->depth);

        // Signal this slot is ready for the consumer. The release
        // guarantees the command write above is visible before the
        // consumer reads op_idx via an acquire load. op_idx == 0 means empty.
        command.op_idx = 0;
        ring->commands[slot] = command;

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> command_op_idx(
            ring->commands[slot].op_idx);
        command_op_idx.store(submission_op_idx, cuda::memory_order_release);

        if (xfer_status != nullptr) {
            ProxyXferStatus pxs{ring_desc.completion_slot, submission_op_idx};
            memcpy(xfer_status->storage, &pxs, sizeof(ProxyXferStatus));
        }

        return NIXL_IN_PROG;
    }

    // Poll the completion slot recorded by enqueue().
    //
    // The completion slot implements collapsed-CQ semantics:
    // - op_idx >= failed_idx != 0 => the first failure latched the channel; report its status
    // - completed_idx >= op_idx   => this op completed, so it succeeded
    // - otherwise                 => this op is still pending
    __device__ inline static nixl_status_t
    pollXferStatus(const xferStatusH &xfer_status) {
        const ProxyXferStatus *pxs = reinterpret_cast<const ProxyXferStatus *>(xfer_status.storage);
        if (pxs->slot == nullptr) {
            return NIXL_ERR_BACKEND;
        }

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> comp_idx(pxs->slot->completed_idx);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> fail_idx(pxs->slot->failed_idx);
        const uint64_t completed_idx = comp_idx.load(cuda::memory_order_acquire);
        const uint64_t failed_idx = fail_idx.load(cuda::memory_order_acquire);

        if (failed_idx != 0 && pxs->op_idx >= failed_idx) {
            return pxs->slot->completion_status;
        }
        return completed_idx >= pxs->op_idx ? NIXL_SUCCESS : NIXL_IN_PROG;
    }
};

} // namespace nixl::gpu::impl::proxy

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_PROXY_DEVICE_PROXY_RING_CUH
