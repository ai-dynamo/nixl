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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RUNTIME_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RUNTIME_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend_aux.h"
#include "device/device_allocator.h"
#include "proxy_protocol.h"
#include "proxy_config.h"
#include "proxy_backend_ops.h"
#include "proxy_control_buffer.h"
#include "proxy_registry.h"

class ProxyWorker;

static constexpr size_t kProxyShutdownSlot = 0;
static constexpr size_t kProxyCiSlotBase = 1;

struct nixlProxyRequestState {
    uint64_t op_idx = 0;
    nixlBackendProxyRequest backend_request{};
    nixl_status_t status = NIXL_IN_PROG;
};

struct alignas(64) nixlProxyChannelState {
    nixlProxyChannelView device_view{};
    /** In-flight state retained until the completion frontier passes its ring slot. */
    std::vector<nixlProxyRequestState> inflight_slots_;
    /** Host-only submit frontier; consumer_idx_shadow_ remains the completion frontier. */
    uint64_t submit_idx_ = 0;
    /** Host shadow of the authoritative GPU-visible consumer index. */
    uint64_t consumer_idx_shadow_ = 0;

    /** Device-resident ring descriptor. */
    nixlDeviceMem work_ring_mem_;
    /** Mapped pinned host records; GPU writes via device alias, worker reads host alias. */
    nixlMappedHostMem records_mem_;
    /** Device-resident producer index; only the GPU updates it. */
    nixlDeviceMem producer_idx_mem_;
    /** Authoritative consumer count; CPU publishes through GDRCopy or mapped host memory. */
    uint64_t *consumer_idx_dev_ = nullptr;
    /** Device-resident cache of consumer_idx_dev_ used by GPU enqueue backpressure. */
    nixlDeviceMem consumer_idx_cache_mem_;
    nixlProxyControlBuffer *control_slots_ = nullptr;
    /** Remembered from allocate() so rearm() needs no arguments. */
    nixlDeviceAllocator *allocator_ = nullptr;
    size_t control_slot_index_ = 0;
    /** Host-side ring depth for the CPU worker; nixlProxyWorkRing itself is device-only. */
    uint32_t ring_depth_ = 0;
    /** Mapped pinned host completion slot; worker writes host alias, GPU polls device alias. */
    nixlMappedHostMem completion_slot_mem_;

    nixlProxyChannelState() = default;
    ~nixlProxyChannelState() = default;
    nixlProxyChannelState(nixlProxyChannelState &&) noexcept = default;
    nixlProxyChannelState &
    operator=(nixlProxyChannelState &&) noexcept = default;
    nixlProxyChannelState(const nixlProxyChannelState &) = delete;
    nixlProxyChannelState &
    operator=(const nixlProxyChannelState &) = delete;

    nixl_status_t
    allocate(nixlDeviceAllocator &allocator,
             uint32_t depth,
             nixlProxyControlBuffer *control_slots,
             size_t control_slot_index);

    /** Reset an allocated ring after producers stop and backend work is quiescent. */
    nixl_status_t
    rearm() noexcept;

    /** No submitted or published work remains. */
    [[nodiscard]] bool
    drained() const noexcept;

    /** All producer tickets must have reached terminal completion before reclamation. */
    void
    verifyDrained() const noexcept;

    nixl_status_t
    publishConsumerIdx(uint64_t value) noexcept;

    nixlProxySubmission *
    recordsHost() const noexcept {
        return records_mem_.asHost<nixlProxySubmission>();
    }

    nixlProxyCompletionSlot *
    completionSlotHost() const noexcept {
        return completion_slot_mem_.asHost<nixlProxyCompletionSlot>();
    }

    bool
    allocated() const {
        return static_cast<bool>(work_ring_mem_);
    }

    void
    deallocate() noexcept;
};

class nixlProxyRuntime {
public:
    ~nixlProxyRuntime();

    nixlProxyRuntime(nixlProxyRuntime &&) = delete;
    nixlProxyRuntime(const nixlProxyRuntime &) = delete;
    nixlProxyRuntime &
    operator=(nixlProxyRuntime &&) = delete;
    nixlProxyRuntime &
    operator=(const nixlProxyRuntime &) = delete;

    /** Build without starting workers; allocator must outlive the runtime. */
    [[nodiscard]] static nixl_status_t
    create(nixlProxyBackendOps backend_ops,
           const nixlProxyConfig &config,
           std::unique_ptr<nixlProxyRuntime> &out,
           nixlDeviceAllocator &allocator = nixlGetDeviceAllocator());

    [[nodiscard]] nixl_status_t
    prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    /** Resolves the backend's direct pointers first, when it offers any. */
    [[nodiscard]] nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    [[nodiscard]] nixl_status_t
    unregisterProxyMemView(nixlMemViewH proxy_memview);

    [[nodiscard]] nixl_status_t
    startWorkers();

    nixl_status_t
    shutdown();

    const nixlProxyChannelView *
    deviceChannelViews() const {
        return device_channel_views_.empty() ? nullptr : device_channel_views_.data();
    }

    nixlProxyDeviceContextData *
    deviceContext() const {
        return device_context_mem_.as<nixlProxyDeviceContextData>();
    }

private:
    /** Ask owning workers to drain and reset, then wait for their acknowledgments. */
    void
    drainChannels() noexcept;

    nixlProxyRuntime(nixlProxyBackendOps backend_ops,
                     const nixlProxyConfig &config,
                     nixlDeviceAllocator &allocator) noexcept;

    /** Allocate rings, device context and workers; see create(). */
    nixl_status_t
    build();

    void
    joinWorkerThreads() noexcept;

    nixlDeviceAllocator &allocator_;
    nixlProxyBackendOps backend_ops_;
    nixlProxyConfig config_;
    mutable std::mutex control_mutex_;
    std::vector<nixlProxyChannelState> channels_;
    nixlProxyControlBuffer control_slots_;
    std::vector<nixlProxyChannelView> device_channel_views_;
    nixlDeviceMem device_channel_views_mem_;
    nixlDeviceMem device_context_mem_;
    std::vector<std::unique_ptr<ProxyWorker>> workers_;
    /** Created after the device context; destroyed after workers stop. */
    std::unique_ptr<nixlProxyMemViewRegistry> memview_registry_;
    alignas(64) std::atomic<uint64_t> shutdown_state_{
        static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN)};
    /** Bumped once per drain; each worker acks it when it has applied it. */
    alignas(64) std::atomic<uint64_t> drain_requested_{0};
    uint64_t *shutdown_word_dev_ = nullptr;
    bool workers_started_ = false;
};

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RUNTIME_H
