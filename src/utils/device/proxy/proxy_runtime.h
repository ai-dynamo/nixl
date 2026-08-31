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
    /**
     * Per-ring-slot backend state. A submitted record remains associated with
     * its ring slot until completion advances consumer_idx_shadow_ past it.
     */
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
    uint32_t         ring_depth_         = 0;
    /** Mapped pinned host completion slot; worker writes host alias, GPU polls device alias. */
    nixlMappedHostMem completion_slot_mem_;

    nixlProxyChannelState() = default;
    ~nixlProxyChannelState() = default;
    nixlProxyChannelState(nixlProxyChannelState &&) noexcept = default;
    nixlProxyChannelState &operator=(nixlProxyChannelState &&) noexcept = default;
    nixlProxyChannelState(const nixlProxyChannelState &) = delete;
    nixlProxyChannelState &operator=(const nixlProxyChannelState &) = delete;

    nixl_status_t
    allocate(nixlDeviceAllocator &allocator,
             uint32_t depth,
             nixlProxyControlBuffer *control_slots,
             size_t control_slot_index);

    /**
     * Put an allocated ring back to the state allocate() left it in - empty,
     * indices zeroed on both sides, completion latch cleared - without
     * releasing any memory. The caller must have established that nobody is
     * producing into or consuming from the ring.
     */
    nixl_status_t
    rearm() noexcept;

    /**
     * Hand every request still in flight back to the backend and forget it.
     * Returns how many were released.
     *
     * This releases bookkeeping only - no transport can abort an operation the
     * proxy has already posted (repo docs/issues/006 B6). A released transfer
     * may still read its source buffer and land remotely until the transport
     * reports the peer gone, so the memory behind one must stay mapped until
     * the endpoint is torn down.
     */
    size_t
    releaseInflightRequests(const nixlProxyBackendOps &backend_ops) noexcept;

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
        nixlProxyRuntime& operator=(nixlProxyRuntime &&) = delete;
        nixlProxyRuntime& operator=(const nixlProxyRuntime &) = delete;

        /**
         * Build a ready - but not yet running - runtime, or fail leaving `out`
         * untouched. All device memory comes from `allocator`, which must
         * outlive the runtime. startWorkers() is a separate, later call: the
         * worker threads call back into the backend that owns the runtime, so
         * the owner has to be fully constructed first.
         */
        [[nodiscard]] static nixl_status_t
        create(nixlProxyBackendOps backend_ops,
               const nixlProxyConfig &config,
               std::unique_ptr<nixlProxyRuntime> &out,
               nixlDeviceAllocator &allocator = nixlGetDeviceAllocator());

        nixl_status_t
        loadRemoteConnInfo(const std::string &remote_name,
                           const nixl_blob_t &conn_info);

        nixl_status_t
        remoteDisconnected(const std::string &remote_name);

        [[nodiscard]] nixl_status_t
        prepMemView(const nixl_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

        /** Resolves the backend's direct pointers first, when it offers any. */
        [[nodiscard]] nixl_status_t
        prepMemView(const nixl_remote_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

        [[nodiscard]] nixl_status_t
        unregisterProxyMemView(nixlMemViewH proxy_memview);

        [[nodiscard]] bool
        resolveProxyMemView(nixlMemViewH proxy_memview,
                            nixlMemViewH &backend_memview) const;

        [[nodiscard]] nixl_status_t
        startWorkers();

        nixl_status_t
        shutdown();

        const nixlProxyMemViewRegistry &
        memviewRegistry() const { return *memview_registry_; }

        const nixlProxyChannelView *
        deviceChannelViews() const {
            return device_channel_views_.empty() ? nullptr : device_channel_views_.data();
        }

        nixlProxyDeviceContextData *
        deviceContext() const { return device_context_mem_.as<nixlProxyDeviceContextData>(); }

    private:
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
        std::vector<nixlProxyChannelState> channels_;
        nixlProxyControlBuffer control_slots_;
        std::vector<nixlProxyChannelView> device_channel_views_;
        nixlDeviceMem device_channel_views_mem_;
        nixlDeviceMem device_context_mem_;
        std::vector<std::unique_ptr<ProxyWorker>> workers_;
        /** Built in build(), once the device context it stamps into memviews
         *  exists; destroyed by shutdown(), which frees every memview with it. */
        std::unique_ptr<nixlProxyMemViewRegistry> memview_registry_;
        alignas(64) std::atomic<uint64_t> shutdown_state_{
            static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN)};
        uint64_t *shutdown_word_dev_ = nullptr;
        bool workers_started_ = false;
};

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RUNTIME_H
