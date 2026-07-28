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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_PROXY_RUNTIME_H
#define NIXL_SRC_CORE_DEVICE_PROXY_PROXY_RUNTIME_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend_aux.h"
#include "proxy_protocol.h"
#include "backend_adapter.h"
#include "gdrcopy/nixl_gdr_buffer.h"

class ProxyWorker;

static constexpr uint32_t kDefaultProxyRingDepth = 256;
/** Slot 0 of the shared control slab: proxy shutdown word. */
static constexpr size_t kProxyShutdownSlot = 0;
/** Consumer-index slots start at index 1: [shutdown][ci_0..N). */
static constexpr size_t kProxyCiSlotBase = 1;

enum class nixl_proxy_channel_lifecycle_t : uint8_t {
    UNALLOCATED = 0,
    INACTIVE = 1,
    ACTIVE = 2,
    RESET_PENDING = 3,
};

struct nixlProxyRequestState {
    uint64_t op_idx = 0;
    nixlBackendProxyRequest backend_request{};
    nixl_status_t status = NIXL_IN_PROG;
};

struct alignas(64) nixlProxyChannelState {
    nixlProxyChannelView device_view{};
    /**
     * Fixed request state indexed by ring slot. A slot remains occupied until
     * its backend request reaches a terminal state.
     */
    std::vector<nixlProxyRequestState> inflight_slots_;
    /** Host-only frontier advanced after a record is posted to the backend. */
    uint64_t submit_idx_ = 0;
    /** Host shadow of the authoritative GPU-visible consumer index. */
    uint64_t consumer_idx_shadow_ = 0;
    bool error_latched = false;

    nixlProxyWorkRing *work_ring_dev_ = nullptr;
    nixlProxySubmission *records_host_ = nullptr;
    /** Device-resident producer index; only the GPU updates it. */
    uint64_t *producer_idx_dev_ = nullptr;
    /** Authoritative consumer count; CPU publishes through GDRCopy or mapped host memory. */
    uint64_t *consumer_idx_dev_ = nullptr;
    /** Device-resident cache of consumer_idx_dev_ used by GPU enqueue backpressure. */
    uint64_t *consumer_idx_cache_dev_ = nullptr;
    nixlGdrBuffer *control_slots_ = nullptr;
    size_t control_slot_index_ = 0;
    /** Host-side ring depth for the CPU worker; nixlProxyWorkRing itself is device-only. */
    uint32_t ring_depth_ = 0;
    /** Mapped pinned host memory; proxy worker writes directly via host alias. */
    nixlProxyCompletionSlot *completion_slot_host_ = nullptr;
    /** Device-mapped alias of completion_slot_host_ for nixlProxyChannelView. */
    nixlProxyCompletionSlot *completion_slot_dev_ = nullptr;

    nixlProxyChannelState() = default;
    ~nixlProxyChannelState();
    nixlProxyChannelState(nixlProxyChannelState &&) noexcept;
    nixlProxyChannelState &
    operator=(nixlProxyChannelState &&) noexcept;
    nixlProxyChannelState(const nixlProxyChannelState &) = delete;
    nixlProxyChannelState &
    operator=(const nixlProxyChannelState &) = delete;

    nixl_status_t
    allocate(uint32_t peer_index,
             uint32_t channel_id,
             uint32_t depth,
             nixlGdrBuffer *control_slots,
             size_t control_slot_index);

    nixl_status_t
    publishConsumerIdx(uint64_t value) noexcept;

    bool
    allocated() const {
        return work_ring_dev_ != nullptr;
    }

    void
    deallocate() noexcept;

    /**
     * Local control-path reset owned by the channel's CPU worker (or the
     * runtime when workers are not running). Discards unsubmitted records and
     * local inflight bookkeeping; does not drain or cancel network work.
     */
    void
    resetLocalState() noexcept;
};

class nixlProxyMemViewRegistry {
public:
    nixl_status_t
    registerProxyMemView(nixlMemViewH backend_memview, nixlMemViewH *proxy_memview);

    nixl_status_t
    prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    nixl_status_t
    prepMemView(nixlMemViewH backend_memview,
                const nixl_meta_dlist_t &dlist,
                nixlMemViewH *proxy_memview);

    nixl_status_t
    prepMemView(nixlMemViewH backend_memview,
                const nixl_remote_meta_dlist_t &dlist,
                nixlMemViewH *proxy_memview);

    nixl_status_t
    unregisterProxyMemView(nixlMemViewH proxy_memview);

    nixl_status_t
    storeMetadata(nixlMemViewH proxy_memview, const nixl_meta_dlist_t &dlist);

    nixl_status_t
    storeMetadata(nixlMemViewH proxy_memview, const nixl_remote_meta_dlist_t &dlist);

    bool
    resolveProxyMemView(nixlMemViewH proxy_memview, nixlMemViewH &backend_memview) const;

    bool
    resolveProxyMemViewId(uint64_t proxy_memview_id, nixlMemViewH &backend_memview) const;

    nixl_status_t
    prepareSubmission(const nixlProxySubmission &submission,
                      nixlBackendProxySubmission &prepared_submission) const;

    void
    clear() noexcept;

private:
    struct ProxyMemViewRegStoredEntry {
        uintptr_t base_addr = 0;
        size_t len = 0;
        uint64_t dev_id = 0;
        nixlBackendMD *metadata = nullptr;
        // Remote agent for this element. Local entries leave this empty.
        std::string remote_agent;
    };

    struct LocalMetadata {
        nixl_mem_t mem_type = DRAM_SEG;
        std::vector<ProxyMemViewRegStoredEntry> entries;
    };

    struct RemoteMetadata {
        nixl_mem_t mem_type = DRAM_SEG;
        std::string remote_agent;
        std::vector<ProxyMemViewRegStoredEntry> entries;
    };

    enum class ProxyMemViewRegEntryState : uint8_t {
        ENTRY_ALLOCATED,
        ENTRY_READY,
        ENTRY_RETIRED,
    };

    enum class ProxyMemViewRegMetadataKind : uint8_t {
        METADATA_KIND_NONE,
        METADATA_KIND_LOCAL,
        METADATA_KIND_REMOTE,
    };

    struct RegistryEntry {
        uint32_t proxy_memview_id = 0;
        nixlMemViewH backend_memview = nullptr;
        ProxyMemViewRegEntryState state = ProxyMemViewRegEntryState::ENTRY_ALLOCATED;
        ProxyMemViewRegMetadataKind metadata_kind = ProxyMemViewRegMetadataKind::METADATA_KIND_NONE;
        LocalMetadata local_metadata{};
        RemoteMetadata remote_metadata{};
    };

    RegistryEntry *
    getEntryForHandle(nixlMemViewH proxy_memview);

    const RegistryEntry *
    getEntryForHandle(nixlMemViewH proxy_memview) const;

    RegistryEntry *
    getEntryForId(uint64_t proxy_memview_id);

    const RegistryEntry *
    getEntryForId(uint64_t proxy_memview_id) const;

    nixl_status_t
    getRemoteEntryForSubmission(uint64_t proxy_memview_id,
                                size_t index,
                                size_t offset,
                                size_t size,
                                const RemoteMetadata *&metadata,
                                const ProxyMemViewRegStoredEntry *&entry) const;

    nixl_status_t
    getLocalEntryForSubmission(uint64_t proxy_memview_id,
                               size_t index,
                               size_t offset,
                               size_t size,
                               const LocalMetadata *&metadata,
                               const ProxyMemViewRegStoredEntry *&entry) const;

    static bool
    rangeFits(const ProxyMemViewRegStoredEntry &entry, size_t offset, size_t size);

    static void
    fillLocalMetadata(const nixl_meta_dlist_t &dlist, LocalMetadata &out);

    static void
    fillRemoteMetadata(const nixl_remote_meta_dlist_t &dlist, RemoteMetadata &out);

    std::vector<std::unique_ptr<RegistryEntry>> entries_;
    uint64_t next_proxy_memview_id_ = 1;
};

class nixlProxyRuntime {
public:
    nixlProxyRuntime();
    ~nixlProxyRuntime();

    nixlProxyRuntime(nixlProxyRuntime &&) = delete;
    nixlProxyRuntime(const nixlProxyRuntime &) = delete;
    nixlProxyRuntime &
    operator=(nixlProxyRuntime &&) = delete;
    nixlProxyRuntime &
    operator=(const nixlProxyRuntime &) = delete;

    nixl_status_t
    init(std::unique_ptr<nixlDeviceProxyBackendAdapter> backend,
         uint32_t peer_capacity,
         uint32_t channel_count,
         uint32_t worker_count,
         uint64_t pthr_delay_us = 0);

    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_name, const nixl_blob_t &conn_info);

    nixl_status_t
    registerProxyMemView(nixlMemViewH backend_memview, nixlMemViewH *proxy_memview);

    nixl_status_t
    prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    nixl_status_t
    prepMemView(nixlMemViewH backend_memview,
                const nixl_meta_dlist_t &dlist,
                nixlMemViewH *proxy_memview);

    nixl_status_t
    prepMemView(nixlMemViewH backend_memview,
                const nixl_remote_meta_dlist_t &dlist,
                nixlMemViewH *proxy_memview);

    nixl_status_t
    unregisterProxyMemView(nixlMemViewH proxy_memview);

    nixl_status_t
    storeMetadata(nixlMemViewH proxy_memview, const nixl_meta_dlist_t &dlist);

    nixl_status_t
    storeMetadata(nixlMemViewH proxy_memview, const nixl_remote_meta_dlist_t &dlist);

    bool
    resolveProxyMemView(nixlMemViewH proxy_memview, nixlMemViewH &backend_memview) const;

    bool
    resolveProxyMemViewId(uint64_t proxy_memview_id, nixlMemViewH &backend_memview) const;

    nixl_status_t
    startWorkers();

    nixl_status_t
    shutdown();

    const nixlProxyMemViewRegistry &
    memviewRegistry() const {
        return memview_registry_;
    }

    uint32_t
    channelCount() const {
        return channel_count_;
    }

    uint32_t
    workerCount() const {
        return worker_count_;
    }

    uint32_t
    peerCapacity() const {
        return peer_capacity_;
    }

    size_t
    channelSlotCount() const {
        return channels_.size();
    }

    const nixlProxyChannelView *
    deviceChannelViews() const {
        return device_channel_views_.empty() ? nullptr : device_channel_views_.data();
    }

    nixlProxyDeviceContextData *
    deviceContext() const {
        return device_context_;
    }

    /** Test/diagnostic accessor for per-channel lifecycle state. */
    nixl_proxy_channel_lifecycle_t
    channelLifecycle(uint32_t peer_index, uint32_t channel_id) const;

    /** Flat index into the channel-major [num_channels][peer_capacity] ring matrix. */
    size_t
    channelSlot(uint32_t peer_index, uint32_t channel_id) const {
        return static_cast<size_t>(channel_id) * peer_capacity_ + peer_index;
    }

private:
    void
    joinWorkerThreads() noexcept;

    nixl_status_t
    allocatePeerRow(uint32_t peer_index);

    nixl_status_t
    publishPeerRow(uint32_t peer_index, bool active);

    nixl_status_t
    deactivatePeer(uint32_t peer_index);

    nixl_status_t
    activatePeer(uint32_t peer_index, const std::string &remote_agent);

    nixl_status_t
    waitPeerChannelsInactive(uint32_t peer_index);

    nixl_status_t
    reconcilePeer(uint32_t peer_index, const std::string &remote_agent);

    nixl_status_t
    reconcileRemotePeers(const nixl_remote_meta_dlist_t &dlist);

    std::vector<nixlProxyChannelState> channels_;
    nixlGdrBuffer control_slots_;
    std::vector<nixlProxyChannelView> device_channel_views_;
    nixlProxyChannelView *device_channel_views_dev_ = nullptr;
    nixlProxyDeviceContextData *device_context_ = nullptr;
    std::unique_ptr<std::atomic<nixl_proxy_channel_lifecycle_t>[]> channel_lifecycle_;
    std::vector<std::string> active_agents_;
    std::vector<std::unique_ptr<ProxyWorker>> workers_;
    nixlProxyMemViewRegistry memview_registry_;
    std::unique_ptr<nixlDeviceProxyBackendAdapter> backend_;
    /** CPU-only shutdown state polled by proxy workers; never aliases GPU memory. */
    alignas(64) std::atomic<uint64_t> shutdown_state_{
        static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
    uint32_t peer_capacity_ = 0;
    uint32_t channel_count_ = 0;
    uint32_t worker_count_ = 0;
    uint32_t ring_depth_ = kDefaultProxyRingDepth;
    bool workers_started_ = false;
    std::mutex peer_reconcile_mutex_;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_RUNTIME_H
