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
#include <string>
#include <unordered_map>
#include <vector>

#include "backend_aux.h"
#include "proxy_protocol.h"
#include "backend_adapter.h"
#include "proxy_control_buffer.h"

class nixlProxyWorker;

static constexpr uint32_t default_proxy_ring_depth = 256;
static constexpr size_t proxy_shutdown_slot = 0;
static constexpr size_t proxy_ci_slot_base = 1;

/** @brief Host-side state for one in-flight proxy request. */
struct nixlProxyRequestState {
    uint64_t opIdx = 0;
    nixlBackendProxyRequest backendRequest{};
    nixl_status_t status = NIXL_IN_PROG;
};

/** @brief Owns host and device state for one proxy work ring. */
struct alignas(64) nixlProxyChannelState {
    nixlProxyChannelState() = default;
    ~nixlProxyChannelState();
    nixlProxyChannelState(nixlProxyChannelState &&) noexcept;
    nixlProxyChannelState &
    operator=(nixlProxyChannelState &&) noexcept;
    nixlProxyChannelState(const nixlProxyChannelState &) = delete;
    nixlProxyChannelState &
    operator=(const nixlProxyChannelState &) = delete;

    /** @brief Create device and host resources for one proxy channel. */
    nixl_status_t
    allocate(uint32_t depth, nixlProxyControlBuffer *control_slots, size_t control_slot_index);

    /** @brief Publish the channel's completed-operation frontier. */
    nixl_status_t
    publishConsumerIdx(uint64_t value) noexcept;

    bool
    allocated() const {
        return workRingDev != nullptr;
    }

    /** @brief Release all channel resources. */
    void
    deallocate() noexcept;

    nixlProxyChannelView deviceView{};
    /**
     * Per-ring-slot backend state. A submitted record remains associated with
     * its ring slot until completion advances consumerIdxShadow past it.
     */
    std::vector<nixlProxyRequestState> inflightSlots;
    /** Host-only submit frontier; consumerIdxShadow remains the completion frontier. */
    uint64_t submitIdx = 0;
    /** Host shadow of the authoritative GPU-visible consumer index. */
    uint64_t consumerIdxShadow = 0;

    nixlProxyWorkRing *workRingDev = nullptr;
    nixlProxySubmission *recordsHost = nullptr;
    /** Device-resident producer index; only the GPU updates it. */
    uint64_t *producerIdxDev = nullptr;
    /** Authoritative consumer count; CPU publishes through GDRCopy or mapped host memory. */
    uint64_t *consumerIdxDev = nullptr;
    /** Device-resident cache of consumerIdxDev used by GPU enqueue backpressure. */
    uint64_t *consumerIdxCacheDev = nullptr;
    nixlProxyControlBuffer *controlSlots = nullptr;
    size_t controlSlotIndex = 0;
    /** Host-side ring depth for the CPU worker; nixlProxyWorkRing itself is device-only. */
    uint32_t ringDepth = 0;
    /** Mapped pinned host memory; proxy worker writes directly via host alias. */
    nixlProxyCompletionSlot *completionSlotHost = nullptr;
    /** Device-mapped alias of completionSlotHost for nixlProxyChannelView. */
    nixlProxyCompletionSlot *completionSlotDev = nullptr;
};

/** @brief Maps backend memory views to GPU-visible proxy memory views. */
class nixlProxyMemViewRegistry {
public:
    nixlProxyMemViewRegistry() = default;
    ~nixlProxyMemViewRegistry();

    nixlProxyMemViewRegistry(const nixlProxyMemViewRegistry &) = delete;
    nixlProxyMemViewRegistry &
    operator=(const nixlProxyMemViewRegistry &) = delete;

    /** @brief Set the GPU-visible proxy context used by registered memory views. */
    void
    setDeviceContext(const nixlProxyDeviceContextData *context) {
        device_context_ = context;
    }

    /** @brief Register one backend view and create its proxy view. */
    nixl_status_t
    registerProxyMemView(nixlMemViewH backend_memview, nixlMemViewH *proxy_memview);

    /** @brief Create a proxy view from local metadata. */
    nixl_status_t
    prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    /** @brief Create a proxy view from remote metadata. */
    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    /** @brief Create a remote proxy view with directly accessible pointers. */
    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist,
                const std::vector<void *> &direct_ptrs,
                nixlMemViewH *proxy_memview);

    /** @brief Associate local metadata with an existing backend view. */
    nixl_status_t
    prepMemView(nixlMemViewH backend_memview,
                const nixl_meta_dlist_t &dlist,
                nixlMemViewH *proxy_memview);

    /** @brief Associate remote metadata with an existing backend view. */
    nixl_status_t
    prepMemView(nixlMemViewH backend_memview,
                const nixl_remote_meta_dlist_t &dlist,
                nixlMemViewH *proxy_memview);

    /** @brief Retire and release a proxy memory view. */
    nixl_status_t
    unregisterProxyMemView(nixlMemViewH proxy_memview);

    /** @brief Store local metadata for an allocated proxy memory view. */
    nixl_status_t
    storeMetadata(nixlMemViewH proxy_memview, const nixl_meta_dlist_t &dlist);

    /** @brief Store remote metadata for an allocated proxy memory view. */
    nixl_status_t
    storeMetadata(nixlMemViewH proxy_memview, const nixl_remote_meta_dlist_t &dlist);

    /** @brief Resolve a live proxy memory view to its backend handle. */
    bool
    resolveProxyMemView(nixlMemViewH proxy_memview, nixlMemViewH &backend_memview) const;

    /** @brief Resolve a proxy memory-view identifier to its backend handle. */
    bool
    resolveProxyMemViewId(uint64_t proxy_memview_id, nixlMemViewH &backend_memview) const;

    /** @brief Convert a GPU submission into a backend submission. */
    nixl_status_t
    prepareSubmission(const nixlProxySubmission &submission,
                      nixlBackendProxySubmission &prepared_submission) const;

    /** @brief Release every registered proxy memory view. */
    void
    clear() noexcept;

private:
    struct proxyMemViewRegStoredEntry {
        uintptr_t baseAddr = 0;
        size_t len = 0;
        uint64_t devId = 0;
        nixlBackendMD *metadata = nullptr;
        std::string remoteAgent;
    };

    struct localMetadataInfo {
        nixl_mem_t memType = DRAM_SEG;
        std::vector<proxyMemViewRegStoredEntry> entries;
    };

    struct remoteMetadataInfo {
        std::vector<proxyMemViewRegStoredEntry> entries;
    };

    enum class proxy_memview_reg_entry_state_t : uint8_t {
        ENTRY_ALLOCATED,
        ENTRY_READY,
        ENTRY_RETIRED,
    };

    enum class proxy_memview_reg_metadata_kind_t : uint8_t {
        METADATA_KIND_NONE,
        METADATA_KIND_LOCAL,
        METADATA_KIND_REMOTE,
    };

    struct registryEntry {
        uint32_t proxyMemViewId = 0;
        nixlMemViewH proxyMemView = nullptr;
        nixlMemViewH backendMemView = nullptr;
        proxy_memview_reg_entry_state_t state = proxy_memview_reg_entry_state_t::ENTRY_ALLOCATED;
        proxy_memview_reg_metadata_kind_t metadataKind =
            proxy_memview_reg_metadata_kind_t::METADATA_KIND_NONE;
        localMetadataInfo localMetadata{};
        remoteMetadataInfo remoteMetadata{};
    };

    nixl_status_t
    registerProxyMemView(nixlMemViewH backend_memview,
                         const std::vector<void *> &direct_ptrs,
                         nixlMemViewH *proxy_memview);

    static void
    releaseDeviceMemView(registryEntry &entry) noexcept;

    registryEntry *
    getEntryForHandle(nixlMemViewH proxy_memview);

    const registryEntry *
    getEntryForHandle(nixlMemViewH proxy_memview) const;

    registryEntry *
    getEntryForId(uint64_t proxy_memview_id);

    const registryEntry *
    getEntryForId(uint64_t proxy_memview_id) const;

    nixl_status_t
    getRemoteEntryForSubmission(uint64_t proxy_memview_id,
                                size_t index,
                                size_t offset,
                                size_t size,
                                const proxyMemViewRegStoredEntry *&entry) const;

    nixl_status_t
    getLocalEntryForSubmission(uint64_t proxy_memview_id,
                               size_t index,
                               size_t offset,
                               size_t size,
                               const localMetadataInfo *&metadata,
                               const proxyMemViewRegStoredEntry *&entry) const;

    static bool
    rangeFits(const proxyMemViewRegStoredEntry &entry, size_t offset, size_t size);

    static void
    fillLocalMetadata(const nixl_meta_dlist_t &dlist, localMetadataInfo &out);

    static void
    fillRemoteMetadata(const nixl_remote_meta_dlist_t &dlist, remoteMetadataInfo &out);

    std::vector<registryEntry> entries_;
    std::unordered_map<nixlMemViewH, uint32_t> handle_to_id_;
    uint64_t next_proxy_memview_id_ = 1;
    const nixlProxyDeviceContextData *device_context_ = nullptr;
};

/** @brief Owns proxy channels, workers, and their backend adapter. */
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

    /** @brief Initialize the proxy runtime and its worker topology. */
    nixl_status_t
    init(std::unique_ptr<nixlDeviceProxyBackendAdapter> backend,
         uint32_t max_peers,
         uint32_t channel_count,
         uint32_t worker_count,
         uint64_t pthr_delay_us = 0);

    /** @brief Load the backend connection information for a remote agent. */
    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_name, const nixl_blob_t &conn_info);

    /** @brief Register a backend memory view with the runtime registry. */
    nixl_status_t
    registerProxyMemView(nixlMemViewH backend_memview, nixlMemViewH *proxy_memview);

    /** @brief Prepare a proxy view from local metadata. */
    nixl_status_t
    prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    /** @brief Prepare a proxy view from remote metadata. */
    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist, nixlMemViewH *proxy_memview);

    /** @brief Prepare a local proxy view associated with a backend view. */
    nixl_status_t
    prepMemView(nixlMemViewH backend_memview,
                const nixl_meta_dlist_t &dlist,
                nixlMemViewH *proxy_memview);

    /** @brief Prepare a remote proxy view associated with a backend view. */
    nixl_status_t
    prepMemView(nixlMemViewH backend_memview,
                const nixl_remote_meta_dlist_t &dlist,
                nixlMemViewH *proxy_memview);

    /** @brief Retire one proxy memory view. */
    nixl_status_t
    unregisterProxyMemView(nixlMemViewH proxy_memview);

    /** @brief Store local metadata for an existing proxy memory view. */
    nixl_status_t
    storeMetadata(nixlMemViewH proxy_memview, const nixl_meta_dlist_t &dlist);

    /** @brief Store remote metadata for an existing proxy memory view. */
    nixl_status_t
    storeMetadata(nixlMemViewH proxy_memview, const nixl_remote_meta_dlist_t &dlist);

    /** @brief Resolve a proxy memory view to its backend handle. */
    bool
    resolveProxyMemView(nixlMemViewH proxy_memview, nixlMemViewH &backend_memview) const;

    /** @brief Resolve a proxy memory-view identifier to its backend handle. */
    bool
    resolveProxyMemViewId(uint64_t proxy_memview_id, nixlMemViewH &backend_memview) const;

    /** @brief Start CPU worker threads. */
    nixl_status_t
    startWorkers();

    /** @brief Stop workers and release runtime resources. */
    nixl_status_t
    shutdown();

    /** @brief Return the proxy memory-view registry. */
    const nixlProxyMemViewRegistry &
    memviewRegistry() const {
        return memview_registry_;
    }

    /** @brief Return GPU-visible channel views. */
    const nixlProxyChannelView *
    deviceChannelViews() const {
        return device_channel_views_.empty() ? nullptr : device_channel_views_.data();
    }

    /** @brief Return the GPU-visible proxy context. */
    nixlProxyDeviceContextData *
    deviceContext() const {
        return device_context_;
    }

private:
    void
    joinWorkerThreads() noexcept;

    std::vector<nixlProxyChannelState> channels_;
    nixlProxyControlBuffer control_slots_;
    std::vector<nixlProxyChannelView> device_channel_views_;
    nixlProxyChannelView *device_channel_views_dev_ = nullptr;
    nixlProxyDeviceContextData *device_context_ = nullptr;
    std::vector<std::unique_ptr<nixlProxyWorker>> workers_;
    nixlProxyMemViewRegistry memview_registry_;
    std::unique_ptr<nixlDeviceProxyBackendAdapter> backend_;
    alignas(64) std::atomic<uint64_t> shutdown_state_{
        static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN)};
    uint64_t *shutdown_word_dev_ = nullptr;
    uint32_t ring_depth_ = default_proxy_ring_depth;
    bool workers_started_ = false;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_RUNTIME_H
