/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_PROXY_MEMVIEW_STORE_H
#define NIXL_SRC_CORE_DEVICE_PROXY_PROXY_MEMVIEW_STORE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "backend_adapter.h"
#include "proxy_protocol.h"

struct nixlPreparedProxyMemView {
    nixlMemViewH handle = nullptr;
    nixlProxyMemViewId id = 0;
};

struct ProxyMemViewMetadataElem {
    uintptr_t base_addr;
    size_t length;
    uint64_t device_id;
    nixlBackendMD *backend_metadata;
};

class ProxyDeviceMemViewAllocation {
public:
    ProxyDeviceMemViewAllocation() = default;
    ProxyDeviceMemViewAllocation(nixlMemViewH handle, int cuda_device) noexcept;
    ~ProxyDeviceMemViewAllocation();

    ProxyDeviceMemViewAllocation(ProxyDeviceMemViewAllocation &&other) noexcept;
    ProxyDeviceMemViewAllocation &
    operator=(ProxyDeviceMemViewAllocation &&other) noexcept;

    ProxyDeviceMemViewAllocation(const ProxyDeviceMemViewAllocation &) = delete;
    ProxyDeviceMemViewAllocation &
    operator=(const ProxyDeviceMemViewAllocation &) = delete;

    nixlMemViewH
    get() const noexcept {
        return handle_;
    }

    void
    reset() noexcept;

private:
    nixlMemViewH handle_ = nullptr;
    int cuda_device_ = -1;
};

enum class ProxyMemViewEntryState : uint8_t {
    ACTIVE,
    RETIRED,
};

struct ProxyMemViewEntry {
    ProxyMemViewEntry(nixlProxyMemViewId id,
                      nixlProxyMemViewKind kind,
                      nixl_mem_t mem_type,
                      std::vector<ProxyMemViewMetadataElem> elements,
                      ProxyDeviceMemViewAllocation device_allocation);

    const nixlProxyMemViewId id;
    const nixlProxyMemViewKind kind;
    const nixl_mem_t mem_type;
    const std::vector<ProxyMemViewMetadataElem> elements;
    const nixlMemViewH device_handle;

    std::atomic<ProxyMemViewEntryState> state{ProxyMemViewEntryState::ACTIVE};

    // Cold-path ownership; workers never access it.
    ProxyDeviceMemViewAllocation device_allocation;
};

class nixlProxyMemViewStore {
public:
    nixlProxyMemViewStore(int cuda_device, const nixlProxyDeviceContextData &context);
    ~nixlProxyMemViewStore();

    nixlProxyMemViewStore(const nixlProxyMemViewStore &) = delete;
    nixlProxyMemViewStore &
    operator=(const nixlProxyMemViewStore &) = delete;

    nixl_status_t
    createLocal(const nixl_meta_dlist_t &dlist, nixlPreparedProxyMemView &prepared);

    nixl_status_t
    createRemote(const nixl_remote_meta_dlist_t &dlist,
                 const std::vector<void *> &direct_ptrs,
                 nixlPreparedProxyMemView &prepared);

    nixl_status_t
    release(nixlProxyMemViewId id, nixlMemViewH expected_handle);

    nixl_status_t
    prepareSubmission(const nixlProxySubmission &submission,
                      nixlBackendProxySubmission &prepared_submission) const;

    void
    clearAfterWorkersStop() noexcept;

private:
    static constexpr uint32_t kInitialSlotCapacity = 64;

    struct SlotArray {
        explicit SlotArray(uint32_t capacity);

        const uint32_t capacity;
        std::unique_ptr<std::atomic<ProxyMemViewEntry *>[]> slots;
    };

    nixl_status_t
    create(nixlProxyMemViewKind kind,
           nixl_mem_t mem_type,
           std::vector<ProxyMemViewMetadataElem> metadata,
           std::vector<void *> direct_ptrs,
           nixlPreparedProxyMemView &prepared);

    ProxyMemViewEntry *
    resolveEntry(nixlProxyMemViewId id) const noexcept;

    static bool
    rangeFits(const ProxyMemViewMetadataElem &element, size_t offset, size_t size) noexcept;

    static std::vector<ProxyMemViewMetadataElem>
    copyMetadata(const nixl_meta_dlist_t &dlist);

    static std::vector<ProxyMemViewMetadataElem>
    copyMetadata(const nixl_remote_meta_dlist_t &dlist);

    int cuda_device_;
    const nixlProxyDeviceContextData context_;
    std::atomic<const SlotArray *> published_slots_{nullptr};
    mutable std::mutex writer_mutex_;

    // Cold-path ownership only.
    std::vector<std::unique_ptr<SlotArray>> retained_slot_arrays_;
    std::vector<std::unique_ptr<ProxyMemViewEntry>> retained_entries_;
    uint64_t next_id_ = 1;
};

#endif
