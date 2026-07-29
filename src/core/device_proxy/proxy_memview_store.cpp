/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "proxy_memview_store.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include <cuda_runtime.h>

#include "nixl_log.h"

namespace {

class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int target) noexcept {
        if (cudaGetDevice(&original_) != cudaSuccess) {
            return;
        }
        if (original_ == target || cudaSetDevice(target) == cudaSuccess) {
            active_ = true;
        }
    }

    ~CudaDeviceGuard() {
        if (active_ && original_ >= 0) {
            cudaSetDevice(original_);
        }
    }

    bool
    active() const noexcept {
        return active_;
    }

private:
    int original_ = -1;
    bool active_ = false;
};

} // namespace

ProxyDeviceMemViewAllocation::ProxyDeviceMemViewAllocation(nixlMemViewH handle,
                                                           int cuda_device) noexcept
    : handle_(handle), cuda_device_(cuda_device) {}

ProxyDeviceMemViewAllocation::~ProxyDeviceMemViewAllocation() {
    reset();
}

ProxyDeviceMemViewAllocation::ProxyDeviceMemViewAllocation(
    ProxyDeviceMemViewAllocation &&other) noexcept {
    *this = std::move(other);
}

ProxyDeviceMemViewAllocation &
ProxyDeviceMemViewAllocation::operator=(ProxyDeviceMemViewAllocation &&other) noexcept {
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        cuda_device_ = std::exchange(other.cuda_device_, -1);
    }
    return *this;
}

void
ProxyDeviceMemViewAllocation::reset() noexcept {
    if (handle_ == nullptr) {
        return;
    }
    CudaDeviceGuard guard(cuda_device_);
    if (!guard.active() || cudaFree(handle_) != cudaSuccess) {
        NIXL_ERROR << "Failed to free proxy device memview on CUDA device " << cuda_device_;
    }
    handle_ = nullptr;
    cuda_device_ = -1;
}

ProxyMemViewEntry::ProxyMemViewEntry(nixlProxyMemViewId id,
                                     nixlProxyMemViewKind kind,
                                     nixl_mem_t mem_type,
                                     std::vector<ProxyMemViewMetadataElem> elements,
                                     ProxyDeviceMemViewAllocation device_allocation)
    : id(id),
      kind(kind),
      mem_type(mem_type),
      elements(std::move(elements)),
      device_handle(device_allocation.get()),
      device_allocation(std::move(device_allocation)) {}

nixlProxyMemViewStore::SlotArray::SlotArray(uint32_t capacity)
    : capacity(capacity), slots(std::make_unique<std::atomic<ProxyMemViewEntry *>[]>(capacity)) {
    for (uint32_t index = 0; index < capacity; ++index) {
        slots[index].store(nullptr, std::memory_order_relaxed);
    }
}

nixlProxyMemViewStore::nixlProxyMemViewStore(int cuda_device,
                                             const nixlProxyDeviceContextData &context)
    : cuda_device_(cuda_device), context_(context) {
    auto initial = std::make_unique<SlotArray>(kInitialSlotCapacity);
    const SlotArray *published = initial.get();
    retained_slot_arrays_.push_back(std::move(initial));
    published_slots_.store(published, std::memory_order_release);
}

nixlProxyMemViewStore::~nixlProxyMemViewStore() {
    clearAfterWorkersStop();
}

std::vector<ProxyMemViewMetadataElem>
nixlProxyMemViewStore::copyMetadata(const nixl_meta_dlist_t &dlist) {
    std::vector<ProxyMemViewMetadataElem> result;
    result.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        result.push_back(
            {desc.addr, desc.len, desc.devId, const_cast<nixlBackendMD *>(desc.metadataP)});
    }
    return result;
}

std::vector<ProxyMemViewMetadataElem>
nixlProxyMemViewStore::copyMetadata(const nixl_remote_meta_dlist_t &dlist) {
    std::vector<ProxyMemViewMetadataElem> result;
    result.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        result.push_back(
            {desc.addr, desc.len, desc.devId, const_cast<nixlBackendMD *>(desc.metadataP)});
    }
    return result;
}

nixl_status_t
nixlProxyMemViewStore::createLocal(const nixl_meta_dlist_t &dlist,
                                   nixlPreparedProxyMemView &prepared) {
    prepared = {};
    try {
        return create(nixlProxyMemViewKind::LOCAL,
                      dlist.getType(),
                      copyMetadata(dlist),
                      std::vector<void *>(dlist.descCount(), nullptr),
                      prepared);
    }
    catch (const std::bad_alloc &) {
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlProxyMemViewStore::createRemote(const nixl_remote_meta_dlist_t &dlist,
                                    const std::vector<void *> &direct_ptrs,
                                    nixlPreparedProxyMemView &prepared) {
    prepared = {};
    try {
        std::vector<void *> elements = direct_ptrs;
        if (elements.empty()) {
            elements.assign(dlist.descCount(), nullptr);
        }
        if (elements.size() != static_cast<size_t>(dlist.descCount())) {
            return NIXL_ERR_INVALID_PARAM;
        }
        return create(nixlProxyMemViewKind::REMOTE,
                      dlist.getType(),
                      copyMetadata(dlist),
                      std::move(elements),
                      prepared);
    }
    catch (const std::bad_alloc &) {
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlProxyMemViewStore::create(nixlProxyMemViewKind kind,
                              nixl_mem_t mem_type,
                              std::vector<ProxyMemViewMetadataElem> metadata,
                              std::vector<void *> direct_ptrs,
                              nixlPreparedProxyMemView &prepared) {
    prepared = {};
    if (metadata.size() != direct_ptrs.size() ||
        metadata.size() > std::numeric_limits<uint32_t>::max()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(writer_mutex_);
    if (next_id_ > std::numeric_limits<nixlProxyMemViewId>::max()) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    const auto id = static_cast<nixlProxyMemViewId>(next_id_++);

    constexpr size_t header_size = offsetof(nixlProxyDeviceMemView, mem_elements);
    if (direct_ptrs.size() >
        (std::numeric_limits<size_t>::max() - header_size) /
            sizeof(nixlProxyDeviceMemViewElem)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const size_t allocation_size =
        header_size + direct_ptrs.size() * sizeof(nixlProxyDeviceMemViewElem);

    std::vector<std::byte> host_storage(allocation_size);
    std::memset(host_storage.data(), 0, host_storage.size());
    auto *host_memview = reinterpret_cast<nixlProxyDeviceMemView *>(host_storage.data());
    host_memview->version = NIXL_PROXY_MEM_LIST_VERSION_V1;
    host_memview->length = static_cast<uint32_t>(metadata.size());
    host_memview->context = context_;
    host_memview->proxy_memview_id = id;
    host_memview->kind = kind;
    for (size_t index = 0; index < direct_ptrs.size(); ++index) {
        host_memview->mem_elements[index].direct_ptr = direct_ptrs[index];
    }

    CudaDeviceGuard guard(cuda_device_);
    if (!guard.active()) {
        return NIXL_ERR_BACKEND;
    }
    nixlProxyDeviceMemView *device_memview = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device_memview), allocation_size) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    ProxyDeviceMemViewAllocation allocation(device_memview, cuda_device_);
    if (cudaMemcpy(device_memview,
                   host_storage.data(),
                   allocation_size,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    auto entry = std::make_unique<ProxyMemViewEntry>(
        id, kind, mem_type, std::move(metadata), std::move(allocation));
    ProxyMemViewEntry *entry_ptr = entry.get();

    const SlotArray *slots = published_slots_.load(std::memory_order_relaxed);
    if (id > slots->capacity) {
        uint64_t new_capacity = slots->capacity;
        while (id > new_capacity) {
            new_capacity *= 2;
            if (new_capacity > std::numeric_limits<uint32_t>::max()) {
                new_capacity = std::numeric_limits<uint32_t>::max();
            }
        }
        auto grown = std::make_unique<SlotArray>(static_cast<uint32_t>(new_capacity));
        for (uint32_t index = 0; index < slots->capacity; ++index) {
            grown->slots[index].store(slots->slots[index].load(std::memory_order_relaxed),
                                      std::memory_order_relaxed);
        }
        slots = grown.get();
        retained_slot_arrays_.push_back(std::move(grown));
        published_slots_.store(slots, std::memory_order_release);
    }

    retained_entries_.push_back(std::move(entry));
    slots->slots[id - 1].store(entry_ptr, std::memory_order_release);
    prepared = {entry_ptr->device_handle, id};
    return NIXL_SUCCESS;
}

ProxyMemViewEntry *
nixlProxyMemViewStore::resolveEntry(nixlProxyMemViewId id) const noexcept {
    if (id == 0) {
        return nullptr;
    }
    const SlotArray *slots = published_slots_.load(std::memory_order_acquire);
    if (slots == nullptr) {
        return nullptr;
    }
    const uint32_t index = id - 1;
    if (index >= slots->capacity) {
        // A reader may have observed the old array while a writer published a
        // valid ID from a newly grown array. One defensive reload is enough.
        slots = published_slots_.load(std::memory_order_acquire);
        if (slots == nullptr || index >= slots->capacity) {
            return nullptr;
        }
    }
    return slots->slots[index].load(std::memory_order_acquire);
}

bool
nixlProxyMemViewStore::rangeFits(const ProxyMemViewMetadataElem &element,
                                 size_t offset,
                                 size_t size) noexcept {
    return offset <= element.length && size <= element.length - offset &&
           offset <= std::numeric_limits<uintptr_t>::max() - element.base_addr;
}

nixl_status_t
nixlProxyMemViewStore::prepareSubmission(
    const nixlProxySubmission &submission,
    nixlBackendProxySubmission &prepared_submission) const {
    size_t transfer_size;
    bool needs_source;
    switch (submission.opcode) {
    case nixl_proxy_opcode_t::PUT:
        transfer_size = submission.size;
        needs_source = true;
        break;
    case nixl_proxy_opcode_t::ATOMIC_ADD:
        transfer_size = sizeof(uint64_t);
        needs_source = false;
        break;
    default:
        return NIXL_ERR_NOT_SUPPORTED;
    }

    ProxyMemViewEntry *dst = resolveEntry(submission.dst_proxy_memview_id);
    if (dst == nullptr ||
        dst->state.load(std::memory_order_acquire) != ProxyMemViewEntryState::ACTIVE) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (dst->kind != nixlProxyMemViewKind::REMOTE ||
        submission.dst_index >= dst->elements.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto &dst_element = dst->elements[submission.dst_index];
    if (!rangeFits(dst_element, submission.dst_offset, transfer_size)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlBackendProxySubmission result{};
    result.op_idx = submission.op_idx;
    result.opcode = submission.opcode;
    result.channel_id = submission.channel_id;
    result.flags = submission.flags;
    result.size = transfer_size;
    result.value = submission.value;
    result.remote.mem_type = dst->mem_type;
    result.remote.desc = nixlMetaDesc(dst_element.base_addr + submission.dst_offset,
                                     transfer_size,
                                     dst_element.device_id,
                                     dst_element.backend_metadata);

    if (needs_source) {
        ProxyMemViewEntry *src = resolveEntry(submission.src_proxy_memview_id);
        if (src == nullptr ||
            src->state.load(std::memory_order_acquire) != ProxyMemViewEntryState::ACTIVE) {
            return NIXL_ERR_NOT_FOUND;
        }
        if (src->kind != nixlProxyMemViewKind::LOCAL ||
            submission.src_index >= src->elements.size()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const auto &src_element = src->elements[submission.src_index];
        if (!rangeFits(src_element, submission.src_offset, transfer_size)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        result.local.mem_type = src->mem_type;
        result.local.desc = nixlMetaDesc(src_element.base_addr + submission.src_offset,
                                        transfer_size,
                                        src_element.device_id,
                                        src_element.backend_metadata);
    }

    prepared_submission = result;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewStore::release(nixlProxyMemViewId id, nixlMemViewH expected_handle) {
    ProxyDeviceMemViewAllocation allocation;
    {
        std::lock_guard<std::mutex> lock(writer_mutex_);
        ProxyMemViewEntry *entry = resolveEntry(id);
        if (entry == nullptr || entry->device_handle != expected_handle) {
            return NIXL_ERR_INVALID_PARAM;
        }
        ProxyMemViewEntryState expected = ProxyMemViewEntryState::ACTIVE;
        if (!entry->state.compare_exchange_strong(expected,
                                                  ProxyMemViewEntryState::RETIRED,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        allocation = std::move(entry->device_allocation);
    }
    return NIXL_SUCCESS;
}

void
nixlProxyMemViewStore::clearAfterWorkersStop() noexcept {
    std::vector<ProxyDeviceMemViewAllocation> allocations;
    {
        std::lock_guard<std::mutex> lock(writer_mutex_);
        published_slots_.store(nullptr, std::memory_order_release);
        allocations.reserve(retained_entries_.size());
        for (auto &entry : retained_entries_) {
            entry->state.store(ProxyMemViewEntryState::RETIRED, std::memory_order_release);
            allocations.push_back(std::move(entry->device_allocation));
        }
        retained_entries_.clear();
        retained_slot_arrays_.clear();
    }
}
