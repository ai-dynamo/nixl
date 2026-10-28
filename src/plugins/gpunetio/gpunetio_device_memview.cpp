// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "gpunetio_backend.h"
#include "gpunetio_cuda_device_guard.h"
#include <gpu/impl/gpunetio/device_gpunetio_types.h>
#include <limits>
#include <stdexcept>

using namespace nixl::gpu::impl::gpunetio;

namespace {
std::atomic<uint64_t> nextCookie{1};

void
checkAllocation(nixl_status_t status) {
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("GPUNETIO native state allocation/upload failed");
    }
}

bool
validRange(uint64_t base, uint64_t length) {
    return length != 0 && length <= std::numeric_limits<uint64_t>::max() - base;
}
} // namespace

void
nixlDocaEngine::initializeNativeState() {
    auto *ops = nixl::getDeviceOps();
    if (ops == nullptr) {
        throw std::runtime_error("GPUNETIO native Device API needs device operations");
    }
    auto &allocator = *ops;
    auto state = std::make_unique<nixlGpunetioNativeState>();
    state->ops = ops;
    state->cookie = nextCookie.fetch_add(1, std::memory_order_relaxed);
    if (state->cookie == 0) {
        throw std::runtime_error("GPUNETIO native context cookie exhausted");
    }
    checkAllocation(allocator.allocDeviceMem(sizeof(GpunetioDeviceLane), state->lane));
    checkAllocation(allocator.allocDeviceMem(sizeof(GpunetioDeviceContext), state->context));
    GpunetioDeviceLane lane{};
    GpunetioDeviceContext context{static_cast<GpunetioDeviceLane *>(state->lane.get())};
    checkAllocation(allocator.copy(
        state->lane.get(), &lane, sizeof(lane), nixl::deviceOps::copyDirection::HostToDevice));
    checkAllocation(allocator.copy(state->context.get(),
                                   &context,
                                   sizeof(context),
                                   nixl::deviceOps::copyDirection::HostToDevice));
    checkAllocation(allocator.synchronize());
    nativeState_ = std::move(state);
}

nixl_status_t
nixlDocaEngine::prepMemView(const nixl_meta_dlist_t &list,
                            nixlMemViewH &out,
                            const nixl_opt_b_args_t *) const {
    return prepareNativeView(&list, nullptr, out);
}

nixl_status_t
nixlDocaEngine::prepMemView(const nixl_remote_meta_dlist_t &list,
                            nixlMemViewH &out,
                            const nixl_opt_b_args_t *) const {
    return prepareNativeView(nullptr, &list, out);
}

nixl_status_t
nixlDocaEngine::prepareNativeView(const nixl_meta_dlist_t *local,
                                  const nixl_remote_meta_dlist_t *remote,
                                  nixlMemViewH &out) const {
    out = nullptr;
    if (!nativeMode_ || !nativeState_) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const size_t count = local ? local->descCount() : remote->descCount();
    const auto type = local ? local->getType() : remote->getType();
    if (type != VRAM_SEG || count == 0 || count > SIZE_MAX / sizeof(GpunetioViewElem)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto &allocator = *nativeState_->ops;
    int device = -1;
    if (allocator.getActiveDevice(device) != NIXL_SUCCESS || device != int(gdevs[0].first)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    std::lock_guard lock(nativeState_->mutex);
    try {
        nixlGpunetioNativeViewStorage storage;
        storage.remote = remote != nullptr;
        std::vector<GpunetioViewElem> elements(count);
        std::string peer = nativeState_->peer;
        doca_gpu_dev_verbs_qp *qp = nullptr;
        for (size_t i = 0; i < count; ++i) {
            if (remote && (*remote)[i].remoteAgent == nixl_null_agent &&
                (*remote)[i].metadataP == nullptr) {
                continue;
            }
            const nixlMetaDesc &desc = local ? (*local)[i] : (*remote)[i];
            if (!desc.metadataP || !validRange(desc.addr, desc.len)) {
                return NIXL_ERR_INVALID_PARAM;
            }
            auto &elem = elements[i];
            elem.base = desc.addr;
            elem.length = desc.len;
            elem.valid = 1;
            elem.peer_slot = local ? UINT32_MAX : 0;
            if (local) {
                const auto *md = static_cast<const nixlDocaPrivateMetadata *>(desc.metadataP);
                if (desc.devId != gdevs[0].first || md->devId != gdevs[0].first || !md->mr) {
                    return NIXL_ERR_INVALID_PARAM;
                }
                const auto base = reinterpret_cast<uintptr_t>(md->mr->get_addr());
                if (desc.addr < base || desc.addr - base > md->mr->get_tot_size() ||
                    desc.len > md->mr->get_tot_size() - (desc.addr - base)) {
                    return NIXL_ERR_INVALID_PARAM;
                }
                elem.key = md->mr->get_lkey();
                storage.pins.push_back(md->mr);
            } else {
                const auto *md = static_cast<const nixlDocaPublicMetadata *>(desc.metadataP);
                const auto &agent = md->conn.remoteAgent;
                if (!peer.empty() && peer != agent) {
                    return NIXL_ERR_NOT_SUPPORTED;
                }
                peer = agent;
                if (!md->mr) {
                    return NIXL_ERR_INVALID_PARAM;
                }
                const auto base = reinterpret_cast<uintptr_t>(md->mr->get_addr());
                if (desc.addr < base || desc.addr - base > md->mr->get_tot_size() ||
                    desc.len > md->mr->get_tot_size() - (desc.addr - base)) {
                    return NIXL_ERR_INVALID_PARAM;
                }
                elem.key = md->mr->get_rkey();
                std::lock_guard qp_lock(qpLock);
                const auto it = qpMap.find(peer);
                if (it == qpMap.end()) {
                    return NIXL_ERR_NOT_FOUND;
                }
                qp = it->second->qp_data->get_qp_gpu_dev();
            }
        }
        if (remote && !qp) {
            return NIXL_ERR_INVALID_PARAM;
        }
        if (qp) {
            doca_gpu_dev_verbs_nic_handler handler;
            if (cudaMemcpy(&handler, &qp->nic_handler, sizeof(handler), cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
                handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
                return NIXL_ERR_NOT_SUPPORTED;
            }
        }
        auto rc = allocator.allocDeviceMem(count * sizeof(GpunetioViewElem), storage.elements);
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
        rc = allocator.allocDeviceMem(sizeof(GpunetioDeviceView), storage.header);
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
        GpunetioDeviceView header{};
        header.abi_version = 1;
        header.role = local ? 1 : 2;
        header.count = count;
        header.context_cookie = nativeState_->cookie;
        header.execution_gpu = gdevs[0].first;
        header.elems = static_cast<GpunetioViewElem *>(storage.elements.get());
        header.context = static_cast<GpunetioDeviceContext *>(nativeState_->context.get());
        rc = allocator.copy(storage.elements.get(),
                            elements.data(),
                            count * sizeof(GpunetioViewElem),
                            nixl::deviceOps::copyDirection::HostToDevice);
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
        rc = allocator.copy(storage.header.get(),
                            &header,
                            sizeof(header),
                            nixl::deviceOps::copyDirection::HostToDevice);
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
        // Bind the shared lane only once; later views must not reset its generation/credits.
        if (remote && nativeState_->peer.empty()) {
            GpunetioDeviceLane lane{};
            lane.qp = qp;
            rc = allocator.copy(nativeState_->lane.get(),
                                &lane,
                                sizeof(lane),
                                nixl::deviceOps::copyDirection::HostToDevice);
            if (rc != NIXL_SUCCESS) {
                return rc;
            }
        }
        nixlMemViewH handle = nullptr;
        rc = nixl::deviceMemViewAllocate(
            allocator, nixl_device_exec_mode_t::GPUNETIO_DIRECT, storage.header.get(), handle);
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
        storage.wrapper = nixl::deviceMem(handle, nixl::deviceMemDeleter{&allocator});
        rc = allocator.synchronize();
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
        const auto [it, inserted] = nativeState_->views.emplace(handle, std::move(storage));
        if (!inserted) {
            return NIXL_ERR_BACKEND;
        }
        nativeState_->peer = std::move(peer);
        if (remote) {
            ++nativeState_->remoteViews;
        }
        out = handle;
        return NIXL_SUCCESS;
    }
    catch (const std::exception &) {
        return NIXL_ERR_BACKEND;
    }
}

void
nixlDocaEngine::releaseMemView(nixlMemViewH handle) const {
    if (!nativeState_) {
        return;
    }
    // Caller must finish its CUDA kernels and retire accepted NIC work before release.
    nixl::doca::cudaDeviceGuard device(gdevs[0].first);
    if (device.status() != cudaSuccess) {
        return;
    }
    std::lock_guard lock(nativeState_->mutex);
    auto it = nativeState_->views.find(handle);
    if (it == nativeState_->views.end()) {
        return;
    }
    auto &allocator = *nativeState_->ops;
    GpunetioDeviceLane snapshot{};
    if (allocator.copy(&snapshot,
                       nativeState_->lane.get(),
                       sizeof(snapshot),
                       nixl::deviceOps::copyDirection::DeviceToHost) != NIXL_SUCCESS ||
        snapshot.phase != gpunetio_lane_phase_idle) {
        // No cancellation is implied by release: keep the view/MR pins until
        // engine teardown has successfully quiesced its QPs.
        return;
    }
    const bool was_remote = it->second.remote;
    nativeState_->views.erase(it);
    if (was_remote && --nativeState_->remoteViews == 0) {
        // Common-wrapper publication can fail after backend prep succeeded.
        // Undo an unused binding, never reset an operated/live lane or its credits.
        GpunetioDeviceLane lane{};
        if (allocator.copy(&lane,
                           nativeState_->lane.get(),
                           sizeof(lane),
                           nixl::deviceOps::copyDirection::DeviceToHost) == NIXL_SUCCESS &&
            lane.phase == gpunetio_lane_phase_idle && lane.generation == 0) {
            lane.qp = nullptr;
            if (allocator.copy(nativeState_->lane.get(),
                               &lane,
                               sizeof(lane),
                               nixl::deviceOps::copyDirection::HostToDevice) == NIXL_SUCCESS &&
                allocator.synchronize() == NIXL_SUCCESS) {
                nativeState_->peer.clear();
            }
        }
    }
}
