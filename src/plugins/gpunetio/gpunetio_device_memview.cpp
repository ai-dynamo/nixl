// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "gpunetio_backend.h"
#include "gpunetio_cuda_device_guard.h"
#include <array>
#include <gpu/impl/gpunetio/device_gpunetio_types.h>
#include <limits>
#include <algorithm>
#include <stdexcept>

using namespace nixl::gpu::impl::gpunetio;

namespace {
std::atomic<uint64_t> nextCookie{1};

bool
copyOnStream(cudaStream_t stream, void *dst, const void *src, size_t size, cudaMemcpyKind kind) {
    const auto result = cudaMemcpyAsync(dst, src, size, kind, stream);
    return result == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
}

bool
allocateOnStream(const std::shared_ptr<nixlGpunetioNativeStream> &stream,
                 size_t size,
                 nixlGpunetioNativeMem &out) {
    void *ptr = nullptr;
    if (cudaMallocAsync(&ptr, size, stream->stream) != cudaSuccess) {
        return false;
    }
    out = nixlGpunetioNativeMem(ptr, nixlGpunetioNativeDeleter{stream});
    return true;
}

bool
validRange(uint64_t base, uint64_t length) {
    return length != 0 && length <= std::numeric_limits<uint64_t>::max() - base;
}
} // namespace

void
nixlDocaEngine::initializeNativeState() {
    nixl::doca::cudaDeviceGuard guard(gdevs[0].first);
    if (guard.status() != cudaSuccess) {
        throw std::runtime_error("Cannot select native CUDA device");
    }
    auto state = std::make_unique<nixlGpunetioNativeState>();
    state->stream = std::make_shared<nixlGpunetioNativeStream>();
    if (cudaStreamCreateWithFlags(&state->stream->stream, cudaStreamNonBlocking) != cudaSuccess) {
        throw std::runtime_error("Cannot create native upload stream");
    }
    state->cookie = nextCookie.fetch_add(1, std::memory_order_relaxed);
    if (state->cookie == 0) {
        throw std::runtime_error("GPUNETIO native context cookie exhausted");
    }
    state->laneCount = dataQpCount_;
    state->boundQps.resize(state->laneCount, nullptr);
    state->atomicResults.resize(state->laneCount);
    state->atomicMr = std::make_shared<nixl::doca::verbs::mr>(
        nullptr, state->atomicResults.data(), state->laneCount, sizeof(uint64_t), pd, false);
    if (!allocateOnStream(
            state->stream, sizeof(GpunetioDeviceLane) * state->laneCount, state->lane) ||
        !allocateOnStream(
            state->stream, sizeof(GpunetioDeviceContext) * state->laneCount, state->context)) {
        throw std::runtime_error("GPUNETIO native allocation failed");
    }
    std::vector<GpunetioDeviceLane> lanes(state->laneCount);
    std::vector<GpunetioDeviceContext> contexts(state->laneCount);
    for (uint32_t i = 0; i < state->laneCount; ++i) {
        contexts[i].lane = static_cast<GpunetioDeviceLane *>(state->lane.get()) + i;
        contexts[i].channels = static_cast<GpunetioDeviceLane *>(state->lane.get());
        contexts[i].channel_count = state->laneCount;
        contexts[i].atomic_results = reinterpret_cast<uintptr_t>(state->atomicResults.data());
        contexts[i].atomic_key = state->atomicMr->get_lkey();
    }
    if (!copyOnStream(state->stream->stream,
                      state->lane.get(),
                      lanes.data(),
                      sizeof(GpunetioDeviceLane) * lanes.size(),
                      cudaMemcpyHostToDevice) ||
        !copyOnStream(state->stream->stream,
                      state->context.get(),
                      contexts.data(),
                      sizeof(GpunetioDeviceContext) * contexts.size(),
                      cudaMemcpyHostToDevice)) {
        throw std::runtime_error("GPUNETIO native upload failed");
    }
    nativeState_ = std::move(state);
}

nixl_status_t
nixlDocaEngine::prepMemView(const nixl_meta_dlist_t &list,
                            nixlMemViewH &out,
                            const nixl_opt_b_args_t *args) const {
    return prepareNativeView(&list, nullptr, out, args);
}

nixl_status_t
nixlDocaEngine::prepMemView(const nixl_remote_meta_dlist_t &list,
                            nixlMemViewH &out,
                            const nixl_opt_b_args_t *args) const {
    return prepareNativeView(nullptr, &list, out, args);
}

nixl_status_t
nixlDocaEngine::prepareNativeView(const nixl_meta_dlist_t *local,
                                  const nixl_remote_meta_dlist_t *remote,
                                  nixlMemViewH &out,
                                  const nixl_opt_b_args_t *args) const {
    out = nullptr;
    if (!nativeMode_ || !nativeState_) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    uint32_t channel = 0;
    if (!nixl::gpunetio::parseNativeChannel(
            args ? args->customParam : std::string_view{}, nativeState_->laneCount, channel)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const size_t count = local ? local->descCount() : remote->descCount();
    const auto type = local ? local->getType() : remote->getType();
    const bool supported_type = type == VRAM_SEG || (remote != nullptr && type == DRAM_SEG);
    if (!supported_type || count == 0 || count > SIZE_MAX / sizeof(GpunetioViewElem)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != int(gdevs[0].first)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    std::lock_guard lock(nativeState_->mutex);
    try {
        nixlGpunetioNativeViewStorage storage;
        storage.remote = remote != nullptr;
        std::vector<GpunetioViewElem> elements(count);
        std::string peer = nativeState_->peer;
        std::vector<doca_gpu_dev_verbs_qp *> qps(nativeState_->laneCount, nullptr);
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
                if (agent.empty() || agent == nixl_null_agent ||
                    (!(*remote)[i].remoteAgent.empty() && (*remote)[i].remoteAgent != agent)) {
                    return NIXL_ERR_INVALID_PARAM;
                }
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
                {
                    std::lock_guard conn_lock(remoteConnLock);
                    auto conn = remoteConnMap.find(peer);
                    if (conn == remoteConnMap.end() || !conn->second.connected) {
                        return NIXL_ERR_NOT_FOUND;
                    }
                }
                std::lock_guard qp_lock(qpLock);
                const auto it = qpMap.find(peer);
                if (it == qpMap.end()) {
                    return NIXL_ERR_NOT_FOUND;
                }
                if (nativeState_->laneCount == 1) {
                    qps[0] = it->second->qp_data->get_qp_gpu_dev();
                } else {
                    if (it->second->qp_write.size() != nativeState_->laneCount) {
                        return NIXL_ERR_BACKEND;
                    }
                    for (uint32_t c = 0; c < nativeState_->laneCount; ++c) {
                        qps[c] = it->second->qp_write[c]->get_qp_gpu_dev();
                    }
                }
            }
        }
        if (remote) {
            for (uint32_t c = 0; c < nativeState_->laneCount; ++c) {
                if (!qps[c]) {
                    return NIXL_ERR_INVALID_PARAM;
                }
                if (nativeState_->boundQps[c] && nativeState_->boundQps[c] != qps[c]) {
                    return NIXL_ERR_NOT_SUPPORTED;
                }
                doca_gpu_dev_verbs_nic_handler handler;
                if (!copyOnStream(nativeState_->stream->stream,
                                  &handler,
                                  &qps[c]->nic_handler,
                                  sizeof(handler),
                                  cudaMemcpyDeviceToHost) ||
                    handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
                    return NIXL_ERR_NOT_SUPPORTED;
                }
            }
        }
        if (!allocateOnStream(
                nativeState_->stream, count * sizeof(GpunetioViewElem), storage.elements) ||
            !allocateOnStream(nativeState_->stream, sizeof(GpunetioDeviceView), storage.header) ||
            !allocateOnStream(
                nativeState_->stream, sizeof(nixlDeviceMemViewWrapper), storage.wrapper)) {
            return NIXL_ERR_BACKEND;
        }
        GpunetioDeviceView header{};
        header.abi_version = gpunetio_device_view_abi_version;
        header.role = local ? 1 : 2;
        header.count = count;
        header.context_cookie = nativeState_->cookie;
        header.execution_gpu = gdevs[0].first;
        header.elems = static_cast<GpunetioViewElem *>(storage.elements.get());
        header.context =
            static_cast<GpunetioDeviceContext *>(nativeState_->context.get()) + channel;
        const auto stream = nativeState_->stream->stream;
        if (!copyOnStream(stream,
                          storage.elements.get(),
                          elements.data(),
                          count * sizeof(GpunetioViewElem),
                          cudaMemcpyHostToDevice) ||
            !copyOnStream(
                stream, storage.header.get(), &header, sizeof(header), cudaMemcpyHostToDevice)) {
            return NIXL_ERR_BACKEND;
        }
        // Bind every channel once; standard calls select their channel at submission.
        if (remote) {
            for (uint32_t c = 0; c < nativeState_->laneCount; ++c) {
                if (nativeState_->boundQps[c]) {
                    continue;
                }
                GpunetioDeviceLane lane{};
                lane.qp = qps[c];
                if (!copyOnStream(stream,
                                  static_cast<GpunetioDeviceLane *>(nativeState_->lane.get()) + c,
                                  &lane,
                                  sizeof(lane),
                                  cudaMemcpyHostToDevice)) {
                    return NIXL_ERR_BACKEND;
                }
            }
        }
        const nixlDeviceMemViewWrapper wrapper{nixl_device_exec_mode_t::GPUNETIO_DIRECT,
                                               storage.header.get()};
        if (!copyOnStream(
                stream, storage.wrapper.get(), &wrapper, sizeof(wrapper), cudaMemcpyHostToDevice)) {
            return NIXL_ERR_BACKEND;
        }
        nixlMemViewH handle = storage.wrapper.get();
        const auto [it, inserted] = nativeState_->views.emplace(handle, std::move(storage));
        if (!inserted) {
            return NIXL_ERR_BACKEND;
        }
        nativeState_->peer = std::move(peer);
        if (remote) {
            nativeState_->boundQps = std::move(qps);
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
    nixl::doca::cudaDeviceGuard guard(gdevs[0].first);
    if (guard.status() != cudaSuccess) {
        return;
    }
    std::lock_guard lock(nativeState_->mutex);
    auto it = nativeState_->views.find(handle);
    if (it == nativeState_->views.end()) {
        return;
    }
    const auto stream = nativeState_->stream->stream;
    // Any standard channel can access this view's registrations.
    if (!nativeState_->lanesIdle()) {
        return;
    }
    const bool was_remote = it->second.remote;
    nativeState_->views.erase(it);
    if (was_remote && --nativeState_->remoteViews == 0) {
        // Common-wrapper publication can fail after backend prep succeeded.
        // Undo an unused binding, never reset an operated/live lane or its credits.
        std::array<GpunetioDeviceLane, nixl::gpunetio::maxNativeQps> lanes{};
        if (!copyOnStream(stream,
                          lanes.data(),
                          nativeState_->lane.get(),
                          sizeof(GpunetioDeviceLane) * nativeState_->laneCount,
                          cudaMemcpyDeviceToHost)) {
            nativeState_->snapshotFailed = true;
            return;
        }
        if (std::all_of(
                lanes.begin(), lanes.begin() + nativeState_->laneCount, [](const auto &lane) {
                    return lane.phase == gpunetio_lane_phase_idle && lane.generation == 0;
                })) {
            for (auto &lane : lanes) {
                lane.qp = nullptr;
            }
            if (copyOnStream(stream,
                             nativeState_->lane.get(),
                             lanes.data(),
                             sizeof(GpunetioDeviceLane) * nativeState_->laneCount,
                             cudaMemcpyHostToDevice)) {
                nativeState_->peer.clear();
                std::fill(nativeState_->boundQps.begin(), nativeState_->boundQps.end(), nullptr);
            } else {
                nativeState_->snapshotFailed = true;
            }
        }
    }
}

bool
nixlGpunetioNativeState::lanesIdle() const {
    std::array<GpunetioDeviceLane, nixl::gpunetio::maxNativeQps> lanes{};
    return !snapshotFailed &&
        copyOnStream(stream->stream,
                     lanes.data(),
                     lane.get(),
                     sizeof(GpunetioDeviceLane) * laneCount,
                     cudaMemcpyDeviceToHost) &&
        std::all_of(lanes.begin(), lanes.begin() + laneCount, [](const auto &entry) {
            return entry.phase == gpunetio_lane_phase_idle;
        });
}
