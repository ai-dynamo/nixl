/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_CUH

#include <gpu/device_types.cuh>
#include <gpu/impl/level_utils.cuh>
#include <gpu/impl/gpunetio/device_gpunetio_compat.cuh>
#include <cstdint>
#include <cuda/atomic>

namespace nixl::gpu::impl::gpunetio {

static_assert(sizeof(GpunetioStatusPayload) <= xfer_status_payload_size);

template<typename T> using device_atomic_ref = cuda::atomic_ref<T, cuda::thread_scope_device>;

template<typename T>
__device__ inline device_atomic_ref<T>
deviceAtomic(T &value) {
    return device_atomic_ref<T>(value);
}

__device__ inline GpunetioStatusPayload *
statusPayload(xferStatusH *status) {
    return reinterpret_cast<GpunetioStatusPayload *>(status->storage);
}

__device__ inline bool
rangeIsValid(uint64_t base, uint64_t length, size_t offset, size_t bytes, uint64_t *address) {
    if (offset > length || bytes > length - offset) {
        return false;
    }
    const uint64_t offset64 = static_cast<uint64_t>(offset);
    const uint64_t bytes64 = static_cast<uint64_t>(bytes);
    if (static_cast<size_t>(offset64) != offset || static_cast<size_t>(bytes64) != bytes) {
        return false;
    }
    if (base > ~uint64_t{0} - offset64) {
        return false;
    }
    *address = base + offset64;
    return *address <= ~uint64_t{0} - bytes64;
}

__device__ inline bool
validView(const GpunetioDeviceView *view, uint32_t role) {
    return view && view->abi_version == gpunetio_device_view_abi_version && view->role == role &&
        view->context_cookie && view->context && view->context->channels && view->elems;
}

#if NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE

__device__ inline void
failLane(GpunetioDeviceLane *lane) {
    deviceAtomic(lane->failed).store(1, cuda::memory_order_relaxed);
    deviceAtomic(lane->phase).store(gpunetio_lane_phase_failed, cuda::memory_order_release);
}

__device__ inline void
unlockLane(GpunetioDeviceLane *lane) {
    const uint32_t phase = lane->failed ?
        gpunetio_lane_phase_failed :
        (lane->next_retire == lane->active_ticket ? gpunetio_lane_phase_idle :
                                                    gpunetio_lane_phase_posted);
    deviceAtomic(lane->phase).store(phase, cuda::memory_order_release);
}

__device__ inline void
publish(GpunetioDeviceLane *lane) {
    if (lane->active_ticket != lane->next_retire) {
        doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                  DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                  DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(lane->qp,
                                                                             lane->active_ticket);
    }
}

// The scalar owner checks every CQE. Polling only the target would skip earlier errors.
__device__ inline nixl_status_t
progress(GpunetioDeviceLane *lane, uint64_t target) {
    while (lane->next_retire < target) {
        const int result = compat::pollOne(lane->qp, lane->next_retire);
        if (result == EBUSY) {
            return NIXL_IN_PROG;
        }
        if (result != 0) {
            failLane(lane);
            return NIXL_ERR_BACKEND;
        }
        // Publish completion to pollers in other CTAs after acquiring NIC writes.
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        deviceAtomic(lane->next_retire).store(lane->next_retire + 1, cuda::memory_order_release);
    }
    return NIXL_SUCCESS;
}

enum class operation { get, put, atomic_add };

template<operation op>
__device__ nixl_status_t
submitLocked(GpunetioDeviceLane *lane,
             const GpunetioDeviceContext *context,
             uint64_t local_address,
             uint32_t local_key,
             uint64_t remote_address,
             uint32_t remote_key,
             size_t bytes,
             uint64_t value,
             unsigned channel,
             uint64_t flags,
             xferStatusH *status) {
    if (lane->failed || !lane->qp || !compat::usesGpuSmDoorbell(lane->qp)) {
        failLane(lane);
        return NIXL_ERR_BACKEND;
    }
    if (compat::needsDump(lane->qp)) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    auto *cq = doca_gpu_dev_verbs_qp_get_cq_sq(lane->qp);
    const uint32_t capacity = cq ? min(uint32_t(lane->qp->sq_wqe_num), cq->cqe_num) : 0;
    const uint64_t chunks = 1 + (bytes - 1) / DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
    if (capacity < 2 || chunks > ~uint64_t{0} - lane->active_ticket) {
        return NIXL_ERR_BACKEND;
    }
    if (lane->generation == ~uint64_t{0}) {
        return NIXL_ERR_BACKEND;
    }
    ++lane->generation;
    uint64_t ticket = 0;
    size_t offset = 0;
    do {
        // Retire before reserve: the SDK availability wait does not propagate CQ errors.
        while (lane->active_ticket - lane->next_retire >= capacity - 1) {
            publish(lane);
            if (progress(lane, lane->next_retire + 1) == NIXL_ERR_BACKEND) {
                return NIXL_ERR_BACKEND;
            }
        }
        const size_t remaining = bytes - offset;
        const size_t chunk = remaining < DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ?
            remaining :
            DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
        ticket = doca_gpu_dev_verbs_reserve_wq_slots<>(lane->qp, 1);
        auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(lane->qp, ticket);
        // A fence also orders a preceding READ/atomic before this channel operation.
        constexpr auto ctrl = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE_FENCE;
        if constexpr (op == operation::get) {
            doca_gpu_dev_verbs_wqe_prepare_read(lane->qp,
                                                wqe,
                                                ticket,
                                                ctrl,
                                                remote_address + offset,
                                                remote_key,
                                                local_address + offset,
                                                local_key,
                                                chunk);
        } else if constexpr (op == operation::put) {
            doca_gpu_dev_verbs_wqe_prepare_write(lane->qp,
                                                 wqe,
                                                 ticket,
                                                 DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
                                                 ctrl,
                                                 0,
                                                 remote_address + offset,
                                                 remote_key,
                                                 local_address + offset,
                                                 local_key,
                                                 chunk);
        } else {
            doca_gpu_dev_verbs_wqe_prepare_atomic(lane->qp,
                                                  wqe,
                                                  ticket,
                                                  DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_FA,
                                                  ctrl,
                                                  remote_address,
                                                  remote_key,
                                                  context->atomic_results +
                                                      channel * sizeof(uint64_t),
                                                  context->atomic_key,
                                                  sizeof(uint64_t),
                                                  value,
                                                  0);
        }
        doca_gpu_dev_verbs_mark_wqes_ready<>(lane->qp, ticket, ticket);
        lane->active_ticket = ticket + 1;
        offset += chunk;
    } while (offset < bytes);
    if (!(flags & nixl::gpu::flags::defer)) {
        publish(lane);
    }
    if (status) {
        auto *payload = statusPayload(status);
        *payload = {lane, ticket, NIXL_IN_PROG, 1};
        status->mode = exec_mode_t::GPUNETIO_DIRECT;
    }
    return NIXL_IN_PROG;
}

template<operation op>
__device__ nixl_status_t
submit(const memViewElem *local,
       const memViewElem &remote,
       size_t bytes,
       uint64_t value,
       unsigned channel,
       uint64_t flags,
       xferStatusH *status) {
    if (flags & ~nixl::gpu::flags::defer) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (status) {
        if (status->mode != exec_mode_t{} && status->mode != exec_mode_t::GPUNETIO_DIRECT) {
            return NIXL_ERR_INVALID_PARAM;
        }
        if (status->mode == exec_mode_t::GPUNETIO_DIRECT && statusPayload(status)->initialized &&
            statusPayload(status)->cached_status == NIXL_IN_PROG) {
            return NIXL_ERR_REPOST_ACTIVE;
        }
    }
    const auto *r = static_cast<const GpunetioDeviceView *>(remote.mvh);
    if (!validView(r, gpunetio_device_view_role_remote) || channel >= r->context->channel_count ||
        remote.index >= r->count || !bytes) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto &re = r->elems[remote.index];
    uint64_t ra, la = 0;
    uint32_t lk = 0;
    if (!re.valid || re.peer_slot != gpunetio_remote_peer_slot ||
        !rangeIsValid(re.base, re.length, remote.offset, bytes, &ra)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if constexpr (op == operation::atomic_add) {
        if (ra % sizeof(uint64_t) || !r->context->atomic_results) {
            return NIXL_ERR_INVALID_PARAM;
        }
    } else {
        const auto *l = static_cast<const GpunetioDeviceView *>(local->mvh);
        if (!validView(l, gpunetio_device_view_role_local) || local->index >= l->count ||
            l->context_cookie != r->context_cookie || l->execution_gpu != r->execution_gpu ||
            l->context->channels != r->context->channels) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const auto &le = l->elems[local->index];
        if (!le.valid || le.peer_slot != gpunetio_local_peer_slot ||
            !rangeIsValid(le.base, le.length, local->offset, bytes, &la)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        lk = le.key;
    }
    auto *lane = r->context->channels + channel;
    auto phase = deviceAtomic(lane->phase);
    for (;;) {
        uint32_t previous = phase.load(cuda::memory_order_acquire);
        if (previous == gpunetio_lane_phase_failed) {
            return NIXL_ERR_BACKEND;
        }
        if (previous == gpunetio_lane_phase_posting) {
            return NIXL_ERR_NOT_ALLOWED; // A readSession owns the lane until finish().
        }
        if ((previous == gpunetio_lane_phase_idle || previous == gpunetio_lane_phase_posted) &&
            phase.compare_exchange_weak(previous,
                                        gpunetio_lane_phase_scalar,
                                        cuda::memory_order_acq_rel,
                                        cuda::memory_order_acquire)) {
            // Unlock inside this branch before leaving the loop: same-warp callers may contend.
            const auto result = submitLocked<op>(
                lane, r->context, la, lk, ra, re.key, bytes, value, channel, flags, status);
            unlockLane(lane);
            return result;
        }
    }
}

__device__ inline nixl_status_t
pollStatus(xferStatusH &status) {
    auto *payload = statusPayload(&status);
    if (!payload->initialized || !payload->lane) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (payload->cached_status != NIXL_IN_PROG) {
        return static_cast<nixl_status_t>(payload->cached_status);
    }
    auto *lane = payload->lane;
    if (payload->ticket < deviceAtomic(lane->next_retire).load(cuda::memory_order_acquire)) {
        payload->cached_status = NIXL_SUCCESS;
        return NIXL_SUCCESS;
    }
    auto phase = deviceAtomic(lane->phase);
    uint32_t previous = phase.load(cuda::memory_order_acquire);
    if (previous == gpunetio_lane_phase_failed) {
        // Acquiring FAILED also acquires the successfully checked prefix before the error.
        const auto result =
            payload->ticket < deviceAtomic(lane->next_retire).load(cuda::memory_order_acquire) ?
            NIXL_SUCCESS :
            NIXL_ERR_BACKEND;
        payload->cached_status = result;
        return result;
    }
    if (previous != gpunetio_lane_phase_posted ||
        !phase.compare_exchange_strong(previous,
                                       gpunetio_lane_phase_scalar,
                                       cuda::memory_order_acq_rel,
                                       cuda::memory_order_acquire)) {
        return NIXL_IN_PROG;
    }
    publish(lane);
    const auto result = progress(lane, payload->ticket + 1);
    payload->cached_status = result;
    unlockLane(lane);
    return result;
}

__device__ inline nixl_status_t
flushChannel(nixlMemViewH remote, unsigned channel) {
    const auto *r = static_cast<const GpunetioDeviceView *>(remote);
    if (!validView(r, gpunetio_device_view_role_remote) || channel >= r->context->channel_count) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *lane = r->context->channels + channel;
    auto phase = deviceAtomic(lane->phase);
    for (;;) {
        uint32_t previous = phase.load(cuda::memory_order_acquire);
        if (previous == gpunetio_lane_phase_failed) {
            return NIXL_ERR_BACKEND;
        }
        if (previous == gpunetio_lane_phase_posting) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        if ((previous == gpunetio_lane_phase_idle || previous == gpunetio_lane_phase_posted) &&
            phase.compare_exchange_weak(previous,
                                        gpunetio_lane_phase_scalar,
                                        cuda::memory_order_acq_rel,
                                        cuda::memory_order_acquire)) {
            publish(lane);
            nixl_status_t result;
            do {
                result = progress(lane, lane->active_ticket);
            } while (result == NIXL_IN_PROG);
            unlockLane(lane);
            return result;
        }
    }
}
#endif

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &status) {
#if NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE
    if constexpr (level != level_t::GRID) {
        nixl_status_t result = NIXL_IN_PROG;
        if (isLeader<level>()) {
            result = pollStatus(status);
        }
        result = broadcastStatus<level>(result);
        if constexpr (level == level_t::WARP) {
            sync<level>();
        }
        return result;
    }
#endif
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
get(const memViewElem &dst,
    const memViewElem &src,
    size_t bytes,
    unsigned channel,
    uint64_t flags,
    xferStatusH *status) {
#if NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE
    if constexpr (level != level_t::GRID) {
        sync<level>();
        nixl_status_t result = NIXL_IN_PROG;
        if (isLeader<level>()) {
            result = submit<operation::get>(&dst, src, bytes, 0, channel, flags, status);
        }
        result = broadcastStatus<level>(result);
        if constexpr (level == level_t::WARP) {
            sync<level>();
        }
        return result;
    }
#endif
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t bytes,
    unsigned channel,
    uint64_t flags,
    xferStatusH *status) {
#if NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE
    if constexpr (level != level_t::GRID) {
        __threadfence_system();
        sync<level>();
        nixl_status_t result = NIXL_IN_PROG;
        if (isLeader<level>()) {
            result = submit<operation::put>(&src, dst, bytes, 0, channel, flags, status);
        }
        result = broadcastStatus<level>(result);
        if constexpr (level == level_t::WARP) {
            sync<level>();
        }
        return result;
    }
#endif
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel,
          uint64_t flags,
          xferStatusH *status) {
#if NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE
    if constexpr (level != level_t::GRID) {
        sync<level>();
        nixl_status_t result = NIXL_IN_PROG;
        if (isLeader<level>()) {
            result = submit<operation::atomic_add>(
                nullptr, counter, sizeof(uint64_t), value, channel, flags, status);
        }
        result = broadcastStatus<level>(result);
        if constexpr (level == level_t::WARP) {
            sync<level>();
        }
        return result;
    }
#endif
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
flush(nixlMemViewH remote, unsigned channel) {
#if NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE
    if constexpr (level != level_t::GRID) {
        nixl_status_t result = NIXL_IN_PROG;
        if (isLeader<level>()) {
            result = flushChannel(remote, channel);
        }
        result = broadcastStatus<level>(result);
        if constexpr (level == level_t::WARP) {
            sync<level>();
        }
        return result;
    }
#endif
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline void *
getPtr(void *, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::impl::gpunetio
#endif
