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

// Every SQ/CQ access below happens while the caller owns the lane (phase SCALAR),
// and a readSession only touches the QP while it owns the lane (phase POSTING).
// The lane's device-scope acquire/release orders the SDK's plain index updates
// between owners, so the SDK needs no atomics or QP lock of its own here. Taking
// the lane needs only acquire ordering: unlockLane's release store publishes.
inline constexpr auto owned_qp_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;

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
        doca_gpu_dev_verbs_submit<owned_qp_mode,
                                  DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                  DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(lane->qp,
                                                                             lane->active_ticket);
    }
}

// The scalar owner checks every CQE. Polling only the target would skip earlier errors.
__device__ inline nixl_status_t
progress(GpunetioDeviceLane *lane, uint64_t target) {
    uint64_t next = lane->next_retire;
    nixl_status_t result = NIXL_SUCCESS;
    while (next < target) {
        // On success the SDK has issued its NIC acquire fence and advanced cqe_ci.
        const int rc = compat::pollOne<owned_qp_mode>(lane->qp, next);
        if (rc == EBUSY) {
            result = NIXL_IN_PROG;
            break;
        }
        if (rc != 0) {
            result = NIXL_ERR_BACKEND;
            break;
        }
        ++next;
    }
    if (next != lane->next_retire) {
        // Publish the checked prefix to pollers in other CTAs once per call.
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        deviceAtomic(lane->next_retire).store(next, cuda::memory_order_release);
    }
    if (result == NIXL_ERR_BACKEND) {
        failLane(lane); // After the successful prefix is visible.
    }
    return result;
}

__device__ inline nixl_status_t
submitGetLocked(GpunetioDeviceLane *lane,
                uint64_t local_address,
                uint32_t local_key,
                uint64_t remote_address,
                uint32_t remote_key,
                size_t bytes,
                uint64_t flags,
                xferStatusH *status) {
    // A bound lane keeps its QP, and a rebind writes a fresh lane, so the QP is checked
    // only until a GET stores its capacity. Later GETs read just the lane under the lock.
    uint32_t capacity = lane->capacity;
    if (lane->failed || !lane->qp || (!capacity && !compat::usesGpuSmDoorbell(lane->qp))) {
        failLane(lane);
        return NIXL_ERR_BACKEND;
    }
    if (!capacity) {
        if (compat::needsDump(lane->qp)) {
            return NIXL_ERR_NOT_SUPPORTED;
        }
        auto *cq = doca_gpu_dev_verbs_qp_get_cq_sq(lane->qp);
        capacity = cq ? min(uint32_t(lane->qp->sq_wqe_num), cq->cqe_num) : 0;
        if (capacity < 2) {
            return NIXL_ERR_BACKEND;
        }
        lane->capacity = capacity;
    }
    const uint64_t chunks = 1 + (bytes - 1) / DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
    if (chunks > ~uint64_t{0} - lane->active_ticket) {
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
        if (lane->active_ticket - lane->next_retire >= capacity - 1) {
            publish(lane); // A full queue may hold deferred WQEs; nothing is added while waiting.
            do {
                // Retire every completed CQE, not just one, before checking capacity again.
                if (progress(lane, lane->active_ticket) == NIXL_ERR_BACKEND) {
                    return NIXL_ERR_BACKEND;
                }
            } while (lane->active_ticket - lane->next_retire >= capacity - 1);
        }
        const size_t remaining = bytes - offset;
        const size_t chunk = remaining < DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ?
            remaining :
            DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
        ticket = doca_gpu_dev_verbs_reserve_wq_slots<owned_qp_mode>(lane->qp, 1);
        auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(lane->qp, ticket);
        // Signal each READ; completion is checked in FIFO order.
        constexpr auto ctrl = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
        doca_gpu_dev_verbs_wqe_prepare_read(lane->qp,
                                            wqe,
                                            ticket,
                                            ctrl,
                                            remote_address + offset,
                                            remote_key,
                                            local_address + offset,
                                            local_key,
                                            chunk);
        doca_gpu_dev_verbs_mark_wqes_ready<owned_qp_mode>(lane->qp, ticket, ticket);
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

__device__ inline nixl_status_t
submitGet(const memViewElem &local,
          const memViewElem &remote,
          size_t bytes,
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
        // A request retired by another poller, or on a failed channel, is final even if its
        // handle was never polled; only a still-outstanding request blocks reuse. Like
        // pollStatus, this reads old->lane: the engine that accepted it must still exist.
        const auto *old = statusPayload(status);
        if (status->mode == exec_mode_t::GPUNETIO_DIRECT && old->initialized &&
            old->cached_status == NIXL_IN_PROG && old->lane &&
            old->ticket >= deviceAtomic(old->lane->next_retire).load(cuda::memory_order_acquire) &&
            deviceAtomic(old->lane->phase).load(cuda::memory_order_acquire) !=
                gpunetio_lane_phase_failed) {
            return NIXL_ERR_REPOST_ACTIVE;
        }
    }
    const auto *r = static_cast<const GpunetioDeviceView *>(remote.mvh);
    if (!validView(r, gpunetio_device_view_role_remote) || channel >= r->context->channel_count ||
        remote.index >= r->count || !bytes) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto &re = r->elems[remote.index];
    uint64_t ra, la;
    if (!re.valid || re.peer_slot != gpunetio_remote_peer_slot ||
        !rangeIsValid(re.base, re.length, remote.offset, bytes, &ra)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto *l = static_cast<const GpunetioDeviceView *>(local.mvh);
    if (!validView(l, gpunetio_device_view_role_local) || local.index >= l->count ||
        l->context_cookie != r->context_cookie || l->execution_gpu != r->execution_gpu ||
        l->context->channels != r->context->channels) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto &le = l->elems[local.index];
    if (!le.valid || le.peer_slot != gpunetio_local_peer_slot ||
        !rangeIsValid(le.base, le.length, local.offset, bytes, &la)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *lane = r->context->channels + channel;
    auto phase = deviceAtomic(lane->phase);
    // Retry without sleeping: a nanosleep backoff measured slower
    // with 128 THREAD callers sharing one channel.
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
                                        cuda::memory_order_acquire,
                                        cuda::memory_order_acquire)) {
            // Unlock inside this branch before leaving the loop: same-warp callers may contend.
            const auto result = submitGetLocked(lane, la, le.key, ra, re.key, bytes, flags, status);
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
                                       cuda::memory_order_acquire,
                                       cuda::memory_order_acquire)) {
        return NIXL_IN_PROG;
    }
    publish(lane);
    // Retire every completed CQE, not only this request's, so other pollers resolve
    // on the lock-free fast path. A failure after this request keeps its success.
    progress(lane, lane->active_ticket);
    nixl_status_t result = NIXL_IN_PROG;
    if (payload->ticket < lane->next_retire) {
        result = NIXL_SUCCESS;
    } else if (lane->failed) {
        result = NIXL_ERR_BACKEND;
    }
    payload->cached_status = result;
    unlockLane(lane);
    return result;
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
            result = submitGet(dst, src, bytes, channel, flags, status);
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
