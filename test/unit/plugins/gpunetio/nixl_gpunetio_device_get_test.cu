/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime.h>
#include <gpu/nixl_device.cuh>
#include <gpu/impl/gpunetio/device_gpunetio_batch.cuh>
#include <nixl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
#include <set>

namespace {
constexpr size_t kBytes = 8192, kScratch = 8000, kScalarDst = 64, kBatchDst = 1024;
constexpr unsigned kCount = 16, kItemBytes = 128;
constexpr size_t kWrapDst = 2048;

void
check(cudaError_t rc, const char *what) {
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc));
    }
}

void
check(nixl_status_t rc, const char *what) {
    if (rc != NIXL_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + std::to_string(rc));
    }
}

__device__ uint64_t
globalTimerNs() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void
inspectLane(nixlMemViewH handle, uint64_t *generation, bool poison) {
    using namespace nixl::gpu::impl::gpunetio;
    const auto *w = static_cast<const nixl::gpu::memViewWrapper *>(handle);
    const auto *v = static_cast<const GpunetioDeviceView *>(w->backend_memview);
    auto *lane = v->context->lane;
    *generation = lane->generation;
    generation[1] = reinterpret_cast<uintptr_t>(lane);
    generation[2] = reinterpret_cast<uintptr_t>(lane->qp);
    if (poison) {
        deviceAtomic(lane->failed).store(1, cuda::memory_order_relaxed);
        deviceAtomic(lane->phase).store(gpunetio_lane_phase_failed, cuda::memory_order_release);
    }
}

__global__ void
rejectMixed(nixlMemViewH local, nixlMemViewH remote, int *result) {
    nixl::gpu::gpunetio::experimental::readSession session;
    if (session.bind(local, remote) != NIXL_ERR_INVALID_PARAM) {
        *result = 16;
    }
}

#if NIXL_GPUNETIO_PUBLIC_FIFO_ONLY
__global__ void
publicDefaultAdmission(nixlMemViewH local, nixlMemViewH remote, int *result) {
    nixl::gpu::gpunetio::experimental::readSession session;
    uint64_t ticket = 0;
    if (session.bind(local, remote) != NIXL_SUCCESS) {
        *result = 1;
        return;
    }
    const auto admission = session.acquire();
    if (admission == NIXL_ERR_NOT_SUPPORTED) {
        *result = 29;
        return;
    }
    *result = admission == NIXL_SUCCESS && session.reserve(1, ticket) == NIXL_ERR_NOT_SUPPORTED &&
            session.poll(ticket) == NIXL_ERR_NOT_SUPPORTED && session.finish() == NIXL_SUCCESS &&
            session.acquireFifo() == NIXL_SUCCESS && session.finish() == NIXL_SUCCESS ?
        0 :
        1;
}
#endif

__global__ void
scalarGet(nixlMemViewH local,
          nixlMemViewH remote,
          unsigned channel,
          uint8_t *destination,
          int *result) {
    if (threadIdx.x) {
        return;
    }
    nixl::gpu::xferStatusH status{};
    if (nixl::gpu::get(
            {local, 0, kScalarDst}, {remote, 0, kBytes - 10}, 256, channel, 0, &status) !=
        NIXL_ERR_INVALID_PARAM) {
        *result = 11;
        return;
    }
    auto rc = nixl::gpu::get({local, 0, kScalarDst}, {remote, 0, 17}, 256, channel, 0, &status);
    if (rc == NIXL_ERR_NOT_SUPPORTED) {
        *result = 29;
        return;
    }
    if (rc != NIXL_IN_PROG) {
        *result = 1;
        return;
    }
    // Take both snapshots while the first request is still active. One copy
    // completes it; polling the original must observe the same terminal result.
    auto completed_copy = status;
    auto stale_copy = status;
    const uint64_t copied_deadline = globalTimerNs() + 5000000000ull;
    do {
        rc = nixl::gpu::getXferStatus(completed_copy);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < copied_deadline);
    if (rc != NIXL_SUCCESS || nixl::gpu::getXferStatus(status) != NIXL_SUCCESS) {
        *result = 2;
        return;
    }

    for (size_t i = 0; i < 256; ++i) {
        if (destination[kScalarDst + i] != uint8_t(((17 + i) * 13 + 7) % 251)) {
            *result = 14;
            return;
        }
    }

    // A fixed-mode request records the selected mode in its active status.
    status = {};
    rc = nixl::gpu::get<nixl::gpu::level_t::THREAD, nixl::gpu::exec_mode_t::GPUNETIO_DIRECT>(
        {local, 0, kScalarDst + 256}, {remote, 0, 17}, 256, channel, 0, &status);
    if (rc != NIXL_IN_PROG) {
        *result = 25;
        return;
    }
    if (nixl::gpu::get({local, 0, kScalarDst}, {remote, 0, 17}, 256, channel, 0, &status) !=
        NIXL_ERR_REPOST_ACTIVE) {
        *result = 26;
        return;
    }
    // This stale IN_PROGRESS snapshot belongs to the prior generation. Polling
    // it while the fixed-mode request is active must leave that owner healthy.
    if (nixl::gpu::getXferStatus(stale_copy) != NIXL_SUCCESS) {
        *result = 27;
        return;
    }
    const uint64_t fixed_deadline = globalTimerNs() + 5000000000ull;
    do {
        rc = nixl::gpu::getXferStatus<nixl::gpu::level_t::THREAD,
                                      nixl::gpu::exec_mode_t::GPUNETIO_DIRECT>(status);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < fixed_deadline);
    if (rc != NIXL_SUCCESS) {
        *result = 28;
        return;
    }
    for (size_t i = 0; i < 256; ++i) {
        if (destination[kScalarDst + 256 + i] != uint8_t(((17 + i) * 13 + 7) % 251)) {
            *result = 14;
            return;
        }
    }
    status = {};
    rc = nixl::gpu::get({local, 0, kScalarDst + 512}, {remote, 0, 17}, 256, channel, 0, &status);
    if (rc != NIXL_IN_PROG) {
        *result = 30;
        return;
    }
    const uint64_t repost_deadline = globalTimerNs() + 5000000000ull;
    do {
        rc = nixl::gpu::getXferStatus(status);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < repost_deadline);
    if (rc != NIXL_SUCCESS) {
        *result = 31;
        return;
    }
    {
        // Consume immediately on the GPU, before a kernel boundary or CPU sync.
        for (size_t i = 0; i < 256; ++i) {
            if (destination[kScalarDst + 512 + i] != uint8_t(((17 + i) * 13 + 7) % 251)) {
                *result = 14;
                break;
            }
        }
    }
}

// Exercise multiple standard requests on the same channel without a caller scratch buffer.
__global__ void
standardGets(nixlMemViewH local, nixlMemViewH remote, unsigned channel, int *result) {
    using namespace nixl::gpu;
    xferStatusH a{}, b{}, c{};
    const memViewElem r{remote, 0, 17};
    if (get({local, 0, 4096}, r, 64, channel, flags::defer, &a) != NIXL_IN_PROG ||
        get({local, 0, 4160}, r, 64, channel, flags::defer, nullptr) != NIXL_IN_PROG ||
        get({local, 0, 4224}, r, 64, channel, flags::defer, &b) != NIXL_IN_PROG ||
        get({local, 0, 4288}, r, 64, channel, 0, &c) != NIXL_IN_PROG ||
        get({local, 0, 4352}, r, 64, channel, 0, &a) != NIXL_ERR_REPOST_ACTIVE) {
        *result = 40;
        return;
    }
    auto poll = [&](xferStatusH &s) {
        nixl_status_t rc;
        const auto deadline = globalTimerNs() + 5000000000ull;
        do {
            rc = getXferStatus(s);
        } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
        return rc;
    };
    if (poll(c) != NIXL_SUCCESS || poll(b) != NIXL_SUCCESS || poll(a) != NIXL_SUCCESS ||
        flush(remote, channel) != NIXL_SUCCESS ||
        get({local, 0, 4096}, r, 64, channel, 2, nullptr) != NIXL_ERR_INVALID_PARAM) {
        *result = 41;
        return;
    }
}

// Independent THREAD-level requests contend on one channel, including within a warp.
__global__ void
concurrentStandardGets(nixlMemViewH local,
                       nixlMemViewH remote,
                       unsigned channel,
                       const uint8_t *destination,
                       int *result) {
    const unsigned i = threadIdx.x;
    nixl::gpu::xferStatusH status{};
    auto rc =
        nixl::gpu::get({local, 0, 4096 + i * 16}, {remote, 0, i * 16}, 16, channel, 0, &status);
    if (rc != NIXL_IN_PROG) {
        atomicExch(result, 43);
        return;
    }
    const auto deadline = globalTimerNs() + 5000000000ull;
    do {
        rc = nixl::gpu::getXferStatus(status);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
    if (rc != NIXL_SUCCESS) {
        atomicExch(result, 44);
        return;
    }
    for (unsigned j = 0; j < 16; ++j) {
        if (destination[4096 + i * 16 + j] != uint8_t(((i * 16 + j) * 13 + 7) % 251)) {
            atomicExch(result, 45);
        }
    }
}

template<nixl::gpu::level_t level>
__global__ void
collectiveStandardGet(nixlMemViewH local,
                      nixlMemViewH remote,
                      const uint8_t *destination,
                      int *result) {
    constexpr unsigned groups_per_block = level == nixl::gpu::level_t::WARP ? 4 : 1;
    const unsigned group = level == nixl::gpu::level_t::WARP ? threadIdx.x / 32 : 0;
    const unsigned group_index = blockIdx.x * groups_per_block + group;
    alignas(16) __shared__ unsigned char storage[groups_per_block * sizeof(nixl::gpu::xferStatusH)];
    auto *status = reinterpret_cast<nixl::gpu::xferStatusH *>(storage) + group;
    if ((level == nixl::gpu::level_t::WARP && threadIdx.x % 32 == 0) ||
        (level == nixl::gpu::level_t::BLOCK && threadIdx.x == 0)) {
        new (status) nixl::gpu::xferStatusH{};
    }
    __syncthreads();
    const size_t dst = 4096 + group_index * 64;
    const size_t src = 17 + group_index * 64;
    auto rc = nixl::gpu::get<level>({local, 0, dst}, {remote, 0, src}, 64, 0, 0, status);
    if (rc != NIXL_IN_PROG) {
        atomicExch(result, 46);
        return;
    }
    for (unsigned attempt = 0; attempt < 1000000 && rc == NIXL_IN_PROG; ++attempt) {
        rc = nixl::gpu::getXferStatus<level>(*status);
    }
    if (rc != NIXL_SUCCESS) {
        atomicExch(result, 47);
        return;
    }
    for (unsigned i = 0; i < 64; ++i) {
        if (destination[dst + i] != uint8_t(((src + i) * 13 + 7) % 251)) {
            atomicExch(result, 52);
        }
    }
}

// Neither deferred GET has a status handle: flush is the only completion path.
__global__ void
nullDeferredGet(nixlMemViewH local, nixlMemViewH remote, const uint8_t *destination, int *result) {
    using namespace nixl::gpu;
    if (get({local, 0, 4096}, {remote, 0, 17}, 64, 0, flags::defer) != NIXL_IN_PROG ||
        get({local, 0, 4160}, {remote, 0, 81}, 64, 0, flags::defer) != NIXL_IN_PROG ||
        flush(remote, 0) != NIXL_SUCCESS) {
        *result = 53;
        return;
    }
    for (unsigned i = 0; i < 128; ++i) {
        if (destination[4096 + i] != uint8_t(((17 + i) * 13 + 7) % 251)) {
            *result = 54;
        }
    }
}

__global__ void
scalarWrap(nixlMemViewH local,
           nixlMemViewH remote,
           const uint8_t *destination,
           uint64_t *count_out,
           int *result) {
    using namespace nixl::gpu;
    using namespace nixl::gpu::impl::gpunetio;
    const auto *wrapper = static_cast<const memViewWrapper *>(remote);
    const auto *view = static_cast<const GpunetioDeviceView *>(wrapper->backend_memview);
    const uint64_t count = 2ull * view->context->channels[0].qp->sq_wqe_num + 3;
    *count_out = count;
    if (count > kBytes - kWrapDst) {
        *result = 55;
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        if (get({local, 0, kWrapDst + i}, {remote, 0, 17 + i}, 1, 0, flags::defer) !=
            NIXL_IN_PROG) {
            *result = 55;
            return;
        }
    }
    if (flush(remote, 0) != NIXL_SUCCESS) {
        *result = 56;
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        if (destination[kWrapDst + i] != uint8_t(((17 + i) * 13 + 7) % 251)) {
            *result = 56;
        }
    }
}

__global__ void
channelIndependence(nixlMemViewH local0,
                    nixlMemViewH remote0,
                    nixlMemViewH held_local,
                    nixlMemViewH held_remote,
                    unsigned held_channel,
                    const uint8_t *destination,
                    int *result) {
    using namespace nixl::gpu;
    gpunetio::experimental::readSession session;
    if (session.bind(held_local, held_remote) != NIXL_SUCCESS ||
        session.acquireFifo() != NIXL_SUCCESS) {
        *result = 57;
        return;
    }
    if (get({local0, 0, 4096}, {remote0, 0, 17}, 64, held_channel) != NIXL_ERR_NOT_ALLOWED ||
        get({local0, 0, 4096}, {remote0, 0, 17}, 64, 0) != NIXL_IN_PROG ||
        flush(remote0, 0) != NIXL_SUCCESS) {
        session.fail();
        *result = 58;
        return;
    }
    for (unsigned i = 0; i < 64; ++i) {
        if (destination[4096 + i] != uint8_t(((17 + i) * 13 + 7) % 251)) {
            *result = 59;
        }
    }
    if (session.finish() != NIXL_SUCCESS) {
        *result = 60;
    }
}

__global__ void
standardInvalid(nixlMemViewH local, nixlMemViewH remote, unsigned qps, int *result) {
    using namespace nixl::gpu;
    using namespace nixl::gpu::impl::gpunetio;
    const auto *wrapper = static_cast<const memViewWrapper *>(remote);
    auto old_view = *static_cast<const GpunetioDeviceView *>(wrapper->backend_memview);
    old_view.abi_version = 2;
    // Reject the old ABI before inspecting its differently sized context.
    old_view.context = reinterpret_cast<GpunetioDeviceContext *>(uintptr_t{1});
    memViewWrapper old_wrapper{exec_mode_t::GPUNETIO_DIRECT, &old_view};
    if (get({local, 0, 4096}, {&old_wrapper, 0, 17}, 64) != NIXL_ERR_INVALID_PARAM) {
        *result = 61;
        return;
    }
    if (get({local, 0, 4096}, {remote, 0, 17}, 64, qps) != NIXL_ERR_INVALID_PARAM ||
        flush(remote, qps) != NIXL_ERR_INVALID_PARAM ||
        get<level_t::GRID>({local, 0, 4096}, {remote, 0, 17}, 64) != NIXL_ERR_NOT_SUPPORTED ||
        put({local, 0, 4096}, {remote, 0, 4096}, 64) != NIXL_ERR_NOT_SUPPORTED ||
        atomicAdd(1, {remote, 0, 5000}) != NIXL_ERR_NOT_SUPPORTED ||
        put({nullptr, 0, 0}, {remote, 0, 4096}, 64) != NIXL_ERR_INVALID_PARAM ||
        atomicAdd(1, {nullptr, 0, 0}) != NIXL_ERR_INVALID_PARAM) {
        *result = 48;
    }
}

// Exercise the standard GET beyond one SDK WQE without allocating host-sized mirrors.
__global__ void
largeFill(uint8_t *buffer, size_t bytes, uint8_t salt, uint8_t guard) {
    const size_t total = bytes + 128;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += size_t(blockDim.x) * gridDim.x) {
        buffer[i] = (i < 64 || i >= bytes + 64) ? guard : uint8_t(((i - 64) * 13 + salt) % 251);
    }
}

__global__ void
largeVerify(const uint8_t *buffer, size_t bytes, uint8_t salt, uint8_t guard, int *result) {
    const size_t total = bytes + 128;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += size_t(blockDim.x) * gridDim.x) {
        const uint8_t expected =
            (i < 64 || i >= bytes + 64) ? guard : uint8_t(((i - 64) * 13 + salt) % 251);
        if (buffer[i] != expected) {
            atomicExch(result, 64);
        }
    }
}

__global__ void
largeTransfer(nixlMemViewH local, nixlMemViewH remote, size_t bytes, int *result) {
    using namespace nixl::gpu;
    xferStatusH status{};
    auto rc = get({local, 0, 64}, {remote, 0, 64}, bytes, 0, 0, &status);
    if (rc != NIXL_IN_PROG) {
        *result = 65;
        return;
    }
    const uint64_t deadline = globalTimerNs() + 60000000000ull;
    do {
        rc = getXferStatus(status);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
    if (rc != NIXL_SUCCESS || flush(remote, 0) != NIXL_SUCCESS) {
        *result = 66;
    }
}

// Disposable-process fault check: corrupt only one channel's imported key.
__global__ void
corruptRemoteKey(nixlMemViewH remote) {
    using namespace nixl::gpu::impl::gpunetio;
    const auto *wrapper = static_cast<const nixl::gpu::memViewWrapper *>(remote);
    const auto *view = static_cast<const GpunetioDeviceView *>(wrapper->backend_memview);
    const_cast<GpunetioViewElem *>(view->elems)[0].key = 0;
}

__global__ void
expectCqError(nixlMemViewH local, nixlMemViewH remote, unsigned channel, int *result, bool sticky) {
    nixl::gpu::xferStatusH status{};
    auto rc = nixl::gpu::get({local, 0, kScalarDst}, {remote, 0, 17}, 256, channel, 0, &status);
    if (sticky) {
        *result = rc == NIXL_ERR_BACKEND ? 0 : 19;
        return;
    }
    if (rc != NIXL_IN_PROG) {
        *result = 17;
        return;
    }
    const auto deadline = globalTimerNs() + 5000000000ULL;
    do {
        rc = nixl::gpu::getXferStatus(status);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
    *result = rc == NIXL_ERR_BACKEND ? 0 : 18;
}

// In a disposable process, a bad middle WQE must remain visible when polling a later handle.
__global__ void
expectStandardFifoCqError(nixlMemViewH local, nixlMemViewH remote, unsigned channel, int *result) {
    using namespace nixl::gpu;
    using namespace nixl::gpu::impl::gpunetio;
    xferStatusH first{}, last{};
    auto *wrapper = static_cast<const memViewWrapper *>(remote);
    auto *view = static_cast<const GpunetioDeviceView *>(wrapper->backend_memview);
    auto *elem = const_cast<GpunetioViewElem *>(view->elems);
    if (get({local, 0, 4096}, {remote, 0, 17}, 64, channel, flags::defer, &first) != NIXL_IN_PROG) {
        *result = 49;
        return;
    }
    const auto key = elem[0].key;
    elem[0].key = 0;
    const auto bad = get({local, 0, 4160}, {remote, 0, 17}, 64, channel, flags::defer, nullptr);
    elem[0].key = key;
    if (bad != NIXL_IN_PROG ||
        get({local, 0, 4224}, {remote, 0, 17}, 64, channel, flags::defer, &last) != NIXL_IN_PROG) {
        *result = 50;
        return;
    }
    auto rc = NIXL_IN_PROG;
    const auto deadline = globalTimerNs() + 5000000000ull;
    do {
        rc = getXferStatus(last);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
    if (rc != NIXL_ERR_BACKEND || getXferStatus(first) != NIXL_SUCCESS ||
        get({local, 0, 4288}, {remote, 0, 17}, 64, channel) != NIXL_ERR_BACKEND) {
        *result = 51;
    }
}

// Disposable-process FIFO fault check. Do not release the failed lane or MRs.
__global__ void
expectFifoCqError(nixlMemViewH local, nixlMemViewH remote, int *result) {
    using namespace nixl::gpu::impl::gpunetio;
    nixl::gpu::gpunetio::experimental::readSession session;
    uint64_t first = 0;
    if (session.bind(local, remote) != NIXL_SUCCESS || session.acquireFifo() != NIXL_SUCCESS ||
        session.reserve(3, first) != NIXL_SUCCESS ||
        session.prepare(first, 0, 17, 0, kBatchDst, 128) != NIXL_SUCCESS) {
        session.fail();
        *result = 22;
        return;
    }
    const auto *wrapper = static_cast<const nixl::gpu::memViewWrapper *>(remote);
    const auto *view = static_cast<const GpunetioDeviceView *>(wrapper->backend_memview);
    auto *elem = const_cast<GpunetioViewElem *>(view->elems);
    const auto saved_key = elem[0].key;
    elem[0].key = 0;
    const auto bad_prepare = session.prepare(first + 1, 0, 145, 0, kBatchDst + 128, 128);
    elem[0].key = saved_key;
    if (bad_prepare != NIXL_SUCCESS ||
        session.prepare(first + 2, 0, 273, 0, kBatchDst + 256, 128) != NIXL_SUCCESS ||
        session.publish(first, 3) != NIXL_IN_PROG) {
        session.fail();
        *result = 22;
        return;
    }
    uint64_t pending = 0;
    if (session.reserve(1, pending) != NIXL_SUCCESS ||
        session.prepare(pending, 0, 17, 0, kBatchDst + 512, 128) != NIXL_SUCCESS) {
        session.fail();
        *result = 22;
        return;
    }
    nixl_status_t rc;
    const uint64_t deadline = globalTimerNs() + 5000000000ull;
    do {
        rc = session.pollFifo(first);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
    if (rc != NIXL_SUCCESS) {
        session.fail();
        *result = 24;
        return;
    }
    do {
        rc = session.pollFifo(first + 1);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
    uint64_t rejected = 0;
    const bool sticky = rc == NIXL_ERR_BACKEND && session.pollFifo(first + 2) == NIXL_ERR_BACKEND &&
        session.reserve(1, rejected) == NIXL_ERR_BACKEND &&
        session.publish(pending, 1) == NIXL_ERR_BACKEND &&
        session.retire(first, 1) == NIXL_ERR_BACKEND && session.finish() == NIXL_ERR_BACKEND;
    session.fail();
    *result = sticky ? 0 : 23;
}

template<bool Fifo>
__global__ void
batchGet(nixlMemViewH local, nixlMemViewH remote, uint8_t *destination, int *result) {
    __shared__ __align__(
        16) unsigned char storage[sizeof(nixl::gpu::gpunetio::experimental::readSession)];
    __shared__ uint64_t first;
    __shared__ int error;
    auto *session = reinterpret_cast<nixl::gpu::gpunetio::experimental::readSession *>(storage);
    if (!threadIdx.x) {
        ::new (session) nixl::gpu::gpunetio::experimental::readSession{};
        error = 0;
        const auto bind_rc = session->bind(local, remote);
        const auto acquire_rc = bind_rc == NIXL_SUCCESS ?
            (Fifo ? session->acquireFifo() : session->acquire()) :
            bind_rc;
        if (acquire_rc != NIXL_SUCCESS) {
            error = acquire_rc == NIXL_ERR_NOT_SUPPORTED ? 29 : 3;
        }
        if (!error &&
            (session->reserve(0, first) != NIXL_ERR_INVALID_PARAM ||
             session->reserve(session->capacity(), first) != NIXL_ERR_INVALID_PARAM)) {
            error = 12;
        }
        if (!error && session->reserve(kCount, first) != NIXL_SUCCESS) {
            error = 3;
        }
        if (!error &&
            session->prepare(first + kCount, 0, 0, 0, kBatchDst, kItemBytes) !=
                NIXL_ERR_INVALID_PARAM) {
            error = 13;
        }
    }
    __syncthreads();
    if (atomicAdd(&error, 0) == 0 && threadIdx.x < kCount &&
        session->prepare(first + threadIdx.x,
                         0,
                         512 + threadIdx.x * kItemBytes,
                         0,
                         kBatchDst + threadIdx.x * kItemBytes,
                         kItemBytes) != NIXL_SUCCESS) {
        atomicExch(&error, 4);
    }
    __syncthreads();
    if (!threadIdx.x && !error && session->publish(first, kCount) != NIXL_IN_PROG) {
        error = 5;
    }
    __syncthreads();
    if constexpr (Fifo) {
        if (!threadIdx.x && !error) {
            if (session->pollFifo(first + 1) != NIXL_ERR_INVALID_PARAM ||
                session->retire(first, 1) != NIXL_ERR_INVALID_PARAM ||
                session->poll(first) != NIXL_ERR_INVALID_PARAM) {
                error = 19;
            }
            for (uint32_t i = 0; !error && i < kCount; ++i) {
                nixl_status_t rc;
                const uint64_t deadline = globalTimerNs() + 5000000000ull;
                do {
                    rc = session->pollFifo(first + i);
                } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
                if (rc != NIXL_SUCCESS) {
                    error = 6;
                } else if (session->pollFifo(first + i) != NIXL_ERR_INVALID_PARAM) {
                    error = 20;
                }
                if (!error && i == kCount / 2 - 1) {
                    if (session->retire(first, kCount) != NIXL_ERR_INVALID_PARAM ||
                        session->retire(first, kCount / 2) != NIXL_SUCCESS ||
                        session->retire(first + kCount / 2, 1) != NIXL_ERR_INVALID_PARAM) {
                        error = 21;
                    }
                }
            }
        }
    } else if (atomicAdd(&error, 0) == 0 && threadIdx.x < kCount) {
        nixl_status_t rc;
        const uint64_t deadline = globalTimerNs() + 5000000000ull;
        do {
            rc = session->poll(first + threadIdx.x);
        } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
        if (rc != NIXL_SUCCESS) {
            atomicExch(&error, 6);
        }
    }
    __syncthreads();
    if (!threadIdx.x) {
        if (!error &&
            session->retire(Fifo ? first + kCount / 2 : first, Fifo ? kCount / 2 : kCount) !=
                NIXL_SUCCESS) {
            error = 7;
        }
        if (!error && session->needsConsistency()) {
            uint64_t ticket;
            if (session->postConsistency(0, kScratch, ticket) != NIXL_IN_PROG) {
                error = 8;
            }
            if (!error) {
                nixl_status_t rc;
                const uint64_t deadline = globalTimerNs() + 5000000000ull;
                do {
                    rc = Fifo ? session->pollFifo(ticket) : session->poll(ticket);
                } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
                if (rc != NIXL_SUCCESS || session->retire(ticket, 1) != NIXL_SUCCESS) {
                    error = 9;
                }
            }
        }
        if (!error && session->finish() != NIXL_SUCCESS) {
            error = 10;
        }
        if (error) {
            session->fail();
        }
        *result = error;
    }
    __syncthreads();
    if (!error) {
        for (size_t i = threadIdx.x; i < kCount * kItemBytes; i += blockDim.x) {
            if (destination[kBatchDst + i] != uint8_t(((512 + i) * 13 + 7) % 251)) {
                atomicExch(result, 15);
            }
        }
    }
}

// Exercise the FIFO cursor while a second batch is outstanding and a third is posted.
__global__ void
fifoPipeline(nixlMemViewH local, nixlMemViewH remote, const uint8_t *destination, int *result) {
    if (threadIdx.x) {
        return;
    }
    nixl::gpu::gpunetio::experimental::readSession session;
    uint64_t base[3]{};
    int error = 0;
    if (session.bind(local, remote) != NIXL_SUCCESS || session.acquireFifo() != NIXL_SUCCESS) {
        *result = 31;
        return;
    }
    if (session.needsConsistency()) {
        *result = session.finish() == NIXL_SUCCESS ? 30 : 31;
        return;
    }
    auto post = [&](unsigned batch) {
        if (session.reserve(8, base[batch]) != NIXL_SUCCESS) {
            return false;
        }
        for (unsigned i = 0; i < 8; ++i) {
            size_t offset = (batch * 8 + i) * kItemBytes;
            if (session.prepare(
                    base[batch] + i, 0, 512 + offset, 0, kBatchDst + offset, kItemBytes) !=
                NIXL_SUCCESS) {
                return false;
            }
        }
        return session.publish(base[batch], 8) == NIXL_IN_PROG;
    };
    auto poll = [&](uint64_t first, unsigned count) {
        for (unsigned i = 0; i < count; ++i) {
            nixl_status_t rc;
            const uint64_t deadline = globalTimerNs() + 5000000000ull;
            do {
                rc = session.pollFifo(first + i);
            } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
            if (rc != NIXL_SUCCESS) {
                return false;
            }
        }
        return true;
    };
    if (!post(0) || !post(1) || base[1] != base[0] + 8 || !poll(base[0], 4) ||
        session.retire(base[0], 4) != NIXL_SUCCESS || !post(2) || base[2] != base[1] + 8 ||
        !poll(base[0] + 4, 4) || session.retire(base[0] + 4, 4) != NIXL_SUCCESS ||
        !poll(base[1], 8) || session.retire(base[1], 8) != NIXL_SUCCESS || !poll(base[2], 8) ||
        session.retire(base[2], 8) != NIXL_SUCCESS || session.finish() != NIXL_SUCCESS) {
        error = 32;
    }
    if (error) {
        session.fail();
    }
    if (!error) {
        for (size_t i = 0; i < 24 * kItemBytes; ++i) {
            if (destination[kBatchDst + i] != uint8_t(((512 + i) * 13 + 7) % 251)) {
                error = 33;
                break;
            }
        }
    }
    *result = error;
}

using ReadSession = nixl::gpu::gpunetio::experimental::readSession;

struct PersistentRead {
    alignas(16) unsigned char storage[sizeof(ReadSession)];
    uint64_t ticket;
    int result;
};

// Each graph node performs one state transition. Graph ordering keeps the lane
// owned across nodes without relying on concurrent CTA scheduling.
__global__ void
publishPersistent(nixlMemViewH local,
                  nixlMemViewH remote,
                  size_t dst_offset,
                  size_t src_offset,
                  PersistentRead *state) {
    auto *session = ::new (state->storage) ReadSession{};
    state->result = session->bind(local, remote);
    if (state->result == NIXL_SUCCESS) {
        state->result = session->acquireFifo();
    }
    if (state->result == NIXL_SUCCESS) {
        state->result = session->reserve(1, state->ticket);
    }
    if (state->result == NIXL_SUCCESS) {
        state->result = session->prepare(state->ticket, 0, src_offset, 0, dst_offset, kItemBytes);
    }
    if (state->result == NIXL_SUCCESS) {
        state->result = session->publish(state->ticket, 1);
        if (state->result == NIXL_IN_PROG) {
            state->result = NIXL_SUCCESS;
        }
    }
    if (state->result != NIXL_SUCCESS) {
        session->fail();
    }
}

__global__ void
rejectBusyLane(nixlMemViewH local, nixlMemViewH remote, int *result) {
    ReadSession contender;
    const auto bind_status = contender.bind(local, remote);
    const auto acquire_status = bind_status == NIXL_SUCCESS ? contender.acquireFifo() : bind_status;
    if (acquire_status == NIXL_SUCCESS) {
        contender.finish();
    }
    *result = acquire_status == NIXL_ERR_NOT_ALLOWED ? 0 : 1;
}

__global__ void
retirePersistent(const uint8_t *destination,
                 size_t dst_offset,
                 size_t src_offset,
                 PersistentRead *state) {
    auto *session = reinterpret_cast<ReadSession *>(state->storage);
    if (state->result != NIXL_SUCCESS) {
        return;
    }
    const uint64_t deadline = globalTimerNs() + 5000000000ull;
    nixl_status_t rc;
    do {
        rc = session->pollFifo(state->ticket);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
    if (rc == NIXL_SUCCESS) {
        rc = session->retire(state->ticket, 1);
    }
    if (rc == NIXL_SUCCESS && session->needsConsistency()) {
        uint64_t consistency = 0;
        rc = session->postConsistency(0, kScratch, consistency);
        if (rc == NIXL_IN_PROG) {
            do {
                rc = session->pollFifo(consistency);
            } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
            if (rc == NIXL_SUCCESS) {
                rc = session->retire(consistency, 1);
            }
        }
    }
    if (rc == NIXL_SUCCESS) {
        rc = session->finish();
    } else {
        session->fail();
    }
    if (rc != NIXL_SUCCESS) {
        state->result = rc;
        return;
    }
    for (size_t i = 0; i < kItemBytes; ++i) {
        if (destination[dst_offset + i] != uint8_t(((src_offset + i) * 13 + 7) % 251)) {
            state->result = 1;
            return;
        }
    }
}
} // namespace

int
main(int argc, char **argv) {
    if (argc < 6 || argc > 8) {
        std::cerr << "usage: " << argv[0]
                  << " <rdma-device> <oob-interface> <gid-index> <base-port> <owner-gpu> "
                     "[DRAM|VRAM] [reader-gpu]\n";
        return 2;
    }
    try {
        const int port = std::stoi(argv[4]), owner_gpu = std::stoi(argv[5]);
        const int reader_gpu = argc == 8 ? std::stoi(argv[7]) : owner_gpu;
        if (port < 1 || port >= 65535 || owner_gpu < 0 || reader_gpu < 0) {
            throw std::runtime_error("invalid port or GPU");
        }
        const std::string source_type = argc >= 7 ? argv[6] : "DRAM";
        if (source_type != "DRAM" && source_type != "VRAM") {
            throw std::runtime_error("invalid source type");
        }
        const bool source_is_host = source_type == "DRAM";
        const char *qps_env = std::getenv("NIXL_GPUNETIO_TEST_QPS");
        const unsigned qps = qps_env ? std::stoul(qps_env) : 1;
        if (qps < 1 || qps > 32) {
            throw std::runtime_error("invalid test QP count");
        }
        const char *peer_qps = std::getenv("NIXL_GPUNETIO_TEST_PEER_QPS");
        if (peer_qps) {
            const auto owner_qps = std::stoul(peer_qps);
            if (owner_qps < 2 || owner_qps > 32 || qps < 2 || owner_qps == qps) {
                throw std::runtime_error("peer QP mismatch test needs unequal counts in [2, 32]");
            }
        }
        const bool large_test = std::getenv("NIXL_GPUNETIO_TEST_LARGE") &&
            std::string(std::getenv("NIXL_GPUNETIO_TEST_LARGE")) == "1";
        if (large_test && peer_qps) {
            throw std::runtime_error("large test requires matching QPs");
        }
        constexpr size_t large_bytes = DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE + 64;
        constexpr size_t large_allocation = large_bytes + 128;
        uint8_t *large_owner = nullptr, *large_reader = nullptr;
        const nixl_mem_t source_mem = source_is_host ? DRAM_SEG : VRAM_SEG;
        check(cudaSetDevice(owner_gpu), "select owner GPU");
        uint8_t *source = nullptr, *destination = nullptr;
        int *device_result = nullptr;
        uint8_t expected_source[kBytes];
        for (size_t i = 0; i < kBytes; ++i) {
            expected_source[i] = static_cast<uint8_t>((i * 13 + 7) % 251);
        }
        if (source_is_host) {
            check(cudaHostAlloc(&source, kBytes, cudaHostAllocDefault), "allocate pinned source");
            std::copy(expected_source, expected_source + kBytes, source);
        } else {
            check(cudaMalloc(&source, kBytes), "allocate VRAM source");
            check(cudaMemcpy(source, expected_source, kBytes, cudaMemcpyHostToDevice),
                  "initialize VRAM source");
        }
        if (large_test) {
            check(cudaMalloc(&large_owner, large_allocation), "allocate large owner VRAM");
        }
        check(cudaSetDevice(reader_gpu), "select reader GPU");
        if (large_test) {
            check(cudaMalloc(&large_reader, large_allocation), "allocate large reader VRAM");
        }
        check(cudaMalloc(&destination, kBytes), "allocate destination");
        check(cudaMalloc(&device_result, sizeof(int)), "allocate status");
        check(cudaMemset(destination, 0xa5, kBytes), "initialize destination");
        check(cudaMemset(device_result, 0, sizeof(int)), "initialize status");

        nixlAgentConfig config;
        config.useProgThread = true;
        config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        const auto suffix = std::to_string(getpid());
        nixlAgent owner("gpunetio-get-owner-" + suffix, config);
        nixlAgent reader("gpunetio-get-reader-" + suffix, config);
        nixlBackendH *owner_backend = nullptr, *reader_backend = nullptr;
        auto params = [&](int value, int device) {
            return nixl_b_params_t{{"native_device_api", "true"},
                                   {"native_qp_count", std::to_string(qps)},
                                   {"gpu_devices", std::to_string(device)},
                                   {"network_devices", argv[1]},
                                   {"oob_interface", argv[2]},
                                   {"gid_index", argv[3]},
                                   {"oob_port", std::to_string(value)}};
        };
        auto owner_params = params(port, owner_gpu), reader_params = params(port + 1, reader_gpu);
        if (peer_qps) {
            owner_params["native_qp_count"] = peer_qps;
        }
        check(cudaSetDevice(owner_gpu), "select owner GPU");
        check(owner.createBackend("GPUNETIO", owner_params, owner_backend), "create owner backend");
        check(cudaSetDevice(reader_gpu), "select reader GPU");
        check(reader.createBackend("GPUNETIO", reader_params, reader_backend),
              "create reader backend");
        nixl_opt_args_t owner_hint, reader_hint;
        owner_hint.backends.push_back(owner_backend);
        reader_hint.backends.push_back(reader_backend);
        nixl_reg_dlist_t owner_reg(source_mem), reader_reg(VRAM_SEG);
        nixl_reg_dlist_t large_owner_reg(VRAM_SEG), large_reader_reg(VRAM_SEG);
        if (large_test) {
            large_owner_reg.addDesc({reinterpret_cast<uintptr_t>(large_owner),
                                     large_allocation,
                                     static_cast<uint64_t>(owner_gpu),
                                     {}});
            large_reader_reg.addDesc({reinterpret_cast<uintptr_t>(large_reader),
                                      large_allocation,
                                      static_cast<uint64_t>(reader_gpu),
                                      {}});
        }
        owner_reg.addDesc(
            {reinterpret_cast<uintptr_t>(source), kBytes, static_cast<uint64_t>(owner_gpu), {}});
        reader_reg.addDesc({reinterpret_cast<uintptr_t>(destination),
                            kBytes,
                            static_cast<uint64_t>(reader_gpu),
                            {}});
        check(cudaSetDevice(owner_gpu), "select owner GPU");
        check(owner.registerMem(owner_reg, &owner_hint), "register source");
        if (large_test) {
            check(owner.registerMem(large_owner_reg, &owner_hint), "register large owner");
        }
        check(cudaSetDevice(reader_gpu), "select reader GPU");
        check(reader.registerMem(reader_reg, &reader_hint), "register destination");
        if (large_test) {
            check(reader.registerMem(large_reader_reg, &reader_hint), "register large reader");
        }
        nixl_blob_t metadata;
        check(owner.getLocalMD(metadata), "export metadata");
        std::string peer;
        if (peer_qps) {
            const auto status = reader.loadRemoteMD(metadata, peer);
            if (status != NIXL_ERR_INVALID_PARAM) {
                throw std::runtime_error(
                    "mismatched pool did not return INVALID_PARAM at metadata load: " +
                    std::to_string(status));
            }
            std::cout << "PASS mismatched native QP counts rejected at metadata load; "
                         "SETUP_RESOURCES_RETAINED_UNTIL_EXIT\n"
                      << std::flush;
            std::_Exit(0);
        }
        check(reader.loadRemoteMD(metadata, peer), "load metadata");
        const auto connection = reader.makeConnection(peer, &reader_hint);
        check(connection, "connect");
        nixl_local_dlist_t local(VRAM_SEG);
        local.addDesc(
            {reinterpret_cast<uintptr_t>(destination), kBytes, static_cast<uint64_t>(reader_gpu)});
        nixl_remote_dlist_t remote(source_mem);
        remote.addDesc(
            {reinterpret_cast<uintptr_t>(source), kBytes, static_cast<uint64_t>(owner_gpu), peer});
        std::vector<nixlMemViewH> locals(qps), remotes(qps);
        for (unsigned channel = 0; channel < qps; ++channel) {
            auto hint = reader_hint;
            hint.customParam = "native_channel=" + std::to_string(channel);
            check(reader.prepMemView(local, locals[channel], &hint), "prepare channel local view");
            check(reader.prepMemView(remote, remotes[channel], &hint),
                  "prepare channel remote view");
        }
        nixlMemViewH large_local_view = nullptr, large_remote_view = nullptr;
        if (large_test) {
            nixl_local_dlist_t large_local(VRAM_SEG);
            large_local.addDesc({reinterpret_cast<uintptr_t>(large_reader),
                                 large_allocation,
                                 static_cast<uint64_t>(reader_gpu)});
            nixl_remote_dlist_t large_remote(VRAM_SEG);
            large_remote.addDesc({reinterpret_cast<uintptr_t>(large_owner),
                                  large_allocation,
                                  static_cast<uint64_t>(owner_gpu),
                                  peer});
            check(reader.prepMemView(large_local, large_local_view, &reader_hint),
                  "prepare large local");
            check(reader.prepMemView(large_remote, large_remote_view, &reader_hint),
                  "prepare large remote");
        }
        auto invalid_hint = reader_hint;
        invalid_hint.customParam = "native_channel=" + std::to_string(qps);
        nixlMemViewH invalid_view = nullptr;
        if (reader.prepMemView(local, invalid_view, &invalid_hint) != NIXL_ERR_INVALID_PARAM ||
            invalid_view) {
            throw std::runtime_error("out-of-range channel accepted");
        }
        if (qps > 1) {
            rejectMixed<<<1, 1>>>(locals[0], remotes[1], device_result);
            check(cudaDeviceSynchronize(), "mixed-channel rejection");
            int error = 0;
            check(cudaMemcpy(&error, device_result, sizeof(error), cudaMemcpyDeviceToHost),
                  "mixed result");
            if (error) {
                throw std::runtime_error("mixed channels accepted");
            }
        }
        uint64_t *device_generation = nullptr;
        check(cudaMalloc(&device_generation, 3 * sizeof(uint64_t)), "generation buffer");
        std::set<uint64_t> lane_addresses, qp_addresses;
        for (unsigned channel = 0; channel < qps; ++channel) {
            auto local_view = locals[channel], remote_view = remotes[channel];
            inspectLane<<<1, 1>>>(local_view, device_generation, false);
            check(cudaDeviceSynchronize(), "channel identities");
            uint64_t identity[3];
            check(cudaMemcpy(identity, device_generation, sizeof(identity), cudaMemcpyDeviceToHost),
                  "copy identities");
            if (!identity[1] || !identity[2] || !lane_addresses.insert(identity[1]).second ||
                !qp_addresses.insert(identity[2]).second) {
                throw std::runtime_error("pool channels alias the same lane or QP");
            }
#if NIXL_GPUNETIO_PUBLIC_FIFO_ONLY
            publicDefaultAdmission<<<1, 1>>>(local_view, remote_view, device_result);
            check(cudaGetLastError(), "launch public default admission");
            check(cudaDeviceSynchronize(), "complete public default admission");
            int admission = 0;
            check(cudaMemcpy(&admission, device_result, sizeof(admission), cudaMemcpyDeviceToHost),
                  "read public default admission");
            if (admission == 29) {
                std::cout << "SKIP native GET qualification: MCST consistency is unsupported\n"
                          << std::flush;
                std::_Exit(0);
            }
            if (admission) {
                throw std::runtime_error("public default lane admission failed");
            }
#endif
            check(cudaMemset(destination + kScalarDst, 0xa5, 3 * 256),
                  "poison scalar destinations");
            scalarGet<<<1, 1>>>(local_view, remote_view, channel, destination, device_result);
            check(cudaGetLastError(), "launch scalar GET");
            check(cudaDeviceSynchronize(), "complete scalar GET");
            int result = 0;
            check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
                  "read scalar status");
            if (result == 29) {
                std::cout << "SKIP native GET qualification: MCST consistency is unsupported\n"
                          << std::flush;
                std::_Exit(0);
            }
            if (result) {
                throw std::runtime_error("scalar GET failed: " + std::to_string(result));
            }
            // Cross the 2048-entry SQ/CQ wrap and repeatedly consume before kernel exit.
            for (unsigned round = 0; round < 130; ++round) {
                check(cudaMemset(destination + kBatchDst, 0xa5, kCount * kItemBytes),
                      "poison batch destination");
#if !defined(NIXL_GPUNETIO_PUBLIC_FIFO_ONLY) || !NIXL_GPUNETIO_PUBLIC_FIFO_ONLY
                batchGet<false><<<1, 32>>>(local_view, remote_view, destination, device_result);
                check(cudaGetLastError(), "launch parallel batch GET");
                check(cudaDeviceSynchronize(), "complete parallel batch GET");
                check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
                      "read parallel batch status");
                if (result == 29) {
                    std::cout << "SKIP native GET qualification: MCST consistency is unsupported\n"
                              << std::flush;
                    std::_Exit(0);
                }
                if (result) {
                    throw std::runtime_error("parallel batch GET failed: " +
                                             std::to_string(result));
                }
#endif
                check(cudaMemset(destination + kBatchDst, 0xa5, kCount * kItemBytes),
                      "re-poison FIFO batch destination");
                batchGet<true><<<1, 32>>>(local_view, remote_view, destination, device_result);
                check(cudaGetLastError(), "launch FIFO batch GET");
                check(cudaDeviceSynchronize(), "complete FIFO batch GET");
                check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
                      "read FIFO batch status");
                if (result == 29) {
                    std::cout << "SKIP native GET qualification: MCST consistency is unsupported\n"
                              << std::flush;
                    std::_Exit(0);
                }
                if (result) {
                    throw std::runtime_error("FIFO batch GET failed: " + std::to_string(result));
                }
            }
            uint64_t before_identity[3], after_identity[3], original_identity[3];
            inspectLane<<<1, 1>>>(local_view, device_generation, false);
            check(cudaDeviceSynchronize(), "read generation before");
            check(cudaMemcpy(before_identity,
                             device_generation,
                             sizeof(before_identity),
                             cudaMemcpyDeviceToHost),
                  "copy lane identity before");
            auto hint = reader_hint;
            hint.customParam = "native_channel=" + std::to_string(channel);
            nixlMemViewH repeated = nullptr;
            check(reader.prepMemView(remote, repeated, &hint), "reprepare used channel");
            inspectLane<<<1, 1>>>(repeated, device_generation, false);
            check(cudaDeviceSynchronize(), "read generation after re-preparation");
            check(cudaMemcpy(after_identity,
                             device_generation,
                             sizeof(after_identity),
                             cudaMemcpyDeviceToHost),
                  "copy re-prepared lane identity");
            inspectLane<<<1, 1>>>(local_view, device_generation, false);
            check(cudaDeviceSynchronize(), "read original lane identity");
            check(cudaMemcpy(original_identity,
                             device_generation,
                             sizeof(original_identity),
                             cudaMemcpyDeviceToHost),
                  "copy original lane identity");
            if (!before_identity[0] ||
                !std::equal(before_identity, before_identity + 3, after_identity) ||
                !std::equal(before_identity, before_identity + 3, original_identity)) {
                throw std::runtime_error("channel lane or generation changed during preparation");
            }
            reader.releaseMemView(repeated);
        }
        uint8_t received[kBytes];
        check(cudaMemcpy(received, destination, kBytes, cudaMemcpyDeviceToHost),
              "read destination");
        for (size_t i = 0; i < kBytes; ++i) {
            uint8_t expected = 0xa5;
            if (i >= kScalarDst && i < kScalarDst + 3 * 256) {
                expected = expected_source[17 + (i - kScalarDst) % 256];
            }
            if (i >= kBatchDst && i < kBatchDst + kCount * kItemBytes) {
                expected = expected_source[512 + i - kBatchDst];
            }
            if (i != kScratch && received[i] != expected) {
                throw std::runtime_error("data/guard mismatch at " + std::to_string(i));
            }
        }
        for (unsigned channel = 0; channel < qps; ++channel) {
            int result = 0;
            for (unsigned round = 0; round < 90; ++round) {
                check(cudaMemset(destination + kBatchDst, 0xa5, 24 * kItemBytes),
                      "poison pipeline destination");
                fifoPipeline<<<1, 1>>>(
                    locals[channel], remotes[channel], destination, device_result);
                check(cudaGetLastError(), "launch FIFO pipeline");
                check(cudaDeviceSynchronize(), "complete FIFO pipeline");
                check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
                      "read pipeline status");
                if (result == 30) {
                    std::cout << "SKIP FIFO pipeline: MCST consistency required\n";
                    break;
                }
                if (result) {
                    throw std::runtime_error("FIFO pipeline failed: " + std::to_string(result));
                }
            }
        }
        // Ordered graph nodes hold a lane across kernel boundaries; no block
        // waits for another block to be scheduled.
        PersistentRead *contention = nullptr;
        int *contention_result = nullptr;
        cudaStream_t contention_stream;
        cudaGraph_t contention_graph;
        cudaGraphExec_t contention_exec;
        check(cudaMalloc(&contention, sizeof(PersistentRead)), "allocate contention session");
        check(cudaMalloc(&contention_result, sizeof(int)), "allocate contention result");
        check(cudaStreamCreateWithFlags(&contention_stream, cudaStreamNonBlocking),
              "create contention stream");
        check(cudaStreamBeginCapture(contention_stream, cudaStreamCaptureModeGlobal),
              "begin contention capture");
        publishPersistent<<<1, 1, 0, contention_stream>>>(
            locals[0], remotes[0], kBatchDst, 512, contention);
        rejectBusyLane<<<1, 1, 0, contention_stream>>>(locals[0], remotes[0], contention_result);
        retirePersistent<<<1, 1, 0, contention_stream>>>(destination, kBatchDst, 512, contention);
        check(cudaGetLastError(), "capture ordered lane ownership nodes");
        check(cudaStreamEndCapture(contention_stream, &contention_graph), "end contention capture");
        check(cudaGraphInstantiate(&contention_exec, contention_graph, 0),
              "instantiate contention graph");
        for (unsigned replay = 0; replay < 10; ++replay) {
            check(cudaMemset(contention_result, 0xff, sizeof(int)), "reset contention result");
            check(cudaMemset(destination + kBatchDst, 0xa5, kItemBytes),
                  "poison contention destination");
            check(cudaDeviceSynchronize(), "complete contention resets");
            check(cudaGraphLaunch(contention_exec, contention_stream), "launch contention graph");
            check(cudaStreamSynchronize(contention_stream), "synchronize contention graph");
            int busy_result = 1;
            check(cudaMemcpy(&busy_result, contention_result, sizeof(int), cudaMemcpyDeviceToHost),
                  "read contention result");
            if (busy_result != 0) {
                throw std::runtime_error("held FIFO lane admitted a second session");
            }
            check(
                cudaMemcpy(&busy_result, &contention->result, sizeof(int), cudaMemcpyDeviceToHost),
                "read contention retirement result");
            if (busy_result != NIXL_SUCCESS) {
                throw std::runtime_error("ordered lane ownership replay failed");
            }
        }
        check(cudaGraphExecDestroy(contention_exec), "destroy contention executable");
        check(cudaGraphDestroy(contention_graph), "destroy contention graph");
        check(cudaStreamDestroy(contention_stream), "destroy contention stream");
        check(cudaFree(contention_result), "free contention result");
        check(cudaFree(contention), "free contention session");

        if (qps >= 2) {
            PersistentRead *parallel[2] = {nullptr, nullptr};
            cudaStream_t streams[2];
            cudaGraph_t publish_graphs[2], retire_graphs[2];
            cudaGraphExec_t publish_execs[2], retire_execs[2];
            cudaEvent_t published[2];
            for (unsigned i = 0; i < 2; ++i) {
                check(cudaMalloc(&parallel[i], sizeof(PersistentRead)),
                      "allocate parallel session");
                check(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking),
                      "create parallel Graph stream");
                check(cudaEventCreateWithFlags(&published[i], cudaEventDisableTiming),
                      "create publication event");
                check(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeGlobal),
                      "begin publication Graph capture");
                publishPersistent<<<1, 1, 0, streams[i]>>>(locals[i],
                                                           remotes[i],
                                                           kBatchDst + i * kItemBytes,
                                                           512 + i * kItemBytes,
                                                           parallel[i]);
                check(cudaGetLastError(), "capture lane publication");
                check(cudaStreamEndCapture(streams[i], &publish_graphs[i]),
                      "end publication Graph capture");
                check(cudaGraphInstantiate(&publish_execs[i], publish_graphs[i], 0),
                      "instantiate publication Graph");
                check(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeGlobal),
                      "begin retirement Graph capture");
                retirePersistent<<<1, 1, 0, streams[i]>>>(
                    destination, kBatchDst + i * kItemBytes, 512 + i * kItemBytes, parallel[i]);
                check(cudaGetLastError(), "capture lane retirement");
                check(cudaStreamEndCapture(streams[i], &retire_graphs[i]),
                      "end retirement Graph capture");
                check(cudaGraphInstantiate(&retire_execs[i], retire_graphs[i], 0),
                      "instantiate retirement Graph");
            }
            for (unsigned replay = 0; replay < 10; ++replay) {
                check(cudaMemset(destination + kBatchDst, 0xa5, 2 * kItemBytes),
                      "poison parallel Graph destinations");
                check(cudaDeviceSynchronize(), "complete parallel destination poison");
                for (unsigned i = 0; i < 2; ++i) {
                    check(cudaGraphLaunch(publish_execs[i], streams[i]),
                          "launch publication Graph");
                    check(cudaEventRecord(published[i], streams[i]), "record publication event");
                }
                for (unsigned i = 0; i < 2; ++i) {
                    check(cudaStreamWaitEvent(streams[i], published[1 - i], 0),
                          "wait for both lane publications");
                    check(cudaGraphLaunch(retire_execs[i], streams[i]), "launch retirement Graph");
                }
                for (auto stream : streams) {
                    check(cudaStreamSynchronize(stream), "synchronize parallel Graphs");
                }
                int results[2]{};
                for (unsigned i = 0; i < 2; ++i) {
                    check(
                        cudaMemcpy(
                            &results[i], &parallel[i]->result, sizeof(int), cudaMemcpyDeviceToHost),
                        "read parallel Graph result");
                }
                if (results[0] != NIXL_SUCCESS || results[1] != NIXL_SUCCESS) {
                    throw std::runtime_error("parallel FIFO Graph GET failed");
                }
            }
            for (unsigned i = 0; i < 2; ++i) {
                check(cudaGraphExecDestroy(publish_execs[i]), "destroy publication executable");
                check(cudaGraphExecDestroy(retire_execs[i]), "destroy retirement executable");
                check(cudaGraphDestroy(publish_graphs[i]), "destroy publication Graph");
                check(cudaGraphDestroy(retire_graphs[i]), "destroy retirement Graph");
                check(cudaEventDestroy(published[i]), "destroy publication event");
                check(cudaStreamDestroy(streams[i]), "destroy parallel Graph stream");
                check(cudaFree(parallel[i]), "free parallel session");
            }
        }
        // Run the standard GET checks after the legacy GET and Graph data oracles.
        auto checkStandardResult = [&]() {
            check(cudaDeviceSynchronize(), "complete standard GET test");
            int result = 0;
            check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
                  "read standard GET result");
            if (result) {
                throw std::runtime_error("standard GET test failed: " + std::to_string(result));
            }
        };
        for (unsigned channel = 0; channel < qps; ++channel) {
            check(cudaMemset(destination + 4096, 0xa5, 2048), "poison concurrent destinations");
            check(cudaMemset(device_result, 0, sizeof(int)), "reset standard result");
            concurrentStandardGets<<<1, 128>>>(
                locals[0], remotes[0], channel, destination, device_result);
            check(cudaGetLastError(), "launch concurrent standard GETs");
            checkStandardResult();
        }
        if (qps > 1) {
            check(cudaMemset(destination + kScalarDst, 0xa5, 3 * 256),
                  "poison mixed-view destinations");
            check(cudaMemset(device_result, 0, sizeof(int)), "reset mixed standard result");
            scalarGet<<<1, 1>>>(locals[0], remotes[1], 1, destination, device_result);
            check(cudaGetLastError(), "launch mixed legacy view standard GET");
            checkStandardResult();
        }
        check(cudaMemset(device_result, 0, sizeof(int)), "reset invalid result");
        standardInvalid<<<1, 1>>>(locals[0], remotes[0], qps, device_result);
        check(cudaGetLastError(), "launch invalid standard GETs");
        checkStandardResult();
        check(cudaMemset(device_result, 0, sizeof(int)), "reset collective result");
        check(cudaMemset(destination + 4096, 0xa5, 8 * 64), "poison warp destinations");
        collectiveStandardGet<nixl::gpu::level_t::WARP>
            <<<2, 128>>>(locals[0], remotes[0], destination, device_result);
        check(cudaGetLastError(), "launch warp standard GET");
        checkStandardResult();
        check(cudaMemset(device_result, 0, sizeof(int)), "reset block result");
        check(cudaMemset(destination + 4096, 0xa5, 2 * 64), "poison block destinations");
        collectiveStandardGet<nixl::gpu::level_t::BLOCK>
            <<<2, 128>>>(locals[0], remotes[0], destination, device_result);
        check(cudaGetLastError(), "launch block standard GET");
        checkStandardResult();

        check(cudaMemset(destination + 4096, 0xa5, 128), "poison null-status destinations");
        check(cudaMemset(device_result, 0, sizeof(int)), "reset null-status result");
        nullDeferredGet<<<1, 1>>>(locals[0], remotes[0], destination, device_result);
        check(cudaGetLastError(), "launch null-status deferred GETs");
        checkStandardResult();
        if (qps > 1) {
            check(cudaMemset(destination + 4096, 0xa5, 64),
                  "poison independent channel destination");
            check(cudaMemset(device_result, 0, sizeof(int)), "reset channel isolation result");
            channelIndependence<<<1, 1>>>(locals[0],
                                          remotes[0],
                                          locals.back(),
                                          remotes.back(),
                                          qps - 1,
                                          destination,
                                          device_result);
            check(cudaGetLastError(), "launch channel independence test");
            checkStandardResult();
        }

        auto verifyStandard = [&]() {
            uint8_t readback[4 * 64];
            check(
                cudaMemcpy(readback, destination + 4096, sizeof(readback), cudaMemcpyDeviceToHost),
                "read standard GET payloads");
            for (size_t i = 0; i < sizeof(readback); ++i) {
                if (readback[i] != expected_source[17 + i % 64]) {
                    throw std::runtime_error("standard GET payload mismatch");
                }
            }
        };
        for (unsigned channel = 0; channel < qps; ++channel) {
            check(cudaMemset(destination + 4096, 0xa5, 256), "poison standard GET destinations");
            check(cudaMemset(device_result, 0, sizeof(int)), "reset standard result");
            standardGets<<<1, 1>>>(locals[0], remotes[0], channel, device_result);
            check(cudaGetLastError(), "launch standard GET operations");
            checkStandardResult();
            verifyStandard();
        }
        std::vector<uint8_t> wrap_bytes(kBytes - kWrapDst);
        for (size_t i = 0; i < wrap_bytes.size(); ++i) {
            wrap_bytes[i] = expected_source[17 + i] ^ 0xff;
        }
        check(cudaMemcpy(destination + kWrapDst,
                         wrap_bytes.data(),
                         wrap_bytes.size(),
                         cudaMemcpyHostToDevice),
              "poison scalar wrap destinations");
        check(cudaMemset(device_result, 0, sizeof(int)), "reset scalar wrap result");
        scalarWrap<<<1, 1>>>(locals[0], remotes[0], destination, device_generation, device_result);
        check(cudaGetLastError(), "launch scalar FIFO wrap");
        checkStandardResult();
        uint64_t count = 0;
        check(cudaMemcpy(&count, device_generation, sizeof(count), cudaMemcpyDeviceToHost),
              "read scalar wrap count");
        check(cudaMemcpy(wrap_bytes.data(),
                         destination + kWrapDst,
                         wrap_bytes.size(),
                         cudaMemcpyDeviceToHost),
              "read scalar wrap bytes");
        for (size_t i = 0; i < wrap_bytes.size(); ++i) {
            const uint8_t expected =
                i < count ? expected_source[17 + i] : expected_source[17 + i] ^ 0xff;
            if (wrap_bytes[i] != expected) {
                throw std::runtime_error("scalar FIFO wrap payload mismatch");
            }
        }
        cudaStream_t standard_stream;
        cudaGraph_t standard_graph;
        cudaGraphExec_t standard_exec;
        check(cudaStreamCreateWithFlags(&standard_stream, cudaStreamNonBlocking),
              "create standard stream");
        check(cudaStreamBeginCapture(standard_stream, cudaStreamCaptureModeGlobal),
              "capture standard GET");
        standardGets<<<1, 1, 0, standard_stream>>>(locals[0], remotes[0], 0, device_result);
        check(cudaGetLastError(), "capture standard GETs");
        check(cudaStreamEndCapture(standard_stream, &standard_graph), "end standard capture");
        check(cudaGraphInstantiate(&standard_exec, standard_graph, 0),
              "instantiate standard graph");
        for (unsigned replay = 0; replay < 10; ++replay) {
            check(cudaMemset(destination + 4096, 0xa5, 256), "poison standard graph destinations");
            check(cudaMemset(device_result, 0, sizeof(int)), "reset standard graph result");
            check(cudaGraphLaunch(standard_exec, standard_stream), "launch standard graph");
            check(cudaStreamSynchronize(standard_stream), "complete standard graph");
            checkStandardResult();
            verifyStandard();
        }
        check(cudaGraphExecDestroy(standard_exec), "destroy standard graph executable");
        check(cudaGraphDestroy(standard_graph), "destroy standard graph");
        check(cudaStreamDestroy(standard_stream), "destroy standard stream");
        if (large_test) {
            check(cudaSetDevice(owner_gpu), "select large owner GPU");
            largeFill<<<256, 256>>>(large_owner, large_bytes, 7, 0xc3);
            check(cudaGetLastError(), "fill large owner");
            check(cudaDeviceSynchronize(), "complete large owner fill");
            check(cudaSetDevice(reader_gpu), "select large reader GPU");
            largeFill<<<256, 256>>>(large_reader, large_bytes, 19, 0xd7);
            check(cudaGetLastError(), "fill large reader");
            check(cudaDeviceSynchronize(), "complete large reader fill");
            check(cudaMemset(device_result, 0, sizeof(int)), "reset large GET result");
            largeTransfer<<<1, 1>>>(
                large_local_view, large_remote_view, large_bytes, device_result);
            check(cudaGetLastError(), "launch large GET");
            checkStandardResult();
            largeVerify<<<256, 256>>>(large_reader, large_bytes, 7, 0xd7, device_result);
            check(cudaGetLastError(), "verify large GET");
            checkStandardResult();
            std::cout << "PASS large GET bytes=" << large_bytes << " with guards\n";
            reader.releaseMemView(large_remote_view);
            reader.releaseMemView(large_local_view);
            check(reader.deregisterMem(large_reader_reg, &reader_hint), "deregister large reader");
            check(cudaFree(large_reader), "free large reader");
            check(cudaSetDevice(owner_gpu), "select owner to free large buffers");
            check(owner.deregisterMem(large_owner_reg, &owner_hint), "deregister large owner");
            check(cudaFree(large_owner), "free large owner");
            check(cudaSetDevice(reader_gpu), "restore reader after large cleanup");
        }
        if (std::getenv("NIXL_GPUNETIO_TEST_RETENTION")) {
            if (qps < 2) {
                throw std::runtime_error("retention test needs multiple channels");
            }
            const auto failed = locals.back();
            uint64_t before = 0, after = 0;
            const bool real_error = std::getenv("NIXL_GPUNETIO_TEST_CQ_ERROR") != nullptr;
            if (real_error) {
                if (std::getenv("NIXL_GPUNETIO_TEST_STANDARD_CQ_ERROR")) {
                    expectStandardFifoCqError<<<1, 1>>>(
                        failed, remotes.back(), qps - 1, device_result);
                } else if (std::getenv("NIXL_GPUNETIO_TEST_FIFO_CQ_ERROR")) {
                    expectFifoCqError<<<1, 1>>>(failed, remotes.back(), device_result);
                } else {
                    corruptRemoteKey<<<1, 1>>>(remotes.back());
                    expectCqError<<<1, 1>>>(failed, remotes.back(), qps - 1, device_result, false);
                }
                check(cudaDeviceSynchronize(), "real CQ failure");
                int error = 0;
                check(cudaMemcpy(&error, device_result, sizeof(error), cudaMemcpyDeviceToHost),
                      "CQ failure result");
                if (error) {
                    throw std::runtime_error("CQ error not detected");
                }
                expectCqError<<<1, 1>>>(failed, remotes.back(), qps - 1, device_result, true);
                check(cudaDeviceSynchronize(), "sticky CQ failure");
                check(cudaMemcpy(&error, device_result, sizeof(error), cudaMemcpyDeviceToHost),
                      "sticky result");
                if (error) {
                    throw std::runtime_error("failed channel reused");
                }
                check(cudaMemset(destination + kScalarDst, 0xa5, 3 * 256),
                      "poison healthy scalar destination");
                scalarGet<<<1, 1>>>(locals[0], remotes[0], 0, destination, device_result);
                check(cudaDeviceSynchronize(), "healthy channel after peer channel error");
                check(cudaMemcpy(&error, device_result, sizeof(error), cudaMemcpyDeviceToHost),
                      "healthy result");
                if (error) {
                    throw std::runtime_error("failure leaked to independent channel");
                }
            }
            inspectLane<<<1, 1>>>(failed, device_generation, !real_error);
            check(cudaDeviceSynchronize(), "poison idle last channel");
            check(cudaMemcpy(&before, device_generation, sizeof(before), cudaMemcpyDeviceToHost),
                  "poison generation");
            reader.releaseMemView(remotes[0]);
            reader.releaseMemView(locals[0]);
            reader.releaseMemView(remotes.back());
            reader.releaseMemView(failed);
            inspectLane<<<1, 1>>>(failed, device_generation, false);
            check(cudaDeviceSynchronize(), "retained channel memory");
            check(cudaMemcpy(&after, device_generation, sizeof(after), cudaMemcpyDeviceToHost),
                  "retained generation");
            if (before != after) {
                throw std::runtime_error("failed channel view was not retained");
            }
            if (real_error) {
                std::cout << "PASS real CQ error, sticky refusal, independent healthy channel\n";
            }
            std::cout << "PASS failed-channel retention; "
                         "QUARANTINED_UNTIL_PROCESS_EXIT\n"
                      << std::flush;
            std::_Exit(0);
        }
        for (auto view : remotes) {
            reader.releaseMemView(view);
        }
        for (auto view : locals) {
            reader.releaseMemView(view);
        }
        check(cudaFree(device_generation), "free generation buffer");
        check(reader.deregisterMem(reader_reg, &reader_hint), "deregister destination");
        check(cudaFree(device_result), "free status");
        check(cudaFree(destination), "free destination");
        check(cudaSetDevice(owner_gpu), "select owner GPU");
        check(owner.deregisterMem(owner_reg, &owner_hint), "deregister source");
        if (source_is_host) {
            check(cudaFreeHost(source), "free source");
        } else {
            check(cudaFree(source), "free source");
        }
        std::cout << "PASS " << source_type << " qps=" << qps
                  << " scalar and CTA batch GET; data, guards and completion validated\n";
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
