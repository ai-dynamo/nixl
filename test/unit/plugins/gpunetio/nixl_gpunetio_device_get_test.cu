/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime.h>
#include <gpu/nixl_device.cuh>
#include <gpu/nixl_device_batch.cuh>
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
    nixl::gpu::readSession session;
    nixl::gpu::xferStatusH status{};
    if (session.bind(local, remote) != NIXL_ERR_INVALID_PARAM ||
        nixl::gpu::get(
            {remote, 0, 17}, {local, 0, kScalarDst}, {local, 0, kScratch}, 256, &status) !=
            NIXL_ERR_INVALID_PARAM) {
        *result = 16;
    }
}

__global__ void
scalarGet(nixlMemViewH local, nixlMemViewH remote, uint8_t *destination, int *result) {
    if (threadIdx.x) {
        return;
    }
    nixl::gpu::xferStatusH status{};
    if (nixl::gpu::get(
            {remote, 0, kBytes - 10}, {local, 0, kScalarDst}, {local, 0, kScratch}, 256, &status) !=
            NIXL_ERR_INVALID_PARAM ||
        nixl::gpu::get(
            {remote, 0, 17}, {local, 0, kScalarDst}, {local, 0, kScalarDst}, 256, &status) !=
            NIXL_ERR_INVALID_PARAM) {
        *result = 11;
        return;
    }
    auto rc =
        nixl::gpu::get({remote, 0, 17}, {local, 0, kScalarDst}, {local, 0, kScratch}, 256, &status);
    if (rc != NIXL_IN_PROG) {
        *result = 1;
        return;
    }
    const uint64_t deadline = globalTimerNs() + 5000000000ull;
    do {
        rc = nixl::gpu::getXferStatus(status);
    } while (rc == NIXL_IN_PROG && globalTimerNs() < deadline);
    if (rc != NIXL_SUCCESS) {
        *result = 2;
    } else {
        // Consume immediately on the GPU, before a kernel boundary or CPU sync.
        for (size_t i = 0; i < 256; ++i) {
            if (destination[kScalarDst + i] != uint8_t(((17 + i) * 13 + 7) % 251)) {
                *result = 14;
                break;
            }
        }
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
expectCqError(nixlMemViewH local, nixlMemViewH remote, int *result, bool sticky) {
    nixl::gpu::xferStatusH status{};
    auto rc =
        nixl::gpu::get({remote, 0, 17}, {local, 0, kScalarDst}, {local, 0, kScratch}, 256, &status);
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

// Disposable-process FIFO fault check. Do not release the failed lane or MRs.
__global__ void
expectFifoCqError(nixlMemViewH local, nixlMemViewH remote, int *result) {
    using namespace nixl::gpu::impl::gpunetio;
    nixl::gpu::readSession session;
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
    const bool sticky = rc == NIXL_ERR_BACKEND && session.pollFifo(first + 2) == NIXL_ERR_BACKEND &&
        session.retire(first, 1) == NIXL_ERR_BACKEND && session.finish() == NIXL_ERR_BACKEND;
    session.fail();
    *result = sticky ? 0 : 23;
}

template<bool Fifo>
__global__ void
batchGet(nixlMemViewH local, nixlMemViewH remote, uint8_t *destination, int *result) {
    __shared__ __align__(16) unsigned char storage[sizeof(nixl::gpu::readSession)];
    __shared__ uint64_t first;
    __shared__ int error;
    auto *session = reinterpret_cast<nixl::gpu::readSession *>(storage);
    if (!threadIdx.x) {
        ::new (session) nixl::gpu::readSession{};
        error = 0;
        if (session->bind(local, remote) != NIXL_SUCCESS ||
            (Fifo ? session->acquireFifo() : session->acquire()) != NIXL_SUCCESS) {
            error = 3;
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
    if (!error && threadIdx.x < kCount &&
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
    } else if (!error && threadIdx.x < kCount) {
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
    nixl::gpu::readSession session;
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

// Two blocks contend for one lane. The winner holds it until both have attempted.
__global__ void
contendFifoLane(nixlMemViewH local, nixlMemViewH remote, const uint8_t *destination, int *state) {
    if (threadIdx.x) {
        return;
    }
    nixl::gpu::readSession session;
    if (session.bind(local, remote) != NIXL_SUCCESS) {
        atomicAdd(state + 3, 1);
        atomicAdd(state, 1);
        return;
    }
    const auto rc = session.acquireFifo();
    if (rc != NIXL_SUCCESS) {
        atomicAdd(state + (rc == NIXL_ERR_NOT_ALLOWED ? 2 : 3), 1);
        atomicAdd(state, 1);
        return;
    }
    atomicAdd(state + 1, 1);
    uint64_t first = 0;
    bool ok = session.reserve(1, first) == NIXL_SUCCESS &&
        session.prepare(first, 0, 512, 0, kBatchDst, kItemBytes) == NIXL_SUCCESS &&
        session.publish(first, 1) == NIXL_IN_PROG;
    atomicAdd(state, 1);
    const uint64_t deadline = globalTimerNs() + 5000000000ull;
    while (atomicAdd(state, 0) < 2 && globalTimerNs() < deadline) {}
    if (atomicAdd(state, 0) != 2) {
        ok = false;
    }
    auto wait = [&](uint64_t ticket) {
        nixl_status_t status;
        do {
            status = session.pollFifo(ticket);
        } while (status == NIXL_IN_PROG && globalTimerNs() < deadline);
        return status == NIXL_SUCCESS && session.retire(ticket, 1) == NIXL_SUCCESS;
    };
    if (ok) {
        ok = wait(first);
    }
    if (ok && session.needsConsistency()) {
        uint64_t dump;
        ok = session.postConsistency(0, kScratch, dump) == NIXL_IN_PROG && wait(dump);
    }
    if (ok) {
        ok = session.finish() == NIXL_SUCCESS;
    }
    if (!ok) {
        session.fail();
        atomicAdd(state + 3, 1);
        return;
    }
    for (size_t i = 0; i < kItemBytes; ++i) {
        if (destination[kBatchDst + i] != uint8_t(((512 + i) * 13 + 7) % 251)) {
            atomicAdd(state + 3, 1);
            break;
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
        check(cudaSetDevice(reader_gpu), "select reader GPU");
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
        const char *peer_qps = std::getenv("NIXL_GPUNETIO_TEST_PEER_QPS");
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
        owner_reg.addDesc(
            {reinterpret_cast<uintptr_t>(source), kBytes, static_cast<uint64_t>(owner_gpu), {}});
        reader_reg.addDesc({reinterpret_cast<uintptr_t>(destination),
                            kBytes,
                            static_cast<uint64_t>(reader_gpu),
                            {}});
        check(cudaSetDevice(owner_gpu), "select owner GPU");
        check(owner.registerMem(owner_reg, &owner_hint), "register source");
        check(cudaSetDevice(reader_gpu), "select reader GPU");
        check(reader.registerMem(reader_reg, &reader_hint), "register destination");
        nixl_blob_t metadata;
        check(owner.getLocalMD(metadata), "export metadata");
        std::string peer;
        if (peer_qps) {
            auto status = reader.loadRemoteMD(metadata, peer);
            if (status == NIXL_SUCCESS) {
                status = reader.makeConnection(peer, &reader_hint);
            }
            if (status == NIXL_SUCCESS) {
                throw std::runtime_error("mismatched pool connected");
            }
            std::cout << "PASS mismatched native QP counts rejected; "
                         "SETUP_RESOURCES_RETAINED_UNTIL_EXIT\n"
                      << std::flush;
            std::_Exit(0);
        }
        // A failed section load invalidates the core's connection record.
        // Reloading the same live peer must still allow a GET.
        nixl_blob_t malformed = metadata;
        const auto tag = malformed.find("nixlSecElms");
        if (tag == nixl_blob_t::npos) {
            throw std::runtime_error("metadata section tag missing");
        }
        malformed[tag] = 'X';
        if (reader.loadRemoteMD(malformed, peer) != NIXL_ERR_MISMATCH) {
            throw std::runtime_error("malformed metadata was not rejected");
        }
        peer.clear();
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
            scalarGet<<<1, 1>>>(local_view, remote_view, destination, device_result);
            check(cudaGetLastError(), "launch scalar GET");
            check(cudaDeviceSynchronize(), "complete scalar GET");
            int result = 0;
            check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
                  "read scalar status");
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
                if (result) {
                    throw std::runtime_error("parallel batch GET failed: " +
                                             std::to_string(result));
                }
#endif
                batchGet<true><<<1, 32>>>(local_view, remote_view, destination, device_result);
                check(cudaGetLastError(), "launch FIFO batch GET");
                check(cudaDeviceSynchronize(), "complete FIFO batch GET");
                check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
                      "read FIFO batch status");
                if (result) {
                    throw std::runtime_error("FIFO batch GET failed: " + std::to_string(result));
                }
            }
            uint64_t before = 0, after = 0;
            inspectLane<<<1, 1>>>(local_view, device_generation, false);
            check(cudaDeviceSynchronize(), "read generation before");
            check(cudaMemcpy(&before, device_generation, sizeof(before), cudaMemcpyDeviceToHost),
                  "copy generation before");
            check(reader.loadRemoteMD(metadata, peer), "same-live metadata reload after use");
            auto hint = reader_hint;
            hint.customParam = "native_channel=" + std::to_string(channel);
            nixlMemViewH repeated = nullptr;
            check(reader.prepMemView(remote, repeated, &hint), "reprepare used channel");
            inspectLane<<<1, 1>>>(local_view, device_generation, false);
            check(cudaDeviceSynchronize(), "read generation after");
            check(cudaMemcpy(&after, device_generation, sizeof(after), cudaMemcpyDeviceToHost),
                  "copy generation after");
            if (!before || before != after) {
                throw std::runtime_error("channel generation reset during preparation");
            }
            reader.releaseMemView(repeated);
        }
        uint8_t received[kBytes];
        check(cudaMemcpy(received, destination, kBytes, cudaMemcpyDeviceToHost),
              "read destination");
        for (size_t i = 0; i < kBytes; ++i) {
            uint8_t expected = 0xa5;
            if (i >= kScalarDst && i < kScalarDst + 256) {
                expected = expected_source[17 + i - kScalarDst];
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
        // Capture competing CTAs and replay after resetting status in the graph.
        int *contention = nullptr;
        cudaStream_t contention_stream;
        cudaGraph_t contention_graph;
        cudaGraphExec_t contention_exec;
        check(cudaMalloc(&contention, 4 * sizeof(int)), "allocate contention state");
        check(cudaStreamCreateWithFlags(&contention_stream, cudaStreamNonBlocking),
              "create contention stream");
        check(cudaStreamBeginCapture(contention_stream, cudaStreamCaptureModeGlobal),
              "begin contention capture");
        check(cudaMemsetAsync(contention, 0, 4 * sizeof(int), contention_stream),
              "reset contention state");
        contendFifoLane<<<2, 1, 0, contention_stream>>>(
            locals[0], remotes[0], destination, contention);
        check(cudaGetLastError(), "launch competing CTA GET");
        check(cudaStreamEndCapture(contention_stream, &contention_graph), "end contention capture");
        check(cudaGraphInstantiate(&contention_exec, contention_graph, 0),
              "instantiate contention graph");
        for (unsigned replay = 0; replay < 10; ++replay) {
            check(cudaGraphLaunch(contention_exec, contention_stream), "launch contention graph");
            check(cudaStreamSynchronize(contention_stream), "synchronize contention graph");
            int counts[4];
            check(cudaMemcpy(counts, contention, sizeof(counts), cudaMemcpyDeviceToHost),
                  "read contention state");
            if (counts[0] != 2 || counts[1] != 1 || counts[2] != 1 || counts[3] != 0) {
                throw std::runtime_error("concurrent FIFO lane ownership failed");
            }
        }
        check(cudaGraphExecDestroy(contention_exec), "destroy contention executable");
        check(cudaGraphDestroy(contention_graph), "destroy contention graph");
        check(cudaStreamDestroy(contention_stream), "destroy contention stream");
        check(cudaFree(contention), "free contention state");
        if (std::getenv("NIXL_GPUNETIO_TEST_RETENTION")) {
            if (qps < 2) {
                throw std::runtime_error("retention test needs multiple channels");
            }
            const auto failed = locals.back();
            uint64_t before = 0, after = 0;
            const bool real_error = std::getenv("NIXL_GPUNETIO_TEST_CQ_ERROR") != nullptr;
            if (real_error) {
                if (std::getenv("NIXL_GPUNETIO_TEST_FIFO_CQ_ERROR")) {
                    expectFifoCqError<<<1, 1>>>(failed, remotes.back(), device_result);
                } else {
                    corruptRemoteKey<<<1, 1>>>(remotes.back());
                    expectCqError<<<1, 1>>>(failed, remotes.back(), device_result, false);
                }
                check(cudaDeviceSynchronize(), "real CQ failure");
                int error = 0;
                check(cudaMemcpy(&error, device_result, sizeof(error), cudaMemcpyDeviceToHost),
                      "CQ failure result");
                if (error) {
                    throw std::runtime_error("CQ error not detected");
                }
                expectCqError<<<1, 1>>>(failed, remotes.back(), device_result, true);
                check(cudaDeviceSynchronize(), "sticky CQ failure");
                check(cudaMemcpy(&error, device_result, sizeof(error), cudaMemcpyDeviceToHost),
                      "sticky result");
                if (error) {
                    throw std::runtime_error("failed channel reused");
                }
                scalarGet<<<1, 1>>>(locals[0], remotes[0], destination, device_result);
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
            std::cout << "PASS pool idle release and failed-channel retention; "
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
