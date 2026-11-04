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
scalarGet(nixlMemViewH local, nixlMemViewH remote, int *result) {
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
    }
}

__global__ void
batchGet(nixlMemViewH local, nixlMemViewH remote, int *result) {
    __shared__ __align__(16) unsigned char storage[sizeof(nixl::gpu::readSession)];
    __shared__ uint64_t first;
    __shared__ int error;
    auto *session = reinterpret_cast<nixl::gpu::readSession *>(storage);
    if (!threadIdx.x) {
        ::new (session) nixl::gpu::readSession{};
        error = 0;
        if (session->bind(local, remote) != NIXL_SUCCESS || session->acquire() != NIXL_SUCCESS) {
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
    if (!error && threadIdx.x < kCount) {
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
        if (!error && session->retire(first, kCount) != NIXL_SUCCESS) {
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
                    rc = session->poll(ticket);
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
                                   {"gpu_devices", std::to_string(device)},
                                   {"network_devices", argv[1]},
                                   {"oob_interface", argv[2]},
                                   {"gid_index", argv[3]},
                                   {"oob_port", std::to_string(value)}};
        };
        auto owner_params = params(port, owner_gpu), reader_params = params(port + 1, reader_gpu);
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
        check(reader.makeConnection(peer, &reader_hint), "connect");
        nixl_local_dlist_t local(VRAM_SEG);
        local.addDesc(
            {reinterpret_cast<uintptr_t>(destination), kBytes, static_cast<uint64_t>(reader_gpu)});
        nixl_remote_dlist_t remote(source_mem);
        remote.addDesc(
            {reinterpret_cast<uintptr_t>(source), kBytes, static_cast<uint64_t>(owner_gpu), peer});
        nixlMemViewH local_view = nullptr, remote_view = nullptr;
        check(reader.prepMemView(local, local_view, &reader_hint), "prepare local view");
        check(reader.prepMemView(remote, remote_view, &reader_hint), "prepare remote view");

        scalarGet<<<1, 1>>>(local_view, remote_view, device_result);
        check(cudaGetLastError(), "launch scalar GET");
        check(cudaDeviceSynchronize(), "complete scalar GET");
        int result = 0;
        check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
              "read scalar status");
        if (result) {
            throw std::runtime_error("scalar GET failed: " + std::to_string(result));
        }
        batchGet<<<1, 32>>>(local_view, remote_view, device_result);
        check(cudaGetLastError(), "launch batch GET");
        check(cudaDeviceSynchronize(), "complete batch GET");
        check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
              "read batch status");
        if (result) {
            throw std::runtime_error("batch GET failed: " + std::to_string(result));
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
        reader.releaseMemView(remote_view);
        reader.releaseMemView(local_view);
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
        std::cout << "PASS " << source_type
                  << " scalar and CTA batch GET; data, guards and completion validated\n";
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
