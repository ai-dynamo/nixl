/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda/atomic>
#include <cuda_runtime.h>
#include "payload.h"

namespace {
constexpr unsigned layers = 27, rows = 64, width = 576, peers = 4, slots = 3;

__global__ void
fill(uint8_t *source, uint64_t *marker, uint64_t epoch) {
    constexpr unsigned size = layers * slots * rows * peers * width;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        unsigned row = i / width;
        const unsigned peer = row % peers;
        row /= peers;
        const unsigned slot = (row / rows) % slots;
        source[i] = payloadByte(epoch, peer, row / (rows * slots), row % rows, slot, i % width);
    }
    if (blockIdx.x == 0 && threadIdx.x < peers) {
        marker[threadIdx.x] = epoch;
    }
}

__global__ void
validate(const uint64_t *marker,
         const uint8_t *data,
         unsigned peer,
         uint64_t epoch,
         uint64_t e0,
         uint64_t e1,
         uint64_t e2,
         unsigned validRows,
         unsigned validLayers,
         unsigned *errors) {
    __shared__ bool ready;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> flag(*const_cast<uint64_t *>(marker));
        ready = flag.load(cuda::memory_order_acquire) == epoch;
        if (!ready) {
            atomicAdd(errors, 1u);
        }
    }
    __syncthreads();
    if (!ready) {
        return;
    }
    const uint64_t epochs[slots] = {e0, e1, e2};
    constexpr unsigned size = layers * slots * rows * width;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        const unsigned row = (i / width) % (slots * rows);
        const uint64_t written = epochs[row / rows];
        const auto expected =
            written == 0 || row % rows >= validRows || i / (slots * rows * width) >= validLayers ?
            uint8_t(0xa5) :
            payloadByte(written,
                        peer,
                        i / (slots * rows * width),
                        row % rows,
                        (written - 1) % slots,
                        i % width);
        if (data[i] != expected) {
            atomicAdd(errors, 1u);
        }
    }
}
} // namespace

extern "C" int
fillPayload(void *source, void *marker, uint64_t epoch) {
    cudaStream_t stream = nullptr;
    auto s = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (s != cudaSuccess) {
        return int(s);
    }
    fill<<<64, 256, 0, stream>>>(
        static_cast<uint8_t *>(source), static_cast<uint64_t *>(marker), epoch);
    s = cudaGetLastError();
    if (s == cudaSuccess) {
        s = cudaStreamSynchronize(stream);
    }
    cudaStreamDestroy(stream);
    return int(s);
}

extern "C" int
validatePayload(const void *marker,
                const void *data,
                unsigned peer,
                uint64_t epoch,
                const uint64_t *epochs,
                unsigned validRows,
                unsigned validLayers,
                unsigned *errors,
                unsigned *hostErrors) {
    cudaStream_t stream = nullptr;
    auto s = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (s != cudaSuccess) {
        return int(s);
    }
    s = cudaMemsetAsync(errors, 0, sizeof(unsigned), stream);
    if (s != cudaSuccess) {
        cudaStreamDestroy(stream);
        return int(s);
    }
    validate<<<32, 256, 0, stream>>>(static_cast<const uint64_t *>(marker),
                                     static_cast<const uint8_t *>(data),
                                     peer,
                                     epoch,
                                     epochs[0],
                                     epochs[1],
                                     epochs[2],
                                     validRows,
                                     validLayers,
                                     errors);
    s = cudaGetLastError();
    if (s == cudaSuccess) {
        s = cudaMemcpyAsync(hostErrors, errors, sizeof(unsigned), cudaMemcpyDeviceToHost, stream);
    }
    if (s == cudaSuccess) {
        s = cudaStreamSynchronize(stream);
    }
    cudaStreamDestroy(stream);
    return int(s);
}
