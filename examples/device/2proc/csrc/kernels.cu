/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernels.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "nixl_device.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void post_write_and_signal_kernel_thread(uintptr_t data_req_handles_ptr,
                                                    int data_req_count,
                                                    uintptr_t signal_req_handle,
                                                    uint64_t *signal_ptr,
                                                    size_t size,
                                                    uint64_t signal_inc,
                                                    int pipeline) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (data_req_count <= 0) {
            return;
        }
        auto *data_req_handles =
            reinterpret_cast<const uintptr_t *>(data_req_handles_ptr);
        nixlGpuXferReqH data_req =
            reinterpret_cast<nixlGpuXferReqH>(data_req_handles[0]);
        nixlGpuXferReqH signal_req =
            reinterpret_cast<nixlGpuXferReqH>(signal_req_handle);
        for (int i = 0; i < pipeline; ++i) {
            nixlGpuPostSingleWriteXferReq<nixl_gpu_level_t::THREAD>(
                data_req, 0, 0, 0, size, 0, true);
        }
        nixlGpuPostSignalXferReq<nixl_gpu_level_t::THREAD>(
            signal_req, 0, signal_inc, 0, 0, true);
    }
}

__global__ void post_write_and_signal_kernel_warp(uintptr_t data_req_handles_ptr,
                                                  int data_req_count,
                                                  uintptr_t signal_req_handle,
                                                  uint64_t *signal_ptr,
                                                  size_t size,
                                                  uint64_t signal_inc,
                                                  int pipeline) {
    if (blockIdx.x == 0 && threadIdx.x < WARP_SIZE) {
        if (data_req_count <= 0) {
            return;
        }
        auto *data_req_handles =
            reinterpret_cast<const uintptr_t *>(data_req_handles_ptr);
        unsigned lane = static_cast<unsigned>(threadIdx.x);
        unsigned handle_idx = (data_req_count == WARP_SIZE) ? lane : 0;
        nixlGpuXferReqH data_req =
            reinterpret_cast<nixlGpuXferReqH>(data_req_handles[handle_idx]);
        nixlGpuXferReqH signal_req =
            reinterpret_cast<nixlGpuXferReqH>(signal_req_handle);
        for (int i = 0; i < pipeline; ++i) {
            nixlGpuPostSingleWriteXferReq<nixl_gpu_level_t::WARP>(
                data_req, 0, 0, 0, size, 0, true);
        }
        if (lane == 0) {
            nixlGpuPostSignalXferReq<nixl_gpu_level_t::WARP>(
                signal_req, 0, signal_inc, 0, 0, true);
        }
    }
}

__global__ void wait_for_signal_kernel_thread(const void* signal_ptr,
                                              uint64_t expected_value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (nixlGpuReadSignal<nixl_gpu_level_t::THREAD>(signal_ptr) < expected_value) {
            // Busy-wait on GPU signal
        }
    }
}

void launch_post_write_and_signal(uintptr_t data_req_handles_ptr,
                                  int data_req_count,
                                  uintptr_t signal_req_handle,
                                  uintptr_t signal_ptr,
                                  size_t size,
                                  int level,
                                  int threads_per_block,
                                  int pipeline,
                                  cudaStream_t stream) {
    uint64_t *signal = reinterpret_cast<uint64_t *>(signal_ptr);
    int threads = (threads_per_block > 0
                       ? threads_per_block
                       : (level == 1 ? WARP_SIZE : 1));
    if (level == 1) {
        post_write_and_signal_kernel_warp<<<1, threads, 0, stream>>>(
            data_req_handles_ptr, data_req_count, signal_req_handle, signal, size, 1, pipeline);
    } else {
        post_write_and_signal_kernel_thread<<<1, threads, 0, stream>>>(
            data_req_handles_ptr, data_req_count, signal_req_handle, signal, size, 1, pipeline);
    }
}

void launch_wait_for_signal(uintptr_t signal_ptr,
                            uint64_t expected_value,
                            int level,
                            int threads_per_block,
                            cudaStream_t stream) {
    const void *signal = reinterpret_cast<const void*>(signal_ptr);
    (void)level;
    (void)threads_per_block;
    wait_for_signal_kernel_thread<<<1, 1, 0, stream>>>(
        signal, expected_value);
}

