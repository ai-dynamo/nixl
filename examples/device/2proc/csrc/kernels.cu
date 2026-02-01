/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "nixl_device.cuh"
#include "kernels.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void
post_write_and_signal_kernel_thread(const nixlMemDesc *src_descs,
                                    const nixlMemDesc *dst_descs,
                                    int desc_count,
                                    nixlMemDesc signal_desc,
                                    uint64_t *signal_ptr,
                                    size_t size,
                                    uint64_t signal_inc,
                                    int pipeline) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (desc_count <= 0) {
            return;
        }

        // Post pipelined writes
        for (int i = 0; i < pipeline; ++i) {
            nixlPut<nixl_gpu_level_t::THREAD>(
                src_descs[0], dst_descs[0], size, 0,
                static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr);
        }

        // Signal completion
        nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
            signal_inc, signal_desc, 0,
            static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr);
    }
}

__global__ void
post_write_and_signal_kernel_warp(const nixlMemDesc *src_descs,
                                  const nixlMemDesc *dst_descs,
                                  int desc_count,
                                  nixlMemDesc signal_desc,
                                  uint64_t *signal_ptr,
                                  size_t size,
                                  uint64_t signal_inc,
                                  int pipeline) {
    if (blockIdx.x == 0 && threadIdx.x < WARP_SIZE) {
        if (desc_count <= 0) {
            return;
        }

        auto lane = static_cast<unsigned>(threadIdx.x);
        unsigned desc_idx = (desc_count == WARP_SIZE) ? lane : 0;

        // Post pipelined writes
        for (int i = 0; i < pipeline; ++i) {
            nixlPut<nixl_gpu_level_t::WARP>(
                src_descs[desc_idx], dst_descs[desc_idx], size, 0,
                static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr);
        }

        // Only lane 0 signals completion
        if (lane == 0) {
            nixlAtomicAdd<nixl_gpu_level_t::WARP>(
                signal_inc, signal_desc, 0,
                static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr);
        }
    }
}

__global__ void
wait_for_signal_kernel_thread(const void *signal_ptr, uint64_t expected_value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (nixlGpuReadSignal<nixl_gpu_level_t::THREAD>(signal_ptr) < expected_value) {
            // Busy-wait on GPU signal
        }
    }
}

void
launch_post_write_and_signal(uintptr_t src_descs_ptr,
                             uintptr_t dst_descs_ptr,
                             int desc_count,
                             nixlMemDesc signal_desc,
                             uintptr_t signal_ptr,
                             size_t size,
                             int level,
                             int threads_per_block,
                             int pipeline,
                             cudaStream_t stream) {
    auto signal = reinterpret_cast<uint64_t *>(signal_ptr);
    auto src_descs = reinterpret_cast<const nixlMemDesc *>(src_descs_ptr);
    auto dst_descs = reinterpret_cast<const nixlMemDesc *>(dst_descs_ptr);
    int threads = (threads_per_block > 0 ? threads_per_block : (level == 1 ? WARP_SIZE : 1));

    if (level == 1) {
        post_write_and_signal_kernel_warp<<<1, threads, 0, stream>>>(
            src_descs, dst_descs, desc_count, signal_desc, signal, size, 1, pipeline);
    } else {
        post_write_and_signal_kernel_thread<<<1, threads, 0, stream>>>(
            src_descs, dst_descs, desc_count, signal_desc, signal, size, 1, pipeline);
    }
}

void
launch_wait_for_signal(uintptr_t signal_ptr,
                       uint64_t expected_value,
                       int level,
                       int threads_per_block,
                       cudaStream_t stream) {
    auto signal = reinterpret_cast<const void *>(signal_ptr);
    wait_for_signal_kernel_thread<<<1, 1, 0, stream>>>(signal, expected_value);
}
