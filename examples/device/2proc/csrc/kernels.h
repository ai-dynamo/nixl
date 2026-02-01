/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// For C++ host code, provide nixlMemDesc definition
// (nixlMemoryViewH is already defined in nixl_types.h as void*)
#ifndef __CUDACC__
struct nixlMemDesc {
    void *mvh;      // nixlMemoryViewH
    size_t index;
    size_t offset;
};
#endif

/**
 * @brief Launch GPU kernel that posts RDMA write + signal using new device API.
 *
 * @param src_descs_ptr Device memory pointer to source memory descriptors
 * @param dst_descs_ptr Device memory pointer to destination memory descriptors
 * @param desc_count Number of descriptors in the arrays
 * @param signal_desc Remote signal memory descriptor
 * @param signal_ptr Device memory pointer to local signal location
 * @param size Transfer size per descriptor
 * @param level GPU cooperation level (0=THREAD, 1=WARP)
 * @param threads_per_block Number of threads per block
 * @param pipeline Number of writes before signaling
 * @param stream CUDA stream for kernel launch
 */
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
                             cudaStream_t stream);

/**
 * @brief Launch a GPU kernel that waits on a signal value.
 *
 * @param signal_ptr Device memory pointer to signal location
 * @param expected_value Value to wait for
 * @param level GPU cooperation level (0=THREAD, 1=WARP)
 * @param threads_per_block Number of threads per block
 * @param stream CUDA stream for kernel launch
 */
void
launch_wait_for_signal(uintptr_t signal_ptr,
                       uint64_t expected_value,
                       int level,
                       int threads_per_block,
                       cudaStream_t stream);
