/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

/**
 * @brief Launch GPU kernel that posts RDMA write + signal.
 *
 * @param data_req_handles_ptr Device memory pointer to GPU request handles
 * @param data_req_count Number of handles in the array
 * @param signal_req_handle GPU request handle for signal operation
 * @param signal_ptr Device memory pointer to signal location
 * @param level GPU cooperation level (0=THREAD, 1=WARP)
 * @param stream CUDA stream for kernel launch
 */
void
launch_post_write_and_signal(uintptr_t data_req_handles_ptr,
                             int data_req_count,
                             uintptr_t signal_req_handle,
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
 * @param stream CUDA stream for kernel launch
 */
void
launch_wait_for_signal(uintptr_t signal_ptr,
                       uint64_t expected_value,
                       int level,
                       int threads_per_block,
                       cudaStream_t stream);
