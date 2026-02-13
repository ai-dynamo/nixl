/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TEST_GTEST_DEVICE_API_KERNEL_PARAMS_H
#define TEST_GTEST_DEVICE_API_KERNEL_PARAMS_H

#include "send_mode.h"

#include <nixl_device.cuh>

namespace nixl::device_api {
enum class operation_t : uint64_t {
    SINGLE_WRITE,
    PARTIAL_WRITE,
    WRITE,
    SIGNAL_POST,
    SIGNAL_WAIT,
    SIGNAL_WRITE
};

static constexpr size_t default_num_iters = 100;
static constexpr size_t default_num_threads = 32;
static constexpr uint64_t default_signal_increment = 42;

struct kernelParams {
    kernelParams(operation_t op, nixl_gpu_level_t l, send_mode_t sm, nixlGpuXferReqH req_handle)
        : operation(op),
          level(l),
          withRequest(request(sm)),
          noDelay(!delay(sm)),
          numChannels(sm == send_mode_t::MULTI_CHANNEL ? 32 : 1),
          reqHandle(req_handle) {}

    kernelParams(operation_t op, nixl_gpu_level_t l) : operation(op), level(l), numIters(1) {}

    kernelParams(operation_t op, nixl_gpu_level_t l, size_t num_threads)
        : operation(op),
          level(l),
          numThreads(num_threads),
          numIters(1) {}

    const operation_t operation;
    const nixl_gpu_level_t level;
    const unsigned numThreads = default_num_threads;
    const unsigned numBlocks = 1;
    const size_t numIters = default_num_iters;
    bool withRequest = false;
    bool noDelay = false;
    unsigned numChannels = 1;

    const nixlGpuXferReqH reqHandle = nullptr;

    struct {
        unsigned index;
        size_t localOffset;
        size_t remoteOffset;
        size_t size;
    } singleWrite;

    struct {
        size_t count;
        const unsigned *descIndices;
        const size_t *sizes;
        const size_t *localOffsets;
        const size_t *remoteOffsets;
        unsigned signalDescIndex;
        uint64_t signalInc;
        size_t signalOffset;
    } partialWrite;

    struct {
        uint64_t signalInc;
    } write;

    struct {
        unsigned signalDescIndex;
        uint64_t signalInc;
        size_t signalOffset;
    } signalPost;

    struct {
        const void *signalAddr;
        uint64_t expectedValue;
    } signalWait;

    struct {
        void *signalAddr;
        uint64_t value;
    } signalWrite;
};
} // namespace nixl::device_api
#endif // TEST_GTEST_DEVICE_API_KERNEL_PARAMS_H
