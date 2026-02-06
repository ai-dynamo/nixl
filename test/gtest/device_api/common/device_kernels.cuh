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

#ifndef TEST_GTEST_DEVICE_API_COMMON_DEVICE_KERNELS_CUH
#define TEST_GTEST_DEVICE_API_COMMON_DEVICE_KERNELS_CUH

#include <nixl_device.cuh>
#include <cstddef>
#include <cstdint>

namespace nixl::test::device_api {
enum class operation_t : uint64_t {
    SINGLE_WRITE,
    PARTIAL_WRITE,
    WRITE,
    SIGNAL_POST,
    SIGNAL_WAIT,
    SIGNAL_WRITE
};

struct kernelParams {
    operation_t operation;
    nixl_gpu_level_t level;
    unsigned numThreads;
    unsigned numBlocks;
    size_t numIters;
    bool withRequest;
    bool withNoDelay;
    unsigned numChannels = 1;

    nixlGpuXferReqH reqHandle;

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

[[nodiscard]] nixl_status_t
launchKernel(const kernelParams &params);
} // namespace nixl::test::device_api
#endif // NIXL_DEVICE_KERNELS_CUH
