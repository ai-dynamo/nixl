// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gpu/device_types.cuh>

// Compile the actual shim against a CPU mock, retaining the real native types.
// This checks forwarding, not transport progress or GPU memory ordering.
namespace {
nixlMemViewElem last_src{}, last_dst{};
nixlGpuXferStatusH *last_request = nullptr;
uint64_t last_size = 0, last_value = 0, last_flags = 0;
unsigned last_channel = 0;
int last_level = -1, submits = 0, polls = 0;
nixl_status_t next_status = NIXL_IN_PROG;
} // namespace

template<nixl_gpu_level_t level>
nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel,
        uint64_t flags,
        nixlGpuXferStatusH *request) {
    last_src = src;
    last_dst = dst;
    last_size = size;
    last_channel = channel;
    last_flags = flags;
    last_request = request;
    last_level = static_cast<int>(level);
    ++submits;
    return next_status;
}

template<nixl_gpu_level_t level>
nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel,
              uint64_t flags,
              nixlGpuXferStatusH *request) {
    last_value = value;
    return nixlPut<level>({}, counter, 0, channel, flags, request);
}

template<nixl_gpu_level_t level>
nixl_status_t
nixlGpuGetXferStatus(nixlGpuXferStatusH &request) {
    last_request = &request;
    last_level = static_cast<int>(level);
    ++polls;
    return next_status;
}

#define NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH
#define __device__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#include "../../../../src/bindings/cute/device.cu"
#pragma GCC diagnostic pop
#undef __device__
#undef NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH

TEST(CuteDeviceBinding, PutForwardsArgumentsAndStatusWithoutPolling) {
    nixlGpuXferStatusH request;
    const auto address = reinterpret_cast<uint64_t>(&request);
    const uint64_t large_offset = (uint64_t{1} << 33) + 5;
    for (int level = 0; level < 3; ++level) {
        for (auto result : {NIXL_SUCCESS, NIXL_IN_PROG, NIXL_ERR_BACKEND}) {
            next_status = result;
            submits = polls = 0;
            EXPECT_EQ(cute_nixl_put(11, 22, 17, 3, 4, large_offset, 6, 2, 1, address, level),
                      result);
            EXPECT_EQ(last_src.mvh, reinterpret_cast<nixlMemViewH>(11));
            EXPECT_EQ(last_dst.mvh, reinterpret_cast<nixlMemViewH>(22));
            EXPECT_EQ(last_src.index, 3u);
            EXPECT_EQ(last_dst.index, 4u);
            EXPECT_EQ(last_src.offset, large_offset);
            EXPECT_EQ(last_dst.offset, 6u);
            EXPECT_EQ(last_size, 17u);
            EXPECT_EQ(last_channel, 2u);
            EXPECT_EQ(last_flags, 1u);
            EXPECT_EQ(last_request, &request);
            EXPECT_EQ(last_level, level);
            EXPECT_EQ(submits, 1);
            EXPECT_EQ(polls, 0);
        }
    }
}

TEST(CuteDeviceBinding, AtomicForwardsArgumentsAndStatusWithoutPolling) {
    nixlGpuXferStatusH request;
    const auto address = reinterpret_cast<uint64_t>(&request);
    for (int level = 0; level < 3; ++level) {
        for (auto result : {NIXL_SUCCESS, NIXL_IN_PROG, NIXL_ERR_BACKEND}) {
            next_status = result;
            submits = polls = 0;
            EXPECT_EQ(cute_nixl_atomic_add(7, 22, 4, 16, 2, 1, address, level), result);
            EXPECT_EQ(last_value, 7u);
            EXPECT_EQ(last_dst.mvh, reinterpret_cast<nixlMemViewH>(22));
            EXPECT_EQ(last_dst.index, 4u);
            EXPECT_EQ(last_dst.offset, 16u);
            EXPECT_EQ(last_channel, 2u);
            EXPECT_EQ(last_flags, 1u);
            EXPECT_EQ(last_request, &request);
            EXPECT_EQ(last_level, level);
            EXPECT_EQ(submits, 1);
            EXPECT_EQ(polls, 0);
        }
    }
}

TEST(CuteDeviceBinding, NullRequestIsForwardedNotConvertedToBlocking) {
    next_status = NIXL_IN_PROG;
    polls = 0;
    EXPECT_EQ(cute_nixl_put(11, 22, 17, 0, 0, 0, 0, 0, 0, 0, 0), NIXL_IN_PROG);
    EXPECT_EQ(last_request, nullptr);
    EXPECT_EQ(cute_nixl_atomic_add(1, 22, 0, 0, 0, 0, 0, 0), NIXL_IN_PROG);
    EXPECT_EQ(last_request, nullptr);
    EXPECT_EQ(polls, 0);
}

TEST(CuteDeviceBinding, ProgressPerformsExactlyOneNativePoll) {
    nixlGpuXferStatusH request;
    const auto address = reinterpret_cast<uint64_t>(&request);
    for (int level = 0; level < 3; ++level) {
        for (auto result : {NIXL_IN_PROG, NIXL_SUCCESS, NIXL_ERR_BACKEND}) {
            next_status = result;
            submits = polls = 0;
            EXPECT_EQ(cute_nixl_get_xfer_status(address, level), result);
            EXPECT_EQ(last_request, &request);
            EXPECT_EQ(last_level, level);
            EXPECT_EQ(polls, 1);
            EXPECT_EQ(submits, 0);
        }
    }
}

TEST(CuteDeviceBinding, UnsupportedCooperationDoesNotCallBackend) {
    nixlGpuXferStatusH request;
    const auto address = reinterpret_cast<uint64_t>(&request);
    submits = polls = 0;
    for (int level : {-1, 3}) {
        EXPECT_EQ(cute_nixl_put(11, 22, 17, 0, 0, 0, 0, 0, 0, address, level),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(cute_nixl_atomic_add(1, 22, 0, 0, 0, 0, address, level), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(cute_nixl_get_xfer_status(address, level), NIXL_ERR_INVALID_PARAM);
    }
    EXPECT_EQ(submits, 0);
    EXPECT_EQ(polls, 0);
}

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
