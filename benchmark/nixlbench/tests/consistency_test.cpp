/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/scope_guard.h"
#include "utils/utils.h"

#include <gtest/gtest.h>

#include <array>
#include <tuple>

namespace {

void
checkHostDestination(const std::string &operation, bool is_initiator, uint8_t expected) {
    const auto saved = std::make_tuple(xferBenchConfig::backend,
                                       xferBenchConfig::op_type,
                                       xferBenchConfig::initiator_seg_type,
                                       xferBenchConfig::target_seg_type,
                                       xferBenchConfig::check_consistency,
                                       xferBenchConfig::page_size);
    auto restore = make_scope_guard([&] {
        std::tie(xferBenchConfig::backend,
                 xferBenchConfig::op_type,
                 xferBenchConfig::initiator_seg_type,
                 xferBenchConfig::target_seg_type,
                 xferBenchConfig::check_consistency,
                 xferBenchConfig::page_size) = saved;
    });
    xferBenchConfig::backend = XFERBENCH_BACKEND_GPUNETIO;
    xferBenchConfig::op_type = operation;
    xferBenchConfig::initiator_seg_type = XFERBENCH_SEG_TYPE_DRAM;
    xferBenchConfig::target_seg_type = XFERBENCH_SEG_TYPE_DRAM;
    xferBenchConfig::check_consistency = true;
    xferBenchConfig::page_size = 4096;

    // Exercise the real checker with host data: no GPU or GPUNETIO plugin is needed.
    std::array<uint8_t, 4096> destination;
    destination.fill(expected);
    std::vector<std::vector<xferBenchIOV>> iovs{
        {xferBenchIOV(reinterpret_cast<uintptr_t>(destination.data()), destination.size(), 0)}};
    EXPECT_TRUE(xferBenchUtils::validateTransfer(is_initiator, iovs, iovs));
    destination.back() ^= 1;
    EXPECT_FALSE(xferBenchUtils::validateTransfer(is_initiator, iovs, iovs));
}

TEST(NixlbenchConsistency, GpunetioWriteChecksDestinationBytes) {
    checkHostDestination(XFERBENCH_OP_WRITE, false, XFERBENCH_INITIATOR_BUFFER_ELEMENT);
}

TEST(NixlbenchConsistency, GpunetioReadChecksDestinationBytes) {
    checkHostDestination(XFERBENCH_OP_READ, true, XFERBENCH_TARGET_BUFFER_ELEMENT);
}

} // namespace
