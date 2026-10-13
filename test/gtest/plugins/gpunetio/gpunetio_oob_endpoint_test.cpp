/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_oob_endpoint.h"

#include <gtest/gtest.h>

#include <limits>

TEST(GpunetioOobEndpoint, BoundsAgentNameAllocation) {
    EXPECT_FALSE(isValidGpunetioAgentNameSize(0));
    EXPECT_TRUE(isValidGpunetioAgentNameSize(1));
    EXPECT_TRUE(isValidGpunetioAgentNameSize(GPUNETIO_MAX_AGENT_NAME_SIZE));
    EXPECT_FALSE(isValidGpunetioAgentNameSize(GPUNETIO_MAX_AGENT_NAME_SIZE + 1));
    EXPECT_FALSE(isValidGpunetioAgentNameSize(std::numeric_limits<std::size_t>::max()));
}

TEST(GpunetioOobEndpoint, AcceptsPortBoundsAndDefault) {
    EXPECT_EQ(parseGpunetioOobPort(""), GPUNETIO_DEFAULT_OOB_PORT);
    EXPECT_EQ(parseGpunetioOobPort("1"), 1);
    EXPECT_EQ(parseGpunetioOobPort("65535"), 65535);
}

TEST(GpunetioOobEndpoint, RejectsInvalidPorts) {
    for (const char *value : {"0", "-1", "+1", "65536", "12x", " 6544", "6544 "}) {
        SCOPED_TRACE(value);
        EXPECT_THROW(parseGpunetioOobPort(value), std::invalid_argument);
    }
}

TEST(GpunetioOobEndpoint, PreservesLegacyDefaultPortMetadata) {
    const auto endpoint = parseGpunetioOobEndpoint("192.0.2.1");
    EXPECT_EQ(endpoint.ipv4, "192.0.2.1");
    EXPECT_EQ(endpoint.port, GPUNETIO_DEFAULT_OOB_PORT);
    EXPECT_EQ(formatGpunetioOobEndpoint(endpoint.ipv4, endpoint.port), "192.0.2.1");
}

TEST(GpunetioOobEndpoint, KeepsDistinctExplicitEndpoints) {
    const auto first = parseGpunetioOobEndpoint("198.51.100.2:16544");
    const auto second = parseGpunetioOobEndpoint("198.51.100.2:16545");
    EXPECT_EQ(first.ipv4, second.ipv4);
    EXPECT_NE(first.port, second.port);
    EXPECT_EQ(formatGpunetioOobEndpoint(first.ipv4, first.port), "198.51.100.2:16544");
    EXPECT_EQ(formatGpunetioOobEndpoint(second.ipv4, second.port), "198.51.100.2:16545");
}

TEST(GpunetioOobEndpoint, RejectsMalformedMetadata) {
    for (const char *value : {"",
                              "192.0.2",
                              "256.0.0.1",
                              "192.0.2.1:",
                              "192.0.2.1:0",
                              "192.0.2.1:65536",
                              "192.0.2.1:12x",
                              "192.0.2.1:6544:1",
                              "[::1]:6544"}) {
        SCOPED_TRACE(value);
        EXPECT_THROW(parseGpunetioOobEndpoint(value), std::invalid_argument);
    }
    EXPECT_THROW(parseGpunetioOobEndpoint(std::string("192.0.2.1\0:6544", 16)),
                 std::invalid_argument);
}
