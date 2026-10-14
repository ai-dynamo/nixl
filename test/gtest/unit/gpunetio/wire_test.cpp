/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_wire.h"
#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <string>

namespace {

TEST(GpunetioWire, MetadataRoundTrip) {
    for (const uint32_t key : {0U, 17U, std::numeric_limits<uint32_t>::max()}) {
        const uintptr_t address = std::numeric_limits<uintptr_t>::max();
        const size_t length = std::numeric_limits<size_t>::max();
        const std::string input =
            std::to_string(key) + "-" + std::to_string(address) + "-" + std::to_string(length);
        uint32_t rkey;
        uintptr_t addr;
        size_t size;
        ASSERT_TRUE(nixl::gpunetio::parseRemoteMetadata(input, rkey, addr, size));
        EXPECT_EQ(rkey, key);
        EXPECT_EQ(addr, address);
        EXPECT_EQ(size, length);
    }
}

TEST(GpunetioWire, MetadataRejectsMissingAndExtraFields) {
    for (const auto input : {"", "1", "1-2", "1-2-", "-2-3", "1--3", "1-2-3-", "1-2-3-4"}) {
        SCOPED_TRACE(input);
        uint32_t rkey = 7;
        uintptr_t addr = 8;
        size_t size = 9;
        EXPECT_FALSE(nixl::gpunetio::parseRemoteMetadata(input, rkey, addr, size));
        EXPECT_EQ(rkey, 7U);
        EXPECT_EQ(addr, 8U);
        EXPECT_EQ(size, 9U);
    }
}

TEST(GpunetioWire, MetadataRejectsInvalidNumbers) {
    for (const auto input : {"x-2-3",
                             "1-x-3",
                             "1-2-x",
                             "1x-2-3",
                             "1-2x-3",
                             "1-2-3x",
                             "+1-2-3",
                             "1-+2-3",
                             "1-2-+3",
                             " 1-2-3",
                             "1- 2-3",
                             "1-2-3 ",
                             "1-2--3",
                             "4294967296-2-3",
                             "1-18446744073709551616-3",
                             "1-2-18446744073709551616"}) {
        SCOPED_TRACE(input);
        uint32_t rkey;
        uintptr_t addr;
        size_t size;
        EXPECT_FALSE(nixl::gpunetio::parseRemoteMetadata(input, rkey, addr, size));
    }
    const std::string input("1-2-3\0ignored", 13);
    uint32_t rkey;
    uintptr_t addr;
    size_t size;
    EXPECT_FALSE(nixl::gpunetio::parseRemoteMetadata(input, rkey, addr, size));
}

TEST(GpunetioWire, MetadataDoesNotRequireTerminator) {
    const std::array<char, 5> input{'1', '-', '2', '-', '3'};
    uint32_t rkey;
    uintptr_t addr;
    size_t size;
    ASSERT_TRUE(nixl::gpunetio::parseRemoteMetadata(
        std::string_view(input.data(), input.size()), rkey, addr, size));
    EXPECT_EQ(rkey, 1U);
    EXPECT_EQ(addr, 2U);
    EXPECT_EQ(size, 3U);
}

TEST(GpunetioWire, NotificationPreservesBinaryPayloadAndIgnoresPadding) {
    for (const std::string &value :
         {std::string(), std::string("hello"), std::string("a\0b", 3), std::string("DOCASDOCAE")}) {
        const std::string packet = "DOCAS" + std::to_string(value.size()) + "DOCAE" + value;
        for (const std::string &padding : {std::string(), std::string(32, '\x7f')}) {
            const std::string slot = packet + padding;
            std::string_view payload;
            ASSERT_TRUE(nixl::gpunetio::parseNotification(slot, "DOCAS", "DOCAE", payload));
            EXPECT_EQ(payload, value);
            EXPECT_EQ(payload.data(), slot.data() + packet.size() - value.size());
        }
    }
}

TEST(GpunetioWire, NotificationRejectsMalformedHeadersAndLengths) {
    for (const auto slot : {"",
                            "DOCAS",
                            "DOCAS3",
                            "DOCASDOCAE",
                            "xDOCAS0DOCAE",
                            "DOCASxDOCAE",
                            "DOCAS-1DOCAE",
                            "DOCAS+1DOCAE",
                            "DOCAS 1DOCAE",
                            "DOCAS1xDOCAE",
                            "DOCAS1 DOCAE",
                            "DOCAS18446744073709551616DOCAE",
                            "DOCAS4DOCAEabc"}) {
        SCOPED_TRACE(slot);
        std::string_view payload = "unchanged";
        EXPECT_FALSE(nixl::gpunetio::parseNotification(slot, "DOCAS", "DOCAE", payload));
        EXPECT_EQ(payload, "unchanged");
    }
}

TEST(GpunetioWire, NotificationDoesNotRequireTerminatorOrReadAdjacentSlot) {
    const std::array<char, 12> slot{'D', 'O', 'C', 'A', 'S', '1', 'D', 'O', 'C', 'A', 'E', 'x'};
    std::string_view payload;
    ASSERT_TRUE(nixl::gpunetio::parseNotification(
        std::string_view(slot.data(), slot.size()), "DOCAS", "DOCAE", payload));
    EXPECT_EQ(payload, "x");
    EXPECT_FALSE(nixl::gpunetio::parseNotification(
        std::string_view(slot.data(), slot.size() - 1), "DOCAS", "DOCAE", payload));
    // The next slot must not supply either a missing end tag or missing payload.
    const std::string adjacent = "DOCAS1DOCAExDOCAS0DOCAE";
    EXPECT_FALSE(nixl::gpunetio::parseNotification(
        std::string_view(adjacent.data(), 10), "DOCAS", "DOCAE", payload));
    EXPECT_FALSE(nixl::gpunetio::parseNotification(
        std::string_view(adjacent.data(), 11), "DOCAS", "DOCAE", payload));
}

} // namespace
