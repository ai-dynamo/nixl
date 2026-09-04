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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "common/uuid_v4.h"

namespace {

using nixl::generateRandomBytes;
using nixl::UUIDv4;

constexpr std::uint8_t canary_byte = 0xAB;
constexpr std::size_t entropy_request_limit = 256;
constexpr std::size_t sample_bytes = 16;
constexpr std::size_t guard_bytes = 8;

std::vector<std::uint8_t>
generateWithGuards(std::size_t size) {
    std::vector<std::uint8_t> buffer(size + 2 * guard_bytes, canary_byte);
    generateRandomBytes(buffer.data() + guard_bytes, size);
    return buffer;
}

TEST(GenerateRandomBytes, FillsRequestedRangeOnly) {
    const auto buffer = generateWithGuards(sample_bytes);

    for (std::size_t index = 0; index < guard_bytes; ++index) {
        EXPECT_EQ(buffer[index], canary_byte);
        EXPECT_EQ(buffer[buffer.size() - 1 - index], canary_byte);
    }
    EXPECT_FALSE(std::all_of(buffer.begin() + guard_bytes,
                             buffer.begin() + guard_bytes + sample_bytes,
                             [](std::uint8_t byte) { return byte == canary_byte; }));
}

TEST(GenerateRandomBytes, ZeroSizeRequestIsNoOp) {
    EXPECT_NO_THROW(generateRandomBytes(nullptr, 0));

    std::array<std::uint8_t, 4> buffer{};
    buffer.fill(canary_byte);
    generateRandomBytes(buffer.data(), 0);
    EXPECT_TRUE(std::all_of(
        buffer.begin(), buffer.end(), [](std::uint8_t byte) { return byte == canary_byte; }));
}

TEST(GenerateRandomBytes, FillsBuffersLargerThanOneEntropyRequest) {
    constexpr std::size_t size = 5 * entropy_request_limit + 17;
    const auto buffer = generateWithGuards(size);

    for (std::size_t offset = guard_bytes; offset < guard_bytes + size;
         offset += entropy_request_limit) {
        const auto end =
            buffer.begin() + std::min(offset + entropy_request_limit, guard_bytes + size);
        EXPECT_FALSE(std::all_of(
            buffer.begin() + offset, end, [](std::uint8_t byte) { return byte == canary_byte; }))
            << "unfilled region at offset " << offset;
    }
}

TEST(GenerateRandomBytes, ConcurrentCallsProduceDistinctValues) {
    constexpr std::size_t threads = 8;
    constexpr std::size_t per_thread = 256;

    std::vector<std::vector<std::string>> results(threads);
    std::barrier start(static_cast<std::ptrdiff_t>(threads));
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (std::size_t worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&results, &start, worker]() {
            auto &local = results[worker];
            local.reserve(per_thread);
            start.arrive_and_wait();
            for (std::size_t index = 0; index < per_thread; ++index) {
                std::array<std::uint8_t, sample_bytes> bytes{};
                generateRandomBytes(bytes.data(), bytes.size());
                local.emplace_back(bytes.begin(), bytes.end());
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }

    std::set<std::string> merged;
    for (const auto &local : results) {
        ASSERT_EQ(local.size(), per_thread);
        merged.insert(local.begin(), local.end());
    }

    EXPECT_EQ(merged.size(), threads * per_thread);
}

TEST(UUIDv4, PreservesVersionAndVariantBits) {
    constexpr std::size_t samples = 256;

    for (std::size_t index = 0; index < samples; ++index) {
        const UUIDv4 uuid;
        const auto &data = uuid.get_data();

        EXPECT_EQ(data[6] & 0xF0, 0x40);
        EXPECT_EQ(data[8] & 0xC0, 0x80);
    }
}

} // namespace
