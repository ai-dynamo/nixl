/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
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

// Unit tests for radosNkvDeriveKey, the RADOS_NKV plugin's token-sequence ->
// NVMe KV key hash. This is SPDK-free, so it runs in CI without a KV target.

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rados_nkv_key.h"

namespace {

std::string
toHex(const std::vector<uint8_t> &bytes) {
    static const char *digits = "0123456789abcdef";
    std::string s;
    s.reserve(bytes.size() * 2);
    for (uint8_t b : bytes) {
        s.push_back(digits[b >> 4]);
        s.push_back(digits[b & 0x0f]);
    }
    return s;
}

// Known-answer vector: pins the 128-bit FNV-1a algorithm so it cannot change
// silently. run_roundtrip_rados.sh reproduces this hash in Python to compute
// the expected RADOS object id; the two MUST agree.
TEST(RadosNkvKey, KnownAnswerVector) {
    std::vector<uint8_t> key;
    ASSERT_TRUE(radosNkvDeriveKey("nixl-key-0000001", 16, key));
    EXPECT_EQ(key.size(), 16u);
    EXPECT_EQ(toHex(key), "1e693b52e8fadc63d197f78380a9f974");
}

TEST(RadosNkvKey, Deterministic) {
    std::vector<uint8_t> a, b;
    ASSERT_TRUE(radosNkvDeriveKey("the-same-sequence", 16, a));
    ASSERT_TRUE(radosNkvDeriveKey("the-same-sequence", 16, b));
    EXPECT_EQ(a, b);
}

TEST(RadosNkvKey, DistinctSequencesDeriveDistinctKeys) {
    std::vector<uint8_t> a, b;
    ASSERT_TRUE(radosNkvDeriveKey("sequence-a", 16, a));
    ASSERT_TRUE(radosNkvDeriveKey("sequence-b", 16, b));
    EXPECT_NE(a, b);
}

TEST(RadosNkvKey, ArbitraryLengthAccepted) {
    // No over-length restriction: a long token sequence still yields a key.
    std::vector<uint8_t> key;
    ASSERT_TRUE(radosNkvDeriveKey(std::string(64 * 1024, 'x'), 16, key));
    EXPECT_EQ(key.size(), 16u);
}

TEST(RadosNkvKey, TruncatesToKeyLen) {
    std::vector<uint8_t> full, eight;
    ASSERT_TRUE(radosNkvDeriveKey("clamp-me", 16, full));
    ASSERT_TRUE(radosNkvDeriveKey("clamp-me", 8, eight));
    EXPECT_EQ(eight.size(), 8u);
    // Truncation keeps the leading bytes of the big-endian hash.
    EXPECT_TRUE(std::equal(eight.begin(), eight.end(), full.begin()));
}

TEST(RadosNkvKey, KeyLenAboveMaxIsClampedTo16) {
    std::vector<uint8_t> key;
    ASSERT_TRUE(radosNkvDeriveKey("seq", 255, key));
    EXPECT_EQ(key.size(), 16u);
}

TEST(RadosNkvKey, EmptySequenceRejected) {
    std::vector<uint8_t> key;
    EXPECT_FALSE(radosNkvDeriveKey("", 16, key));
}

TEST(RadosNkvKey, ZeroKeyLenRejected) {
    std::vector<uint8_t> key;
    EXPECT_FALSE(radosNkvDeriveKey("seq", 0, key));
}

} // namespace
