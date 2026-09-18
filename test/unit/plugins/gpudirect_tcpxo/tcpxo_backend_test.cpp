/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#include <vector>

#include "gtest/gtest.h"
#include "tcpxo_backend.h"

namespace tcpxo {

class TcpxoBackendTest : public ::testing::Test {
protected:
    std::vector<DxsOpParams>
    Coalesce(std::vector<DxsOpParams> ops) {
        return nixlTcpxoEngine::CoalesceDxsOps(std::move(ops));
    }
};

TEST_F(TcpxoBackendTest, SingleDlistEntry) {
    std::vector<DxsOpParams> input = {{{0, 0}, {0x1000, 100}, {0x2000, 100}, 1, 1, 0}};
    auto result = Coalesce(input);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].local_mr.addr, 0x1000);
    EXPECT_EQ(result[0].local_mr.len, 100);
}

TEST_F(TcpxoBackendTest, TwoContiguousEntriesCoalesce) {
    std::vector<DxsOpParams> input = {{{0, 0}, {0x1000, 100}, {0x2000, 100}, 1, 1, 0},
                                      {{0, 0}, {0x1064, 100}, {0x2064, 100}, 1, 1, 0}};
    auto result = Coalesce(input);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].local_mr.addr, 0x1000);
    EXPECT_EQ(result[0].local_mr.len, 200);
    EXPECT_EQ(result[0].remote_mr.addr, 0x2000);
    EXPECT_EQ(result[0].remote_mr.len, 200);
}

TEST_F(TcpxoBackendTest, ThreeContiguousEntriesCoalesce) {
    std::vector<DxsOpParams> input = {{{0, 0}, {0x1000, 100}, {0x2000, 100}, 1, 1, 0},
                                      {{0, 0}, {0x1064, 100}, {0x2064, 100}, 1, 1, 0},
                                      {{0, 0}, {0x10C8, 100}, {0x20C8, 100}, 1, 1, 0}};
    auto result = Coalesce(input);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].local_mr.len, 300);
}

TEST_F(TcpxoBackendTest, NonContiguousEntriesNoCoalesce) {
    std::vector<DxsOpParams> input = {{{0, 0}, {0x1000, 100}, {0x2000, 100}, 1, 1, 0},
                                      {{0, 0}, {0x10C8, 100}, {0x20C8, 100}, 1, 1, 0}};
    auto result = Coalesce(input);
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].local_mr.addr, 0x1000);
    EXPECT_EQ(result[1].local_mr.addr, 0x10C8);
}

TEST_F(TcpxoBackendTest, OverlappingEntriesNoCoalesce) {
    std::vector<DxsOpParams> input = {{{0, 0}, {0x1000, 100}, {0x2000, 100}, 1, 1, 0},
                                      {{0, 0}, {0x1032, 100}, {0x2032, 100}, 1, 1, 0}};
    auto result = Coalesce(input);
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].local_mr.addr, 0x1000);
    EXPECT_EQ(result[1].local_mr.addr, 0x1032);
}

TEST_F(TcpxoBackendTest, DifferentRemoteRoutingNoCoalesce) {
    std::vector<DxsOpParams> input = {{{0, 0}, {0x1000, 100}, {0x2000, 100}, 1, 1, 0},
                                      {{0, 1}, {0x1064, 100}, {0x2064, 100}, 1, 1, 0}};
    auto result = Coalesce(input);
    ASSERT_EQ(result.size(), 2);
}

TEST_F(TcpxoBackendTest, DifferentRegistrationHandlesNoCoalesce) {
    std::vector<DxsOpParams> input = {{{0, 0}, {0x1000, 4096}, {0x3000, 4096}, 1, 1, 0},
                                      {{0, 0}, {0x2000, 4096}, {0x4000, 4096}, 2, 1, 0}};
    auto result = Coalesce(input);
    ASSERT_EQ(result.size(), 2);
}

TEST_F(TcpxoBackendTest, DifferentRemoteRegistrationHandlesNoCoalesce) {
    std::vector<DxsOpParams> input = {{{0, 0}, {0x1000, 4096}, {0x3000, 4096}, 1, 1, 0},
                                      {{0, 0}, {0x2000, 4096}, {0x4000, 4096}, 1, 2, 0}};
    auto result = Coalesce(input);
    ASSERT_EQ(result.size(), 2);
}

} // namespace tcpxo

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}