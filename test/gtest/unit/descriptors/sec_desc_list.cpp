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
#include <random>
#include <vector>

#include "mem_section.h"

namespace gtest {
namespace descriptors {

    class secDescListTest : public ::testing::Test {
    protected:
        static constexpr uint64_t defaultDevId = 0;
        static constexpr size_t defaultLen = 64;

        static nixlSectionDesc
        makeDescAddr(uintptr_t addr) {
            return nixlSectionDesc(addr, defaultLen, defaultDevId);
        }

        static nixlSectionDesc
        makeDescDevIdAddr(uint64_t dev_id, uintptr_t addr) {
            return nixlSectionDesc(addr, defaultLen, dev_id);
        }

        nixlSecDescList
        makeList() {
            return nixlSecDescList(DRAM_SEG);
        }

        nixlSecDescList
        makeListAddrs(std::initializer_list<uintptr_t> addrs) {
            auto list = makeList();
            std::vector<nixlSectionDesc> batch;
            batch.reserve(addrs.size());
            for (auto a : addrs) {
                batch.push_back(makeDescAddr(a));
            }
            list.addDescs(std::move(batch));
            assertSorted(list);
            return list;
        }

        static void
        assertSorted(const nixlSecDescList &list) {
            ASSERT_TRUE(std::is_sorted(list.begin(), list.end()));
        }

        static void
        expectAddrs(const nixlSecDescList &list, const std::vector<uintptr_t> &addrs) {
            ASSERT_EQ(list.descCount(), static_cast<int>(addrs.size()));
            ASSERT_TRUE(std::is_sorted(list.begin(), list.end()));

            for (size_t i = 0; i < addrs.size(); ++i) {
                EXPECT_EQ(list[i].addr, addrs[i]) << "mismatch at index " << i;
            }
        }

        static void
        expectDevIdsAddrs(const nixlSecDescList &list,
                          const std::vector<std::pair<uint64_t, uintptr_t>> &expected) {
            ASSERT_EQ(list.descCount(), static_cast<int>(expected.size()));
            ASSERT_TRUE(std::is_sorted(list.begin(), list.end()));

            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_EQ(list[i].devId, expected[i].first) << "devId mismatch at index " << i;
                EXPECT_EQ(list[i].addr, expected[i].second) << "addr mismatch at index " << i;
            }
        }
    };

    TEST_F(secDescListTest, EmptyBatchOnEmptyList) {
        auto list = makeList();
        list.addDescs({});
        ASSERT_TRUE(list.isEmpty());
    }

    TEST_F(secDescListTest, EmptyBatchOnNonEmptyList) {
        auto list = makeListAddrs({100, 200});
        list.addDescs({});
        expectAddrs(list, {100, 200});
    }

    TEST_F(secDescListTest, SingleElementBatch) {
        auto list = makeListAddrs({20, 40});

        list.addDescs({makeDescAddr(10)});
        expectAddrs(list, {10, 20, 40});

        list.addDescs({makeDescAddr(30)});
        expectAddrs(list, {10, 20, 30, 40});

        list.addDescs({makeDescAddr(50)});
        expectAddrs(list, {10, 20, 30, 40, 50});
    }

    TEST_F(secDescListTest, AllAfterAppend) {
        auto list = makeListAddrs({10, 20});
        list.addDescs({makeDescAddr(30), makeDescAddr(40)});
        expectAddrs(list, {10, 20, 30, 40});
    }

    TEST_F(secDescListTest, AllBeforePrepend) {
        auto list = makeListAddrs({30, 40});
        list.addDescs({makeDescAddr(10), makeDescAddr(20)});
        expectAddrs(list, {10, 20, 30, 40});
    }

    TEST_F(secDescListTest, InterleavedMerge) {
        auto list = makeListAddrs({10, 30, 50});
        list.addDescs({makeDescAddr(20), makeDescAddr(40)});
        expectAddrs(list, {10, 20, 30, 40, 50});
    }

    TEST_F(secDescListTest, UnsortedInput) {
        auto list = makeListAddrs({40, 10, 30, 20});
        expectAddrs(list, {10, 20, 30, 40});
    }

    TEST_F(secDescListTest, SortedInput) {
        auto list = makeList();
        list.addDescs({makeDescAddr(10), makeDescAddr(20), makeDescAddr(30)},
                      nixlSecDescList::order::SORTED);
        expectAddrs(list, {10, 20, 30});
    }

    TEST_F(secDescListTest, SortedHintWithUnsortedInputDies) {
        auto list = makeList();
        EXPECT_DEBUG_DEATH(list.addDescs({makeDescAddr(30), makeDescAddr(10), makeDescAddr(20)},
                                         nixlSecDescList::order::SORTED),
                           "");
    }

    TEST_F(secDescListTest, ListMoveOverload) {
        auto list = makeListAddrs({20, 40});
        auto other = makeListAddrs({10, 30, 50});

        list.addDescs(std::move(other));
        expectAddrs(list, {10, 20, 30, 40, 50});
        EXPECT_TRUE(other.isEmpty());
    }

    TEST_F(secDescListTest, DuplicateDescriptors) {
        auto list = makeListAddrs({10, 20});
        list.addDescs({makeDescAddr(10), makeDescAddr(20)});
        expectAddrs(list, {10, 10, 20, 20});
    }

    TEST_F(secDescListTest, MultipleDevIds) {
        auto list = makeList();

        // empty batch
        list.addDescs({});
        expectDevIdsAddrs(list, {});

        // single element
        list.addDesc(makeDescDevIdAddr(1, 50));
        expectDevIdsAddrs(list, {{1, 50}});

        // all after existing
        list.addDescs({makeDescDevIdAddr(1, 100), makeDescDevIdAddr(2, 10)});
        expectDevIdsAddrs(list, {{1, 50}, {1, 100}, {2, 10}});

        // all before existing
        list.addDescs({makeDescDevIdAddr(1, 10), makeDescDevIdAddr(0, 50)});
        expectDevIdsAddrs(list, {{0, 50}, {1, 10}, {1, 50}, {1, 100}, {2, 10}});

        // interleaved
        list.addDescs({makeDescDevIdAddr(0, 10),
                       makeDescDevIdAddr(1, 75),
                       makeDescDevIdAddr(1, 200),
                       makeDescDevIdAddr(2, 20)});
        expectDevIdsAddrs(
            list,
            {{0, 10}, {0, 50}, {1, 10}, {1, 50}, {1, 75}, {1, 100}, {1, 200}, {2, 10}, {2, 20}});
    }

    TEST_F(secDescListTest, AddLargeBatches) {
        constexpr int num_batches = 16;
        constexpr int batch_size = 128;

        auto list = makeList();

        std::mt19937 rng(42);
        std::uniform_int_distribution<uintptr_t> dist(1, 1000000);

        for (int i = 0; i < num_batches; ++i) {
            std::vector<nixlSectionDesc> batch;
            batch.reserve(batch_size);
            for (int j = 0; j < batch_size; ++j) {
                batch.push_back(makeDescAddr(dist(rng)));
            }

            list.addDescs(std::move(batch));
            assertSorted(list);
            ASSERT_EQ(list.descCount(), (i + 1) * batch_size);
        }
    }

} // namespace descriptors
} // namespace gtest
