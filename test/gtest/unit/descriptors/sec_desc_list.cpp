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

namespace descriptors {

class secDescListTest : public ::testing::Test {
protected:
    static constexpr uint64_t defaultDevId = 1;
    static constexpr size_t defaultLen = 64;

    static nixlSectionDesc
    makeDesc(uintptr_t addr, uint64_t dev_id = defaultDevId, size_t len = defaultLen) {
        return nixlSectionDesc(addr, len, dev_id);
    }

    nixlSecDescList
    makeList() {
        return nixlSecDescList(DRAM_SEG);
    }

    nixlSecDescList
    makeList(std::initializer_list<uintptr_t> addrs) {
        auto list = makeList();
        std::vector<nixlSectionDesc> batch;
        batch.reserve(addrs.size());
        for (auto a : addrs) {
            batch.push_back(makeDesc(a));
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
    expectAddrs(const nixlSecDescList &list, const std::vector<uintptr_t> &expected) {
        ASSERT_EQ(list.descCount(), static_cast<int>(expected.size()));
        ASSERT_TRUE(std::is_sorted(list.begin(), list.end()));

        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(list[i].addr, expected[i]) << "mismatch at index " << i;
            EXPECT_EQ(list[i].devId, defaultDevId) << "devId mismatch at index " << i;
            EXPECT_EQ(list[i].len, defaultLen) << "len mismatch at index " << i;
        }
    }

    static void
    expectAddrsDevIds(const nixlSecDescList &list,
                      const std::vector<std::pair<uint64_t, uintptr_t>> &expected) {
        ASSERT_EQ(list.descCount(), static_cast<int>(expected.size()));
        ASSERT_TRUE(std::is_sorted(list.begin(), list.end()));

        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(list[i].addr, expected[i].first) << "addr mismatch at index " << i;
            EXPECT_EQ(list[i].devId, expected[i].second) << "devId mismatch at index " << i;
            EXPECT_EQ(list[i].len, defaultLen) << "len mismatch at index " << i;
        }
    }
};

TEST_F(secDescListTest, EmptyBatchOnEmptyList) {
    auto list = makeList();
    list.addDescs({});
    ASSERT_TRUE(list.isEmpty());
}

TEST_F(secDescListTest, EmptyBatchOnNonEmptyList) {
    auto list = makeList({100, 200});
    list.addDescs({});
    expectAddrs(list, {100, 200});
}

TEST_F(secDescListTest, SingleElementBatch) {
    auto list = makeList({20, 40});

    list.addDescs({makeDesc(10)});
    expectAddrs(list, {10, 20, 40});

    list.addDescs({makeDesc(30)});
    expectAddrs(list, {10, 20, 30, 40});

    list.addDescs({makeDesc(50)});
    expectAddrs(list, {10, 20, 30, 40, 50});
}

TEST_F(secDescListTest, AllAfterAppend) {
    auto list = makeList({10, 20});
    list.addDescs({makeDesc(30), makeDesc(40)});
    expectAddrs(list, {10, 20, 30, 40});
}

TEST_F(secDescListTest, AllBeforePrepend) {
    auto list = makeList({30, 40});
    list.addDescs({makeDesc(10), makeDesc(20)});
    expectAddrs(list, {10, 20, 30, 40});
}

TEST_F(secDescListTest, InterleavedMerge) {
    auto list = makeList({10, 30, 50});
    list.addDescs({makeDesc(20), makeDesc(40)});
    expectAddrs(list, {10, 20, 30, 40, 50});
}

TEST_F(secDescListTest, UnsortedInput) {
    auto list = makeList({40, 10, 30, 20});
    expectAddrs(list, {10, 20, 30, 40});
}

TEST_F(secDescListTest, SortedInput) {
    auto list = makeList();
    list.addDescs({makeDesc(10), makeDesc(20), makeDesc(30)}, nixlSecDescList::order::SORTED);
    expectAddrs(list, {10, 20, 30});
}

TEST_F(secDescListTest, SortedHintWithUnsortedInputDies) {
    auto list = makeList();
    EXPECT_DEBUG_DEATH(
        list.addDescs({makeDesc(30), makeDesc(10), makeDesc(20)}, nixlSecDescList::order::SORTED),
        "");
}

TEST_F(secDescListTest, OtherListOverload) {
    auto list = makeList({20, 40});
    auto other = makeList({10, 30, 50});

    list.addDescs(std::move(other));
    expectAddrs(list, {10, 20, 30, 40, 50});
}

TEST_F(secDescListTest, DuplicateDescriptors) {
    auto list = makeList({10, 20});
    list.addDescs({makeDesc(10), makeDesc(20)});
    expectAddrs(list, {10, 10, 20, 20});
}

TEST_F(secDescListTest, MultipleDevIds) {
    auto list = makeList();

    // empty batch
    list.addDescs({});
    expectAddrsDevIds(list, {});

    // single element
    list.addDesc(makeDesc(50, 1));
    expectAddrsDevIds(list, {{50, 1}});

    // all after existing
    list.addDescs({makeDesc(100, 1), makeDesc(10, 2)});
    expectAddrsDevIds(list, {{50, 1}, {100, 1}, {10, 2}});

    // all before existing
    list.addDescs({makeDesc(10, 1), makeDesc(50, 0)});
    expectAddrsDevIds(list, {{50, 0}, {10, 1}, {50, 1}, {100, 1}, {10, 2}});

    // interleaved
    list.addDescs({makeDesc(10, 0), makeDesc(75, 1), makeDesc(200, 1), makeDesc(20, 2)});
    expectAddrsDevIds(
        list, {{10, 0}, {50, 0}, {10, 1}, {50, 1}, {75, 1}, {100, 1}, {200, 1}, {10, 2}, {20, 2}});
}

TEST_F(secDescListTest, AddLargeBatches) {
    constexpr int num_batches = 16;
    constexpr int batch_size = 128;

    auto list = makeList();

    std::mt19937 rng(42);
    std::uniform_int_distribution<uintptr_t> addr_dist(1, 1000000);
    std::uniform_int_distribution<uint64_t> dev_dist(0, 8);
    std::uniform_int_distribution<size_t> len_dist(1, 64);

    for (int i = 0; i < num_batches; ++i) {
        std::vector<nixlSectionDesc> batch;
        batch.reserve(batch_size);
        for (int j = 0; j < batch_size; ++j) {
            batch.push_back(makeDesc(addr_dist(rng), dev_dist(rng), len_dist(rng)));
        }

        list.addDescs(std::move(batch));
        assertSorted(list);
        ASSERT_EQ(list.descCount(), (i + 1) * batch_size);
    }
}

} // namespace descriptors
