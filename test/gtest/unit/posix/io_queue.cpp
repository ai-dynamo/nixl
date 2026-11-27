/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Eylon Eliyahu Krause. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_set>
#include <utility>

#include "io_queue.h"

namespace {

class ioQueueProbe : public nixlPosixIOQueueImpl<uint64_t> {
public:
    explicit ioQueueProbe(uint32_t ios_pool_size)
        : nixlPosixIOQueueImpl(ios_pool_size, DEF_KERNEL_QUEUE_SIZE) {}

    nixl_status_t
    enqueue(int, void *, size_t, off_t, bool, nixlPosixIOQueueDoneCb, void *) override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    post() override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    poll() override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const std::vector<uint64_t> &
    entries() const {
        return ios_;
    }

    const std::list<uint64_t *> &
    freeEntries() const {
        return free_ios_;
    }
};

class ioQueuePoolTest : public ::testing::TestWithParam<std::pair<uint32_t, uint32_t>> {};

constexpr auto min_pool_size = nixlPosixIOQueue::MIN_IOS_POOL_SIZE;
constexpr auto max_pool_size = nixlPosixIOQueue::MAX_IOS_POOL_SIZE;

TEST_P(ioQueuePoolTest, FreeEntriesMatchNormalizedStorage) {
    const auto [requested_size, expected_size] = GetParam();
    ioQueueProbe queue(requested_size);

    ASSERT_EQ(queue.entries().size(), expected_size);
    ASSERT_EQ(queue.freeEntries().size(), expected_size);

    std::unordered_set<const uint64_t *> available_entries;
    for (const auto &entry : queue.entries()) {
        available_entries.insert(&entry);
    }

    for (const auto *entry : queue.freeEntries()) {
        ASSERT_EQ(available_entries.erase(entry), 1u);
    }
    EXPECT_TRUE(available_entries.empty());
}

INSTANTIATE_TEST_SUITE_P(PoolSizeBounds,
                         ioQueuePoolTest,
                         ::testing::Values(std::pair{1u, min_pool_size},
                                           std::pair{min_pool_size - 1, min_pool_size},
                                           std::pair{min_pool_size, min_pool_size},
                                           std::pair{min_pool_size + 1, min_pool_size + 1},
                                           std::pair{max_pool_size - 1, max_pool_size - 1},
                                           std::pair{max_pool_size, max_pool_size},
                                           std::pair{max_pool_size + 1, max_pool_size}));

} // namespace
