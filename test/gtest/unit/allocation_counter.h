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
#ifndef NIXL_TEST_GTEST_UNIT_ALLOCATION_COUNTER_H
#define NIXL_TEST_GTEST_UNIT_ALLOCATION_COUNTER_H

#include <cstddef>

namespace gtest {

class AllocationCounter {
public:
    AllocationCounter() noexcept;
    ~AllocationCounter();

    AllocationCounter(const AllocationCounter &) = delete;
    AllocationCounter &
    operator=(const AllocationCounter &) = delete;

    [[nodiscard]] std::size_t
    count() const noexcept;

private:
    const std::size_t start_;
    const bool wasCounting_;
};

} // namespace gtest

#endif
