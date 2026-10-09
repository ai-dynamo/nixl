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

#include <atomic>
#include <cstdint>

#include "gpunetio_completion_status.h"

namespace {

TEST(GpuNetioCompletionStatus, MapsLatchedGpuErrorToBackendFailure) {
    uint32_t error_flag = 0;
    EXPECT_EQ(nixl::gpunetio::completionStatus(&error_flag), NIXL_SUCCESS);

    error_flag = 1;
    EXPECT_EQ(nixl::gpunetio::completionStatus(&error_flag), NIXL_ERR_BACKEND);
}

TEST(GpuNetioCompletionStatus, ReservesNotificationSlotAndMatchingAddress) {
    std::atomic<uint32_t> send_pi = 3;
    uint32_t notification_slot = 0;
    uint8_t buffer[64] = {};
    constexpr uint32_t elems_num = 8;
    constexpr size_t slot_size = 8;

    uint8_t *notification_addr = nixl::gpunetio::reserveNotificationSlot(
        send_pi, elems_num, buffer, slot_size, notification_slot);

    EXPECT_EQ(notification_slot, 3);
    EXPECT_EQ(notification_addr, buffer + 24);
    EXPECT_EQ(send_pi.load(), 4);
}

} // namespace
