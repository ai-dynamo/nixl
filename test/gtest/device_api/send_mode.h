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

#ifndef TEST_GTEST_DEVICE_API_SEND_MODE_H
#define TEST_GTEST_DEVICE_API_SEND_MODE_H

namespace nixl::device_api {
enum class send_mode_t {
    NODELAY_WITH_REQ,
    NODELAY_WITHOUT_REQ,
    DELAY_WITHOUT_REQ,
    MULTI_CHANNEL,
};

[[nodiscard]] inline bool
delay(send_mode_t mode) {
    return mode == send_mode_t::DELAY_WITHOUT_REQ;
}

[[nodiscard]] inline bool
request(send_mode_t mode) {
    return mode == send_mode_t::NODELAY_WITH_REQ || mode == send_mode_t::MULTI_CHANNEL;
}
} // namespace nixl::device_api
#endif // TEST_GTEST_DEVICE_API_SEND_MODE_H
