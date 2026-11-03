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
#ifndef NIXL_TEST_GTEST_UCX_TRACE_PEER_H
#define NIXL_TEST_GTEST_UCX_TRACE_PEER_H

#include "nixl.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ucx_trace_peer {

constexpr const char *sender_name = "sender";
constexpr const char *receiver_name = "receiver";
constexpr const char *tagged_msg = "tagged";
constexpr const char *untagged_msg = "untagged";
constexpr const char *efa_warning =
    "Amazon EFA\\(s\\) were detected, but the UCX backend was configured";
constexpr const char *receiver_binary = "ucx_trace_receiver";
constexpr const char *up_fd_var = "NIXL_TEST_RECEIVER_UP_FD";
constexpr const char *down_fd_var = "NIXL_TEST_RECEIVER_DOWN_FD";
constexpr const char *md_file_var = "NIXL_TEST_RECEIVER_MD_FILE";
constexpr size_t buffer_size = 4096;
constexpr std::chrono::seconds wait_timeout{30};
constexpr std::chrono::milliseconds poll_interval{10};

[[nodiscard]] inline nixlAgentConfig
agentConfig() {
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    return cfg;
}

[[nodiscard]] inline uintptr_t
address(const std::vector<char> &buffer) {
    return reinterpret_cast<uintptr_t>(buffer.data());
}

} // namespace ucx_trace_peer

#endif // NIXL_TEST_GTEST_UCX_TRACE_PEER_H
