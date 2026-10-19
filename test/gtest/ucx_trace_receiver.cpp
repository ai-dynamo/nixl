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
#include "common.h"
#include "nixl.h"
#include "ucx_trace_peer.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

using ucx_trace_peer::receiver_name;
using ucx_trace_peer::sender_name;

[[nodiscard]] bool
sendByte(int fd) {
    const char byte = 1;
    return ::write(fd, &byte, 1) == 1;
}

} // namespace

TEST(ucxTraceReceiver, ReceivesBothNotifications) {
    const char *up_fd = std::getenv(ucx_trace_peer::up_fd_var);
    const char *down_fd = std::getenv(ucx_trace_peer::down_fd_var);
    const char *md_file = std::getenv(ucx_trace_peer::md_file_var);
    if ((up_fd == nullptr) || (down_fd == nullptr) || (md_file == nullptr)) {
        GTEST_SKIP() << "started only by ucxTraceContextPropagation";
    }
    const int up = std::atoi(up_fd);
    const int down = std::atoi(down_fd);

    const gtest::LogIgnoreGuard efa(ucx_trace_peer::efa_warning);
    const std::vector<char> buffer(ucx_trace_peer::buffer_size);
    nixlAgent receiver(receiver_name, ucx_trace_peer::agentConfig());
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(receiver.createBackend("UCX", {}, backend), NIXL_SUCCESS);
    nixl_reg_dlist_t regs(DRAM_SEG);
    regs.addDesc(nixlBlobDesc(ucx_trace_peer::address(buffer), buffer.size(), 0));
    ASSERT_EQ(receiver.registerMem(regs), NIXL_SUCCESS);
    std::string md;
    ASSERT_EQ(receiver.getLocalMD(md), NIXL_SUCCESS);
    std::ofstream(md_file, std::ios::binary) << md;
    const uintptr_t address = ucx_trace_peer::address(buffer);
    ASSERT_EQ(::write(up, &address, sizeof(address)), static_cast<ssize_t>(sizeof(address)));

    nixl_notifs_t notifs;
    const auto deadline = std::chrono::steady_clock::now() + ucx_trace_peer::wait_timeout;
    while ((notifs[sender_name].size() < 2) && (std::chrono::steady_clock::now() < deadline)) {
        ASSERT_EQ(receiver.getNotifs(notifs), NIXL_SUCCESS);
        std::this_thread::sleep_for(ucx_trace_peer::poll_interval);
    }
    EXPECT_EQ(
        std::multiset<std::string>(notifs[sender_name].begin(), notifs[sender_name].end()),
        (std::multiset<std::string>{ucx_trace_peer::tagged_msg, ucx_trace_peer::untagged_msg}));
    ASSERT_TRUE(sendByte(up));

    char quit = 0;
    while (::read(down, &quit, 1) > 0) {}
    EXPECT_EQ(receiver.deregisterMem(regs), NIXL_SUCCESS);
}
