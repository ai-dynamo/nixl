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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace {

class UcxRequestRelease : public testing::TestWithParam<size_t> {};

TEST_P(UcxRequestRelease, ActiveReadRemainsPollableAfterRefusedRelease) {
    gtest::ScopedEnv env;
    // TCP READ has no asynchronous data progress on the target. Withholding target
    // progress keeps the request pending. cuda_copy satisfies CUDA-enabled UCX validation.
    env.addVar("UCX_TLS", "tcp,cuda_copy");
    env.addVar("NIXL_PLUGIN_DIR", std::string(BUILD_DIR) + "/src/plugins/ucx");
    const gtest::LogIgnoreGuard ib_warning(
        "IB device\\(s\\) were detected, but accelerated IB support was not found");
    const gtest::LogIgnoreGuard efa_warning(
        "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
    const gtest::LogIgnoreGuard mock_plugin_probe(
        "Plugin file does not exist: .*/test/gtest/mocks/libplugin_UCX.so");

    constexpr size_t chunk_size = 4096;
    const std::vector<char> source(2 * chunk_size, 'x');
    std::vector<char> destination(2 * chunk_size, 0);
    nixlAgentConfig cfg;
    cfg.useProgThread = false;
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    nixlAgent target("release_target", cfg);
    nixlAgent initiator("release_initiator", cfg);
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(target.createBackend("UCX", {}, backend), NIXL_SUCCESS);
    const nixl_b_params_t params{{"num_threads", std::to_string(GetParam())},
                                 {"split_batch_size", "0"}};
    ASSERT_EQ(initiator.createBackend("UCX", params, backend), NIXL_SUCCESS);

    const auto source_addr = reinterpret_cast<uintptr_t>(source.data());
    const auto destination_addr = reinterpret_cast<uintptr_t>(destination.data());
    nixl_reg_dlist_t source_regs(DRAM_SEG), destination_regs(DRAM_SEG);
    source_regs.addDesc(nixlBlobDesc(source_addr, source.size(), 0));
    destination_regs.addDesc(nixlBlobDesc(destination_addr, destination.size(), 0));
    ASSERT_EQ(target.registerMem(source_regs), NIXL_SUCCESS);
    ASSERT_EQ(initiator.registerMem(destination_regs), NIXL_SUCCESS);
    std::string metadata, target_name;
    ASSERT_EQ(target.getLocalMD(metadata), NIXL_SUCCESS);
    ASSERT_EQ(initiator.loadRemoteMD(metadata, target_name), NIXL_SUCCESS);

    nixl_xfer_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
    for (size_t i = 0; i < 2; ++i) {
        local.addDesc(nixlBasicDesc(destination_addr + i * chunk_size, chunk_size, 0));
        remote.addDesc(nixlBasicDesc(source_addr + i * chunk_size, chunk_size, 0));
    }
    nixlXferReqH *request = nullptr;
    // A prepared request that was never posted must remain immediately releasable.
    ASSERT_EQ(initiator.createXferReq(NIXL_READ, local, remote, target_name, request),
              NIXL_SUCCESS);
    ASSERT_EQ(initiator.releaseXferReq(request), NIXL_SUCCESS);
    ASSERT_EQ(initiator.createXferReq(NIXL_READ, local, remote, target_name, request),
              NIXL_SUCCESS);
    EXPECT_EQ(initiator.postXferReq(request), NIXL_IN_PROG);

    // Repeated refusals must preserve ownership without logging errors.
    for (size_t attempt = 0; attempt < 3; ++attempt) {
        const nixl_status_t release_status = initiator.releaseXferReq(request);
        EXPECT_EQ(release_status, NIXL_ERR_REPOST_ACTIVE);
        if (release_status == NIXL_SUCCESS) {
            // A regressed implementation deleted request. Never poll that freed handle.
            return;
        }
    }
    // Continue to drain even if an intermediate expectation fails.
    EXPECT_EQ(initiator.getXferStatus(request), NIXL_IN_PROG);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    nixl_notifs_t notifs;
    nixl_status_t status = NIXL_IN_PROG;
    while ((status == NIXL_IN_PROG) && (std::chrono::steady_clock::now() < deadline)) {
        EXPECT_EQ(target.getNotifs(notifs), NIXL_SUCCESS);
        status = initiator.getXferStatus(request);
    }
    ASSERT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(destination, source);
    EXPECT_EQ(initiator.releaseXferReq(request), NIXL_SUCCESS);
    EXPECT_EQ(initiator.deregisterMem(destination_regs), NIXL_SUCCESS);
    EXPECT_EQ(target.deregisterMem(source_regs), NIXL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(SingleAndComposite, UcxRequestRelease, testing::Values(0, 2));

} // namespace
