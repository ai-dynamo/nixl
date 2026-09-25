/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
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

#include <cstdlib>
#include <memory>
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "backend_aux.h"
#include "common/nixl_log.h"
#include "gtest/gtest.h"
#include "nixl_types.h"
#include "nixl_descriptors.h"

#include "host_connection.h"
#include "params.h"
#include "tcpxo_backend.h"
#include "tcpxo_common.h"
#include "test_common.h"

namespace tcpxo {

class TcpxoValidationTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        nixlBackendInitParams init;
        nixl_b_params_t custom_params;

        custom_params["FASTRAK_CTRL_DEV"] = "lo";
        init.localAgent = "TestAgent";
        init.customParams = &custom_params;
        init.enableProgTh = false;
        init.pthrDelay = 0;

        engine_ = std::make_unique<nixlTcpxoEngine>(&init);

        for (size_t i = 0; i < kMaxGpuDevices; ++i) {
            engine_->endpoint_manager_->InjectDevIdMappingForTest(i, i);
        }
    }

    void
    TearDown() override {
        for (auto dev_id : injected_devices_) {
            auto &mem_device = engine_->mem_devices_[dev_id];
            absl::MutexLock lock(mem_device.reg_mutex);
            mem_device.cache.clear();
        }
    }

    void
    InjectFakeMemoryRegion(nixlMetaDesc &local_desc, dxs::Reg reg_handle = 0) {
        auto region_info = engine_->GetPageOrientedRegion(local_desc.addr, local_desc.len);
        // RETURN_IF_ERROR macro barfs on the rhs
        const auto fastrak_idx_or_status =
            engine_->endpoint_manager_->GetFastrakIdxFromDevId(local_desc.devId);
        if (!fastrak_idx_or_status.ok()) {
            ADD_FAILURE() << "Unable to get FasTrak index of device ID: " << local_desc.devId;
            return;
        }
        const auto fastrak_idx = *fastrak_idx_or_status;
        nixlMemDev &mem_device = engine_->mem_devices_[fastrak_idx];
        absl::MutexLock lock(mem_device.reg_mutex);

        auto mem_md = std::make_unique<nixlTcpxoLocalMemoryMetadata>(
            reg_handle,
            reinterpret_cast<void *>(region_info.first),
            region_info.second,
            /* dxs registration handle */ -1,
            fastrak_idx);
        local_desc.metadataP = mem_md.get();

        mem_device.cache.insert(
            {region_info,
             CacheValueType{std::move(mem_md), /* number of regions using this page */ 1}});
        injected_devices_.insert(fastrak_idx);
    }

    void
    ConnectFakeAgent(const std::string &remote_agent) {
        nixlTcpxoConnection conn_info;
        absl::MutexLock lock(engine_->remote_agents_mutex_);
        engine_->remote_agents_[remote_agent].info = conn_info;
        fake_channel_ = test::MakeControlChannel();
        engine_->remote_agents_[remote_agent].conn =
            std::make_unique<HostConnection>(remote_agent,
                                             static_cast<PeerHandle>(1),
                                             *fake_channel_,
                                             std::move(fake_connection_map_),
                                             engine_->params(),
                                             fake_notifs_,
                                             []() {});
    }

    absl::flat_hash_map<EndpointPair, EndpointConnection> fake_connection_map_;
    std::unique_ptr<ControlChannel> fake_channel_;
    MPMCQueue<std::pair<std::string, std::string>> fake_notifs_;
    absl::flat_hash_set<uint64_t> injected_devices_;
    std::unique_ptr<nixlTcpxoEngine> engine_;
};

TEST_F(TcpxoValidationTest, EmptyDlistsSucceeds) {
    nixl_meta_dlist_t local(VRAM_SEG);
    nixl_meta_dlist_t remote(VRAM_SEG);
    nixlBackendReqH *handle = nullptr;

    nixl_status_t status = engine_->prepXfer(NIXL_READ, local, remote, "RemoteAgent", handle);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_F(TcpxoValidationTest, InvalidOperationLengthFails) {
    nixl_meta_dlist_t local(VRAM_SEG);
    nixl_meta_dlist_t remote(VRAM_SEG);

    nixlMetaDesc local_desc(0x1000, 200, 0);
    nixlMetaDesc remote_desc(0x2000, 100, 0);

    local.addDesc(local_desc);
    remote.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_WRITE, local, remote, "RemoteAgent", handle);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

TEST_F(TcpxoValidationTest, DlistLengthMismatchFails) {
    nixl_meta_dlist_t local(VRAM_SEG);
    nixl_meta_dlist_t remote(VRAM_SEG);

    nixlMetaDesc local_desc(0, 100, 0);
    local.addDesc(local_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_READ, local, remote, "RemoteAgent", handle);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

TEST_F(TcpxoValidationTest, InvalidDeviceIDFails) {
    nixl_meta_dlist_t local(VRAM_SEG);
    nixl_meta_dlist_t remote(VRAM_SEG);

    nixlMetaDesc local_desc(0x1000, 100, kMaxGpuDevices);
    nixlMetaDesc remote_desc(0x2000, 100, 0);

    local.addDesc(local_desc);
    remote.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_READ, local, remote, "RemoteAgent", handle);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

TEST_F(TcpxoValidationTest, UnregisteredLocalMemoryFails) {
    nixl_meta_dlist_t local(VRAM_SEG);
    nixl_meta_dlist_t remote(VRAM_SEG);

    nixlMetaDesc local_desc(0x1000, 100, 0);
    nixlMetaDesc remote_desc(0x2000, 100, 0);

    local.addDesc(local_desc);
    remote.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_READ, local, remote, "RemoteAgent", handle);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

TEST_F(TcpxoValidationTest, RegisteredLocalMemoryPassesValidation) {
    nixl_meta_dlist_t local(VRAM_SEG);
    nixl_meta_dlist_t remote(VRAM_SEG);

    nixlMetaDesc local_desc(0x1000, 100, 0);
    nixlMetaDesc remote_desc(0x2000, 100, 0);
    auto remote_mem_md = std::make_unique<nixlTcpxoLocalMemoryMetadata>(
        0, reinterpret_cast<void *>(remote_desc.addr), remote_desc.len, -1, 0);
    remote_desc.metadataP = remote_mem_md.get();

    InjectFakeMemoryRegion(local_desc);

    local.addDesc(local_desc);
    remote.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_READ, local, remote, "RemoteAgent", handle);

    // We expect NIXL_ERR_NOT_FOUND because RemoteAgent is not connected, but this means it passed
    // the validation checks.
    EXPECT_EQ(status, NIXL_ERR_NOT_FOUND);
}

TEST_F(TcpxoValidationTest, PrepXferCalculatesCorrectParams) {
    ConnectFakeAgent("RemoteAgent");

    constexpr DxsOpParams expected_op_params{
        .endpoint_pair = {.local_fastrak_idx = 0, .remote_fastrak_idx = 0},
        .local_mr = {.addr = 0x1234, .len = 100},
        .remote_mr = {0x2000, 100},
        .local_reg_handle = 1,
        .remote_reg_handle = 2,
        /*.local_page_offset doesn't matter, we'll calculate it later*/};
    nixlMetaDesc local_desc(expected_op_params.local_mr.addr,
                            expected_op_params.local_mr.len,
                            expected_op_params.endpoint_pair.local_fastrak_idx);
    nixlMetaDesc remote_desc(expected_op_params.remote_mr.addr,
                             expected_op_params.remote_mr.len,
                             expected_op_params.endpoint_pair.remote_fastrak_idx);
    InjectFakeMemoryRegion(local_desc, expected_op_params.local_reg_handle);

    auto remote_mem_md = std::make_unique<nixlTcpxoLocalMemoryMetadata>(
        expected_op_params.remote_reg_handle,
        reinterpret_cast<void *>(expected_op_params.remote_mr.addr),
        expected_op_params.remote_mr.len,
        -1 /*dmabuf_fd, doesn't matter*/,
        expected_op_params.endpoint_pair.remote_fastrak_idx);
    remote_desc.metadataP = remote_mem_md.get();

    nixl_meta_dlist_t local_dlist(VRAM_SEG);
    nixl_meta_dlist_t remote_dlist(VRAM_SEG);
    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status =
        engine_->prepXfer(NIXL_READ, local_dlist, remote_dlist, "RemoteAgent", handle);
    EXPECT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto *req_handle = static_cast<nixlTcpxoBackendReqH *>(handle);
    const auto &ops = req_handle->ops();
    ASSERT_EQ(ops.size(), 1);

    const auto &op_params = ops[0].params();
    EXPECT_EQ(op_params.local_reg_handle, expected_op_params.local_reg_handle);
    EXPECT_EQ(op_params.remote_reg_handle, expected_op_params.remote_reg_handle);

    // Intentionally replicating the math in GetPageOrientedRegion, instead of making it public
    size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    uintptr_t expected_page_start = expected_op_params.local_mr.addr & -page_size;
    size_t expected_offset = expected_op_params.local_mr.addr - expected_page_start;
    EXPECT_EQ(op_params.local_page_offset, expected_offset);

    EXPECT_EQ(op_params.local_mr.addr, expected_op_params.local_mr.addr);
    EXPECT_EQ(op_params.local_mr.len, expected_op_params.local_mr.len);
    EXPECT_EQ(op_params.remote_mr.addr, expected_op_params.remote_mr.addr);
    EXPECT_EQ(op_params.remote_mr.len, expected_op_params.remote_mr.len);

    EXPECT_EQ(op_params.endpoint_pair.local_fastrak_idx,
              expected_op_params.endpoint_pair.local_fastrak_idx);
    EXPECT_EQ(op_params.endpoint_pair.remote_fastrak_idx,
              expected_op_params.endpoint_pair.remote_fastrak_idx);

    engine_->releaseReqH(handle);
}

} // namespace tcpxo

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
