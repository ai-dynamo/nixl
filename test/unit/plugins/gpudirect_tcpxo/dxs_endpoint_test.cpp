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

#include <cstdint>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dxs_endpoint.h"
#ifndef TCPXO_STUB_RXDM_DXS
#include "dxs/client/dxs-client-interface.h"
#else
#include "rxdm_dxs_stub.h"
#endif
#include "nixl_cuda/cuda_common.h"
#include "tcpxo_common.h"
#include "test_common.h"

namespace tcpxo {
namespace {

    using ::testing::_;
    using ::testing::ByMove;
    using ::testing::Return;
    using ::testing::Test;

    static constexpr uint8_t kDefaultFastrakIdx = 0;

    static constexpr absl::string_view kDefaultNicName = "lo";
    static constexpr absl::string_view kDefaultNicPciPath = "/sys/class/net/lo/device";

    static constexpr uint64_t kMaxNumFlowsPerDxsConn = 8;
    static constexpr uint64_t kDefaultDxsListenTimeoutMs = 1000; // 1 second

    class MockDxsEndpoint : public DxsEndpoint {
    public:
        MockDxsEndpoint(absl::string_view nic_dev_name,
                        absl::string_view nic_pci_path,
                        uint8_t fastrak_idx,
                        GpuDev &&gpu_dev)
            : DxsEndpoint(nic_dev_name, nic_pci_path, fastrak_idx, std::move(gpu_dev)) {}

        MOCK_METHOD(absl::StatusOr<DxsConnection>, Listen, (uint64_t, absl::Duration), (override));
    };

    class DxsEndpointTest : public Test {
    protected:
        void
        SetUp() override {
            endpoint_ = std::make_unique<MockDxsEndpoint>(
                kDefaultNicName, kDefaultNicPciPath, kDefaultFastrakIdx, GpuDev{});
        }

        DxsConnection
        Listen(MockDxsEndpoint *endpoint,
               uint64_t max_num_flows_per_dxs_conn,
               absl::Duration dxs_listen_timeout_ms) {
            absl::StatusOr<DxsConnection> dxs_conn =
                endpoint->Listen(max_num_flows_per_dxs_conn, dxs_listen_timeout_ms);
            if (!dxs_conn.ok()) {
                ADD_FAILURE() << "Listen failed: " << dxs_conn.status();
                return DxsConnection({});
            }
            return *std::move(dxs_conn);
        }

        std::unique_ptr<MockDxsEndpoint> endpoint_;
    };

    TEST_F(DxsEndpointTest, ListenOnLocalEndpoint) {
        std::vector<DxsFlow> mock_flows;
        for (uint64_t i = 0; i < kMaxNumFlowsPerDxsConn; ++i) {
            mock_flows.push_back(DxsFlow{
                .listen_socket = std::make_unique<test::MockListenSocket>(
                    test::kDefaultListenSocketPort, test::kDefaultNicAddr),
            });
        }
        DxsConnection mock_conn(std::move(mock_flows));

        EXPECT_CALL(*endpoint_, Listen(_, _)).WillOnce(Return(ByMove(std::move(mock_conn))));

        auto conn = Listen(endpoint_.get(),
                           kMaxNumFlowsPerDxsConn,
                           absl::Milliseconds(kDefaultDxsListenTimeoutMs));
        EXPECT_EQ(conn.flows().size(), kMaxNumFlowsPerDxsConn);
        for (const DxsFlow &flow : conn.flows()) {
            ASSERT_TRUE(flow.listen_socket);
            ASSERT_TRUE(flow.listen_socket->SocketReady().has_value());
            EXPECT_TRUE(flow.listen_socket->SocketReady().value().ok());
            EXPECT_FALSE(flow.recv_socket);
            EXPECT_FALSE(flow.send_socket);
        }
        EXPECT_EQ(conn.last_flow_used(), 0);
        EXPECT_NE(conn.connection_id(), ConnectionTraceId::kInvalid);
    }

    TEST(DxsEndpointManagerTest, SetupAndFetchCudaMapping) {
#ifdef HAVE_CUDA
        ASSERT_TRUE(InitCuda().ok()) << "Failed to initialize CUDA";
#endif

        // In the stub case or on the VM, it will setup mock or real endpoints.
        absl::StatusOr<std::unique_ptr<DxsEndpointManager>> manager_or =
            DxsEndpointManager::InitializeNetIfs(false, "", "", "", "", "eth0", false, false, "");

        ASSERT_TRUE(manager_or.ok())
            << "Failed to initialize DxsEndpointManager: " << manager_or.status();
        auto manager = std::move(*manager_or);
        ASSERT_FALSE(manager->endpoints().empty());

#ifdef HAVE_CUDA
        auto num_devices_or = GetDeviceCount();
        ASSERT_TRUE(num_devices_or.ok())
            << "Failed to get CUDA device count: " << num_devices_or.status();
        int device_count = *num_devices_or;

        for (int cuda_dev_id = 0; cuda_dev_id < device_count; ++cuda_dev_id) {
            auto fastrak_idx_or = manager->GetFastrakIdxFromDevId(cuda_dev_id);
            ASSERT_TRUE(fastrak_idx_or.ok())
                << "Failed to get FasTrak index for CUDA dev ID: " << cuda_dev_id;
            EXPECT_GE(*fastrak_idx_or, 0);
        }
#else
        for (const auto &endpoint : manager->endpoints()) {
            int expected_fastrak_idx = endpoint->fastrak_idx();
            auto fastrak_idx_or = manager->GetFastrakIdxFromDevId(expected_fastrak_idx);
            ASSERT_TRUE(fastrak_idx_or.ok())
                << "Failed to get FasTrak index for mock CUDA dev ID: " << expected_fastrak_idx;
            EXPECT_EQ(*fastrak_idx_or, expected_fastrak_idx);
        }
#endif
    }

} // namespace
} // namespace tcpxo

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
