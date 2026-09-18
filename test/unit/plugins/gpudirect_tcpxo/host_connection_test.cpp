/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nixl_types.h"

#include "control_channel.h"
#include "control_channel.pb.h"
#include "dxs_endpoint.h"
#include "host_connection.h"
#include "mpmc_queue.h"
#include "params.h"
#include "tcpxo_common.h"
#include "test_common.h"

#ifndef TCPXO_STUB_RXDM_DXS
#include "dxs/client/dxs-client-interface.h"
#else
#include "rxdm_dxs_stub.h"
#endif

namespace tcpxo {
namespace {
    using ::testing::Return;
    using ::testing::UnorderedElementsAre;

    static constexpr char kDefaultNicDevName[] = "lo";

    static constexpr absl::Duration kConnectionReadyTimeout = absl::Seconds(5);
    static constexpr absl::Duration kPollInterval = absl::Milliseconds(10);

    class MockHostConnection : public HostConnection {
    public:
        MockHostConnection(
            std::string remote_name,
            PeerHandle remote_handle,
            ControlChannel &channel,
            absl::flat_hash_map<EndpointPair, EndpointConnection> &&connection_map,
            MPMCQueue<std::pair<std::string, std::string>> &pending_notifs,
            HostConnectionTaskCallback host_connection_task_cb = []() {})
            : HostConnection(remote_name,
                             remote_handle,
                             channel,
                             std::move(connection_map),
                             GetUnsetParams(),
                             pending_notifs,
                             std::move(host_connection_task_cb)) {}

        MOCK_METHOD(bool,
                    EstablishEndpointConnections,
                    (const absl::flat_hash_set<EndpointPair> &),
                    (override));
    };

    class MockDxsEndpoint : public DxsEndpoint {
    public:
        MockDxsEndpoint(absl::string_view nic_dev_name,
                        absl::string_view nic_pci_path,
                        uint8_t fastrak_idx,
                        GpuDev &&gpu_dev)
            : DxsEndpoint(nic_dev_name, nic_pci_path, fastrak_idx, std::move(gpu_dev)) {}

        MOCK_METHOD(absl::StatusOr<DxsConnection>,
                    Listen,
                    (uint64_t max_num_flows_per_dxs_conn, absl::Duration dxs_listen_timeout_ms),
                    (override));
    };

    class HostConnectionTest : public ::testing::Test {
    protected:
        void
        SetUp() override {
            initiator_control_channel_ = test::MakeControlChannel();

            ConnectionCallback target_connection_callback =
                [this](PeerHandle handle,
                       const AgentAddress &addr,
                       const AgentAddress &service_add,
                       const std::string &agent_name,
                       const DxsAddressExchangeMessage & /* unused */) {
                    auto mock_connection_map =
                        absl::flat_hash_map<EndpointPair, EndpointConnection>();
                    target_ = std::make_unique<HostConnection>("test_remote_2",
                                                               handle,
                                                               *target_control_channel_,
                                                               std::move(mock_connection_map),
                                                               GetUnsetParams(),
                                                               target_pending_notifs_,
                                                               []() {});
                };
            NotificationCallback target_notification_callback =
                [this](
                    PeerHandle handle, const AgentAddress &src, WorkloadNotificationMessage &&msg) {
                    target_pending_notifs_.Enqueue({src.ip(), msg.message()});
                };

            target_control_channel_ = test::MakeControlChannel(
                std::move(target_connection_callback),
                [this](PeerHandle handle,
                       const AgentAddress &socket_addr,
                       const AgentAddress &service_addr) {
                    if (optional_disconnection_callback_) {
                        optional_disconnection_callback_(handle, socket_addr, service_addr);
                    }
                },
                std::move(target_notification_callback));

            ASSERT_TRUE(initiator_control_channel_->Listen().ok());
            ASSERT_TRUE(target_control_channel_->Listen().ok());

            static constexpr char kInitiatorAgentName[] = "test_remote_1";
            auto handle_or_status = initiator_control_channel_->Connect(
                target_control_channel_->GetServiceAddress(), kInitiatorAgentName, {});
            ASSERT_TRUE(handle_or_status.ok());
            auto mock_connection_map = absl::flat_hash_map<EndpointPair, EndpointConnection>();
            intiator_ = std::make_unique<HostConnection>(kInitiatorAgentName,
                                                         *handle_or_status,
                                                         *initiator_control_channel_,
                                                         std::move(mock_connection_map),
                                                         GetUnsetParams(),
                                                         initiator_pending_notifs_,
                                                         []() {});
        }

        void
        WaitForConnectionReady(HostConnection &conn) {
            const auto start_wait = absl::Now();
            bool connected = false;
            while (absl::Now() - start_wait < kConnectionReadyTimeout) {
                const auto ready = conn.IsConnectionReady();
                if (ready.ok() && *ready) {
                    connected = true;
                    break;
                }
                absl::SleepFor(kPollInterval);
            }
            ASSERT_TRUE(connected);
        }

        DisconnectionCallback optional_disconnection_callback_{nullptr};

        MPMCQueue<std::pair<std::string, std::string>> initiator_pending_notifs_;
        MPMCQueue<std::pair<std::string, std::string>> target_pending_notifs_;

        std::unique_ptr<ControlChannel> initiator_control_channel_;
        std::unique_ptr<ControlChannel> target_control_channel_;

        std::unique_ptr<HostConnection> intiator_;
        std::unique_ptr<HostConnection> target_;
    };

    TEST_F(HostConnectionTest, ConnectionWorks) {
        WaitForConnectionReady(*intiator_);
    }

    TEST_F(HostConnectionTest, DisconnectionWorks) {
        WaitForConnectionReady(*intiator_);

        absl::Notification disconnected;
        optional_disconnection_callback_ = [&disconnected](PeerHandle /* unused */,
                                                           const AgentAddress & /* unused */,
                                                           const AgentAddress & /* unused */) {
            disconnected.Notify();
        };

        // speed up this test
        EXPECT_TRUE(initiator_control_channel_->Disconnect(intiator_->remote_handle()).ok());
        intiator_.reset();

        EXPECT_TRUE(disconnected.WaitForNotificationWithTimeout(test::kHeartbeatTimeout +
                                                                absl::Seconds(1)));
    }

    TEST_F(HostConnectionTest, SendNotification) {
        const std::string kExpectedMessage = "test message 1";

        // Send the notification now, even before we know the connection is ready
        EXPECT_TRUE(intiator_->SendNotification(kExpectedMessage).ok());

        const auto start_wait = absl::Now();
        while (absl::Now() - start_wait < kConnectionReadyTimeout) {
            if (target_pending_notifs_.size() > 0) {
                break;
            }
            absl::SleepFor(kPollInterval);
        }

        EXPECT_FALSE(target_pending_notifs_.size() == 0);
        auto msg_or_status = target_pending_notifs_.TryDequeue();
        ASSERT_TRUE(msg_or_status.ok());
        EXPECT_EQ(msg_or_status->second, kExpectedMessage);
    }

    TEST_F(HostConnectionTest, XferLifecycle) {
        WaitForConnectionReady(*intiator_);

        nixlTcpxoBackendReqH *handle = nullptr;
        EXPECT_EQ(intiator_->PrepXfer(NIXL_WRITE, {}, handle), absl::OkStatus());
        ASSERT_NE(handle, nullptr);

        EXPECT_EQ(intiator_->PostXfer(*handle), absl::OkStatus());
        EXPECT_EQ(intiator_->CheckXfer(*handle), NIXL_SUCCESS);

        EXPECT_EQ(intiator_->ReleaseReqH(*handle), NIXL_SUCCESS);

        // Verify operations fail properly for an untracked/invalid request handle
        nixlTcpxoBackendReqH dummy_handle(NIXL_WRITE, *intiator_, {});
        EXPECT_EQ(intiator_->PostXfer(dummy_handle).code(), absl::StatusCode::kNotFound);
        EXPECT_EQ(intiator_->CheckXfer(dummy_handle), NIXL_ERR_BACKEND);
    }

    TEST_F(HostConnectionTest, PostXferTriggersEstablishEndpointConnections) {
        static constexpr int kNumConns = 2;

        auto mock_target_channel = test::MakeControlChannel();
        ASSERT_TRUE(mock_target_channel->Listen().ok());

        auto mock_channel = test::MakeControlChannel();
        ASSERT_TRUE(mock_channel->Listen().ok());
        auto handle_or_status =
            mock_channel->Connect(mock_target_channel->GetServiceAddress(), "test_agent", {});
        ASSERT_TRUE(handle_or_status.ok());

        MPMCQueue<std::pair<std::string, std::string>> mock_notifs;

        std::vector<std::unique_ptr<DxsEndpoint>> mock_endpoints;
        absl::flat_hash_map<EndpointPair, EndpointConnection> mock_connection_map;
        for (uint8_t i = 0; i < kNumConns; ++i) {
            mock_endpoints.push_back(
                std::make_unique<MockDxsEndpoint>("lo", "0000:00:00.0", i, GpuDev{}));
        }
        for (auto i = 0u; i < kNumConns; ++i) {
            std::vector<DxsFlow> mock_flows;
            mock_flows.push_back(
                DxsFlow{.listen_socket = std::make_unique<test::MockListenSocket>(
                            test::kDefaultListenSocketPort, test::kDefaultNicAddr)});
            DxsConnection dummy_conn(std::move(mock_flows));
            mock_connection_map.insert(
                {EndpointPair{i, i},
                 EndpointConnection{*(mock_endpoints[i]), std::move(dummy_conn)}});
        }

        MockHostConnection mock_conn("test_remote_mock",
                                     *handle_or_status,
                                     *mock_channel,
                                     std::move(mock_connection_map),
                                     mock_notifs);

        std::vector<DxsOpParams> ops = {{{0, 0}, {0x1000, 100}, {0x2000, 100}, 1, 1, 0},
                                        {{1, 1}, {0x3000, 100}, {0x4000, 100}, 2, 2, 0},
                                        {{0, 0}, {0x5000, 100}, {0x6000, 100}, 3, 3, 0}};

        DxsAddressExchangeMessage dummy_msg;
        for (int i = 0; i < kNumConns; ++i) {
            auto *ep_pair = dummy_msg.add_endpoint_pairs();
            ep_pair->set_local_fastrak_idx(i);
            ep_pair->set_remote_fastrak_idx(i);
            auto *handle = ep_pair->add_listen_handles();
            handle->mutable_addr()->set_addr("127.0.0.1");
            handle->set_port(12345 + i);
        }
        EXPECT_EQ(mock_conn.ParseRemoteListenMap(dummy_msg), absl::OkStatus());

        nixlTcpxoBackendReqH *handle = nullptr;
        EXPECT_EQ(mock_conn.PrepXfer(NIXL_WRITE, std::move(ops), handle), absl::OkStatus());
        ASSERT_NE(handle, nullptr);

        const absl::flat_hash_set<EndpointPair> kEndpointsToEstablish = {{0, 0}, {1, 1}};
        EXPECT_CALL(mock_conn,
                    EstablishEndpointConnections(
                        UnorderedElementsAre(EndpointPair{0, 0}, EndpointPair{1, 1})))
            .WillOnce(Return(true));

        EXPECT_EQ(mock_conn.PostXfer(*handle), absl::OkStatus());
    }

} // namespace
} // namespace tcpxo

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
