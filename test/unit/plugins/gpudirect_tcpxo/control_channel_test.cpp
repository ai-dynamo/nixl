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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <google/protobuf/repeated_ptr_field.h>
#include <atomic>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "control_channel.h"
#include "control_channel.pb.h"
#include "test_common.h"

namespace tcpxo {
namespace {

    // Default timeouts for waiting on condition variables or events
    static constexpr absl::Duration kDefaultNotificationTimeout = absl::Seconds(2);
    static constexpr absl::Duration kConnectionReadyTimeout = absl::Seconds(5);
    static constexpr absl::Duration kMaxMessageTransmissionTimeout = absl::Seconds(60);
    static constexpr absl::Duration kDisconnectDetectionTimeout = absl::Seconds(10);
    static constexpr absl::Duration kPollInterval = absl::Milliseconds(10);

    static constexpr size_t kRandStringMaxLen = 10000;
    static constexpr int kRandDelayMaxMs = 10;
    static constexpr char kDummyAgentName[] = "test_agent";

    TEST(AgentAddressTest, OstreamOperator) {
        IPAddress ip_addr;
        ip_addr.set_addr("192.168.1.1");
        AgentAddress addr(ip_addr, 1234);
        std::stringstream ss;
        ss << addr;
        EXPECT_EQ(ss.str(), "192.168.1.1:1234");
    }

    TEST(PeerHandleTest, OstreamOperator) {
        auto handle = static_cast<PeerHandle>(1);
        std::stringstream ss;
        ss << handle;
        EXPECT_EQ(ss.str(), "PeerHandle:1");
    }

    class ControlChannelTest : public ::testing::Test {
    protected:
        void
        SetUp() override {
            channel1_ = test::MakeControlChannel();
            channel2_ = test::MakeControlChannel();
        }

        void
        WaitForConnectionReady(ControlChannel &channel, PeerHandle handle) {
            auto start_wait = absl::Now();
            bool connected = false;
            while (absl::Now() - start_wait < kConnectionReadyTimeout) {
                auto ready = channel.IsConnectionReady(handle);
                if (ready.ok() && *ready) {
                    connected = true;
                    break;
                }
                absl::SleepFor(kPollInterval);
            }
            ASSERT_TRUE(connected);
        }

        AgentAddress
        Listen(ControlChannel &channel) {
            auto addr_or_status = channel.Listen();
            if (!addr_or_status.ok()) {
                ADD_FAILURE() << "Listen failed: " << addr_or_status.status();
                return {};
            }
            return *addr_or_status;
        }

        PeerHandle
        Connect(ControlChannel &channel,
                const AgentAddress &addr,
                std::vector<DxsEndpointListenInfo> endpoint_listen_infos = {}) {
            auto handle = channel.Connect(addr, kDummyAgentName, std::move(endpoint_listen_infos));
            if (!handle.ok()) {
                ADD_FAILURE() << "Connect failed: " << handle.status();
                return static_cast<PeerHandle>(-1);
            }
            return *handle;
        }

        PeerHandle
        ConnectAndWait(ControlChannel &channel,
                       const AgentAddress &addr,
                       std::vector<DxsEndpointListenInfo> endpoint_listen_infos = {}) {
            PeerHandle handle = Connect(channel, addr, std::move(endpoint_listen_infos));
            WaitForConnectionReady(channel, handle);
            return handle;
        }

        struct ConnectionCallbackTracker {
            absl::Notification connected;
            AgentAddress peer_addr;
            PeerHandle peer_handle;
            ConnectionCallback absl_nonnull cb;
        };

        std::unique_ptr<ConnectionCallbackTracker>
        MakeConnectionCallbackTracker() {
            auto tracker = std::make_unique<ConnectionCallbackTracker>();
            tracker->cb = [c = tracker.get()](PeerHandle handle,
                                              const AgentAddress & /* unused */,
                                              const AgentAddress &service_addr,
                                              const std::string & /* unused */,
                                              const DxsAddressExchangeMessage & /* unused */) {
                c->peer_addr = service_addr;
                c->peer_handle = handle;
                c->connected.Notify();
            };
            return tracker;
        }

        std::unique_ptr<ControlChannel> channel1_;
        std::unique_ptr<ControlChannel> channel2_;
    };

    TEST_F(ControlChannelTest, ListenOnLoopback) {
        auto addr = Listen(*channel1_);

        EXPECT_EQ(addr.ip(), "127.0.0.1");
        EXPECT_GT(addr.port(), 0);
    }

    TEST_F(ControlChannelTest, ConnectAndDisconnect) {
        Listen(*channel1_);
        auto addr2 = Listen(*channel2_);
        auto handle = ConnectAndWait(*channel1_, addr2);

        EXPECT_TRUE(channel1_->Disconnect(handle).ok());
    }

    TEST_F(ControlChannelTest, SendIdentityMessage) {
        constexpr size_t num_gpus_per_local_node = 2;
        constexpr size_t num_gpus_per_remote_node = 2;
        constexpr size_t num_flows_per_dxs_conn = 1;

        constexpr absl::string_view service_ip = "192.168.1.1";
        constexpr uint16_t service_port = 1234;

        absl::Notification received;
        size_t received_endpoint_listen_infos_size = 0;
        std::vector<size_t> received_addrs_size;
        ConnectionCallback cb = [&received,
                                 &received_endpoint_listen_infos_size,
                                 &received_addrs_size](PeerHandle /* unused */,
                                                       const AgentAddress & /* unused */,
                                                       const AgentAddress & /* unused */,
                                                       const std::string & /* unused */,
                                                       const DxsAddressExchangeMessage &dxs_msg) {
            received_endpoint_listen_infos_size = dxs_msg.endpoint_pairs_size();
            for (const auto &listen_info : dxs_msg.endpoint_pairs()) {
                received_addrs_size.push_back(listen_info.listen_handles_size());
            }
            received.Notify();
        };

        std::vector<DxsEndpointListenInfo> endpoint_listen_infos;
        Message msg;
        auto *identity_msg = msg.mutable_identity();
        identity_msg->mutable_addr()->set_addr(service_ip);
        identity_msg->set_port(service_port);
        auto *dxs_msg = identity_msg->mutable_dxs();
        for (size_t local_fastrak_idx = 0; local_fastrak_idx < num_gpus_per_local_node;
             ++local_fastrak_idx) {
            for (size_t peer_rank = 0; peer_rank < num_gpus_per_remote_node; ++peer_rank) {
                auto *endpoints = dxs_msg->add_endpoint_pairs();
                endpoints->set_local_fastrak_idx(local_fastrak_idx);
                endpoints->set_remote_fastrak_idx(peer_rank);
                for (size_t flow = 0; flow < num_flows_per_dxs_conn; ++flow) {
                    auto port = (1000 << peer_rank) + flow;
                    auto *dxs_addr = endpoints->add_listen_handles();
                    dxs_addr->mutable_addr()->set_addr(service_ip);
                    dxs_addr->set_port(port);
                }
                endpoint_listen_infos.push_back(*endpoints);
            }
        }

        channel2_ = test::MakeControlChannel(std::move(cb));
        Listen(*channel1_);
        auto addr2 = Listen(*channel2_);
        auto handle = ConnectAndWait(*channel1_, addr2, std::move(endpoint_listen_infos));

        ASSERT_TRUE(channel1_->ScheduleSend(handle, std::move(msg)).ok());

        ASSERT_TRUE(received.WaitForNotificationWithTimeout(kDefaultNotificationTimeout));
        EXPECT_EQ(received_endpoint_listen_infos_size,
                  num_gpus_per_local_node * num_gpus_per_remote_node);
        ASSERT_EQ(received_addrs_size.size(), received_endpoint_listen_infos_size);
        for (const auto &addrs_size : received_addrs_size) {
            EXPECT_EQ(addrs_size, num_flows_per_dxs_conn);
        }
    }

    TEST_F(ControlChannelTest, SendNotification) {
        absl::Notification received;
        std::string received_msg;
        NotificationCallback cb = [&received_msg, &received](PeerHandle /* unused */,
                                                             const AgentAddress & /* unused */,
                                                             WorkloadNotificationMessage &&msg) {
            received_msg = msg.message();
            received.Notify();
        };

        channel2_ = test::MakeControlChannel([](...) {}, [](...) {}, std::move(cb));
        Listen(*channel1_);
        auto addr2 = Listen(*channel2_);
        auto handle = ConnectAndWait(*channel1_, addr2);

        Message msg;
        const std::string message_text = "Hello World";
        msg.mutable_notif()->set_message(message_text);
        ASSERT_TRUE(channel1_->ScheduleSend(handle, std::move(msg)).ok());

        ASSERT_TRUE(received.WaitForNotificationWithTimeout(kDefaultNotificationTimeout));
        EXPECT_EQ(received_msg, message_text);
    }

    TEST_F(ControlChannelTest, SendDxsAddressExchange) {
        constexpr size_t num_gpus_per_local_node = 2;
        constexpr size_t num_gpus_per_remote_node = 2;
        constexpr size_t num_flows_per_dxs_conn = 1;

        constexpr absl::string_view local_ip = "192.168.1.1";

        absl::Notification received;
        size_t received_endpoint_listen_infos_size = 0;
        std::vector<size_t> received_addrs_size;
        DxsAddressExchangeCallback cb =
            [&received, &received_endpoint_listen_infos_size, &received_addrs_size](
                PeerHandle /* unused */,
                const AgentAddress & /* unused */,
                const DxsAddressExchangeMessage &msg) {
                received_endpoint_listen_infos_size = msg.endpoint_pairs_size();
                for (const auto &listen_info : msg.endpoint_pairs()) {
                    received_addrs_size.push_back(listen_info.listen_handles_size());
                }
                received.Notify();
            };

        std::vector<DxsEndpointListenInfo> endpoint_listen_infos;
        Message msg;
        auto *dxs_msg = msg.mutable_dxs();
        for (size_t local_fastrak_idx = 0; local_fastrak_idx < num_gpus_per_local_node;
             ++local_fastrak_idx) {
            for (size_t peer_rank = 0; peer_rank < num_gpus_per_remote_node; ++peer_rank) {
                auto *endpoints = dxs_msg->add_endpoint_pairs();
                endpoints->set_local_fastrak_idx(local_fastrak_idx);
                endpoints->set_remote_fastrak_idx(peer_rank);
                for (size_t flow = 0; flow < num_flows_per_dxs_conn; ++flow) {
                    auto port = (1000 << peer_rank) + flow;
                    auto *dxs_addr = endpoints->add_listen_handles();
                    dxs_addr->mutable_addr()->set_addr(local_ip);
                    dxs_addr->set_port(port);
                }
                endpoint_listen_infos.push_back(*endpoints);
            }
        }

        channel2_ = test::MakeControlChannel([](...) {}, [](...) {}, [](...) {}, std::move(cb));
        Listen(*channel1_);
        auto addr2 = Listen(*channel2_);
        auto handle = ConnectAndWait(*channel1_, addr2, std::move(endpoint_listen_infos));

        ASSERT_TRUE(channel1_->ScheduleSend(handle, std::move(msg)).ok());

        ASSERT_TRUE(received.WaitForNotificationWithTimeout(kDefaultNotificationTimeout));
        EXPECT_EQ(received_endpoint_listen_infos_size,
                  num_gpus_per_local_node * num_gpus_per_remote_node);
        ASSERT_EQ(received_addrs_size.size(), received_endpoint_listen_infos_size);
        for (const auto &addrs_size : received_addrs_size) {
            EXPECT_EQ(addrs_size, num_flows_per_dxs_conn);
        }
    }

    TEST_F(ControlChannelTest, CheckSendStatus) {
        Listen(*channel1_);
        auto addr2 = Listen(*channel2_);
        auto handle = ConnectAndWait(*channel1_, addr2);

        Message msg;
        msg.mutable_notif()->set_message("Status Test");
        auto send_id_or = channel1_->ScheduleSend(handle, std::move(msg));
        ASSERT_TRUE(send_id_or.ok());
        SendId send_id = *send_id_or;

        SendStatus status = channel1_->CheckSendStatus(send_id);
        EXPECT_TRUE(status == SendStatus::kPending || status == SendStatus::kCompleted);

        auto start_time = absl::Now();
        while (status != SendStatus::kCompleted &&
               absl::Now() - start_time < kDefaultNotificationTimeout) {
            status = channel1_->CheckSendStatus(send_id);
            if (status == SendStatus::kCompleted) {
                break;
            }
            absl::SleepFor(kPollInterval);
        }

        EXPECT_EQ(status, SendStatus::kCompleted);
    }

    TEST_F(ControlChannelTest, SendMixOfNotifications) {
        static constexpr int kNumNotifications = 500;
        std::atomic<int> received_count{0};
        std::vector<std::string> sent_messages;
        std::vector<std::string> received_messages;
        absl::Mutex received_messages_mutex;

        NotificationCallback cb = [&received_messages_mutex, &received_messages, &received_count](
                                      PeerHandle /* unused */,
                                      const AgentAddress & /* unused */,
                                      WorkloadNotificationMessage &&msg) {
            absl::MutexLock lock(received_messages_mutex);
            received_messages.push_back(msg.message());
            received_count++;
        };
        channel2_ = test::MakeControlChannel([](...) {}, [](...) {}, std::move(cb));
        Listen(*channel1_);
        auto addr2 = Listen(*channel2_);
        auto handle = ConnectAndWait(*channel1_, addr2);

        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<size_t> dist(1, kRandStringMaxLen);
        std::uniform_int_distribution<int> delay_dist(0, kRandDelayMaxMs);
        for (int i = 0; i < kNumNotifications; ++i) {
            Message msg;
            size_t len = dist(gen);
            std::string content(len, 'A' + (i % 26));
            sent_messages.push_back(content);
            msg.mutable_notif()->set_message(content);

            ASSERT_TRUE(channel1_->ScheduleSend(handle, std::move(msg)).ok());

            absl::SleepFor(absl::Milliseconds(delay_dist(gen)));
        }

        // Wait for all notifications to be received
        auto start_time = absl::Now();
        while (received_count < kNumNotifications &&
               absl::Now() - start_time < kDisconnectDetectionTimeout) {
            absl::SleepFor(kPollInterval);
        }
        EXPECT_EQ(received_count, kNumNotifications);

        absl::MutexLock lock(received_messages_mutex);
        // We can't guarantee order if there were multiple workers, but here we have 1 worker.
        // The implementation uses a single thread for epoll and one worker, so order should be
        // preserved.
        ASSERT_EQ(sent_messages.size(), received_messages.size());
        for (size_t i = 0; i < sent_messages.size(); ++i) {
            EXPECT_EQ(sent_messages[i], received_messages[i]) << "Mismatch at index " << i;
        }
    }

    TEST_F(ControlChannelTest, BidirectionalSendMixOfNotifications) {
        static constexpr int kNumNotifications = 500;
        std::atomic<int> channel1_recv_msg_cnt{0};
        std::atomic<int> channel2_recv_msg_cnt{0};
        NotificationCallback cb1 =
            [&channel1_recv_msg_cnt](PeerHandle /* unused */,
                                     const AgentAddress & /* unused */,
                                     WorkloadNotificationMessage && /* unused */) {
                channel1_recv_msg_cnt++;
            };
        channel1_ = test::MakeControlChannel([](...) {}, [](...) {}, std::move(cb1));

        NotificationCallback cb2 =
            [&channel2_recv_msg_cnt](PeerHandle /* unused */,
                                     const AgentAddress & /* unused */,
                                     WorkloadNotificationMessage && /* unused */) {
                channel2_recv_msg_cnt++;
            };
        auto conn_cb_tracker = MakeConnectionCallbackTracker();
        channel2_ =
            test::MakeControlChannel(std::move(conn_cb_tracker->cb), [](...) {}, std::move(cb2));

        auto channel1_listen_addr = Listen(*channel1_);
        auto channel2_listen_addr = Listen(*channel2_);
        auto handle = Connect(*channel1_, channel2_listen_addr);
        ASSERT_TRUE(
            conn_cb_tracker->connected.WaitForNotificationWithTimeout(kConnectionReadyTimeout));
        EXPECT_EQ(conn_cb_tracker->peer_addr, channel1_listen_addr);
        auto channel2_ready = channel2_->IsConnectionReady(conn_cb_tracker->peer_handle);
        ASSERT_TRUE(channel2_ready.ok() && *channel2_ready);

        auto send_func = [](ControlChannel &chan, PeerHandle dest, int count, uint32_t seed) {
            std::mt19937 gen(seed);
            std::uniform_int_distribution<size_t> dist(1, kRandStringMaxLen);
            std::uniform_int_distribution<int> delay_dist(0, kRandDelayMaxMs);
            for (int i = 0; i < count; ++i) {
                Message msg;
                size_t len = dist(gen);
                msg.mutable_notif()->set_message(std::string(len, 'B'));
                auto status = chan.ScheduleSend(dest, std::move(msg));
                EXPECT_TRUE(status.ok()) << status.status();
                absl::SleepFor(absl::Milliseconds(delay_dist(gen)));
            }
        };

        std::thread channel1_send_thread(
            send_func, std::ref(*channel1_), handle, kNumNotifications, 42);
        std::thread channel2_send_thread(
            send_func, std::ref(*channel2_), conn_cb_tracker->peer_handle, kNumNotifications, 43);

        channel1_send_thread.join();
        channel2_send_thread.join();

        // Wait for all notifications
        auto start_time = absl::Now();
        while ((channel1_recv_msg_cnt < kNumNotifications ||
                channel2_recv_msg_cnt < kNumNotifications) &&
               absl::Now() - start_time < kMaxMessageTransmissionTimeout) {
            absl::SleepFor(kPollInterval);
        }

        EXPECT_EQ(channel1_recv_msg_cnt, kNumNotifications);
        EXPECT_EQ(channel2_recv_msg_cnt, kNumNotifications);
    }

    TEST_F(ControlChannelTest, FullyConnectedMesh) {
        static constexpr int kNumChannels = 96;
        static constexpr int kNumMessagesPerPeer = 5;
        static constexpr int kNumConnectedPeers = kNumChannels - 1;
        static constexpr int kNumMessagesPerChannel = kNumConnectedPeers * kNumMessagesPerPeer;

        struct ChannelState {
            std::unique_ptr<ControlChannel> channel;
            AgentAddress listen_addr;
            absl::Mutex mutex;
            std::vector<PeerHandle> peer_handles ABSL_GUARDED_BY(mutex);
            std::atomic<int> msgs_received{0};
        };

        std::vector<std::unique_ptr<ChannelState>> states;
        for (int i = 0; i < kNumChannels; ++i) {
            auto state = std::make_unique<ChannelState>();

            NotificationCallback notif_cb = [i,
                                             &states](PeerHandle /* unused */,
                                                      const AgentAddress & /* unused */,
                                                      WorkloadNotificationMessage && /* unused */) {
                states[i]->msgs_received++;
            };
            ConnectionCallback conn_cb = [i,
                                          &states](PeerHandle handle,
                                                   const AgentAddress & /* unused */,
                                                   const AgentAddress & /* unused */,
                                                   const std::string & /* unused */,
                                                   const DxsAddressExchangeMessage & /* unused */) {
                absl::MutexLock lock(states[i]->mutex);
                states[i]->peer_handles.push_back(handle);
            };

            state->channel =
                test::MakeControlChannel(std::move(conn_cb), [](...) {}, std::move(notif_cb));
            state->listen_addr = Listen(*state->channel);
            states.push_back(std::move(state));
        }

        // Connect in a cascade. Channel 1 makes 95 connections, channel 2 makes 94, channel 3
        // makes 93, and so on
        for (int i = 0; i < kNumChannels; ++i) {
            for (int j = i + 1; j < kNumChannels; ++j) {
                auto handle_or_status =
                    states[i]->channel->Connect(states[j]->listen_addr, kDummyAgentName, {});

                ASSERT_TRUE(handle_or_status.ok());

                absl::MutexLock lock(states[i]->mutex);
                states[i]->peer_handles.push_back(*handle_or_status);
            }
        }

        // Wait for all expected connections
        auto start_wait = absl::Now();
        bool all_connections_received = false;
        while (absl::Now() - start_wait < absl::Seconds(30)) {
            all_connections_received = true;
            for (int i = 0; i < kNumChannels; ++i) {
                absl::MutexLock lock(states[i]->mutex);
                if (states[i]->peer_handles.size() < kNumConnectedPeers) {
                    all_connections_received = false;
                    break;
                }
            }
            if (all_connections_received) {
                break;
            }
            absl::SleepFor(kPollInterval);
        }
        ASSERT_TRUE(all_connections_received) << "Failed to anticipate all connections in time.";

        // Ensure all connections are ready
        start_wait = absl::Now();
        bool all_ready = false;
        while (absl::Now() - start_wait < kDisconnectDetectionTimeout) {
            all_ready = true;
            for (int i = 0; i < kNumChannels; ++i) {
                std::vector<PeerHandle> handles;
                {
                    absl::MutexLock lock(states[i]->mutex);
                    handles = states[i]->peer_handles;
                }

                for (const auto &handle : handles) {
                    auto ready = states[i]->channel->IsConnectionReady(handle);
                    if (!ready.ok() || !*ready) {
                        all_ready = false;
                        break;
                    }
                }
                if (!all_ready) {
                    break;
                }
            }
            if (all_ready) {
                break;
            }
            absl::SleepFor(kPollInterval);
        }
        ASSERT_TRUE(all_ready) << "Connections failed to be ready in time.";

        std::string big_message = "DATA:" + std::string(100000, 'C'); // ~100KB message

        std::vector<std::thread> send_threads;
        for (int i = 0; i < kNumChannels; ++i) {
            send_threads.emplace_back([i, &states, &big_message]() {
                std::vector<PeerHandle> dests;
                {
                    absl::MutexLock lock(states[i]->mutex);
                    dests = states[i]->peer_handles;
                }

                for (const auto &dest : dests) {
                    for (int m = 0; m < kNumMessagesPerPeer; ++m) {
                        Message msg;
                        msg.mutable_notif()->set_message(big_message);
                        auto status = states[i]->channel->ScheduleSend(dest, std::move(msg));
                        EXPECT_TRUE(status.ok()) << status.status();
                    }
                }
            });
        }

        for (auto &t : send_threads) {
            t.join();
        }

        // Make sure everything has propagated
        start_wait = absl::Now();
        bool all_data_received = false;
        while (absl::Now() - start_wait < kMaxMessageTransmissionTimeout) {
            all_data_received = true;
            for (int i = 0; i < kNumChannels; ++i) {
                if (states[i]->msgs_received < kNumMessagesPerChannel) {
                    all_data_received = false;
                    break;
                }
            }
            if (all_data_received) {
                break;
            }
            absl::SleepFor(kPollInterval);
        }

        // Output details if failed
        if (!all_data_received) {
            for (int i = 0; i < kNumChannels; ++i) {
                if (states[i]->msgs_received < kNumMessagesPerChannel) {
                    LOG(INFO) << "Channel " << i << " only received " << states[i]->msgs_received
                              << " data messages.";
                }
            }
        }
        ASSERT_TRUE(all_data_received) << "Failed to receive all data messages in time.";
    }

    TEST_F(ControlChannelTest, SendMaxMessageSize) {
        absl::Notification received;
        size_t received_msg_size = 0;
        NotificationCallback cb = [&received_msg_size,
                                   &received](PeerHandle /* unused */,
                                              const AgentAddress & /* unused */,
                                              WorkloadNotificationMessage &&msg) {
            received_msg_size = msg.message().size();
            received.Notify();
        };

        channel2_ = test::MakeControlChannel([](...) {}, [](...) {}, std::move(cb));
        Listen(*channel1_);
        auto channel2_listen_addr = Listen(*channel2_);
        auto handle = ConnectAndWait(*channel1_, channel2_listen_addr);

        Message msg;
        msg.mutable_notif()->set_message(std::string(kMaxNotificiationSize, 'X'));
        // Ensure our message size doesn't overflow what the protocol claims to handle.
        EXPECT_LE(msg.ByteSizeLong(), kMaxSerializedProtobufSize);

        ASSERT_TRUE(channel1_->ScheduleSend(handle, std::move(msg)).ok());
        ASSERT_TRUE(received.WaitForNotificationWithTimeout(kMaxMessageTransmissionTimeout));
        EXPECT_EQ(received_msg_size, kMaxNotificiationSize);
    }

    TEST_F(ControlChannelTest, BidirectionalSendMaxMessageSize) {
        absl::Notification received1;
        size_t received_msg_size1 = 0;
        NotificationCallback cb1 = [&received_msg_size1,
                                    &received1](PeerHandle /* unused */,
                                                const AgentAddress & /* unused */,
                                                WorkloadNotificationMessage &&msg) {
            received_msg_size1 = msg.message().size();
            received1.Notify();
        };
        channel1_ = test::MakeControlChannel([](...) {}, [](...) {}, std::move(cb1));

        absl::Notification received2;
        size_t received_msg_size2 = 0;
        NotificationCallback cb2 = [&received_msg_size2,
                                    &received2](PeerHandle /* unused */,
                                                const AgentAddress & /* unused */,
                                                WorkloadNotificationMessage &&msg) {
            received_msg_size2 = msg.message().size();
            received2.Notify();
        };
        auto conn_cb_tracker = MakeConnectionCallbackTracker();
        channel2_ =
            test::MakeControlChannel(std::move(conn_cb_tracker->cb), [](...) {}, std::move(cb2));

        auto channel1_listen_addr = Listen(*channel1_);
        auto channel2_listen_addr = Listen(*channel2_);

        auto handle1 = Connect(*channel1_, channel2_listen_addr);
        ASSERT_TRUE(
            conn_cb_tracker->connected.WaitForNotificationWithTimeout(kConnectionReadyTimeout));
        WaitForConnectionReady(*channel1_, handle1);

        Message msg1;
        msg1.mutable_notif()->set_message(std::string(kMaxNotificiationSize, 'X'));
        Message msg2;
        msg2.mutable_notif()->set_message(std::string(kMaxNotificiationSize, 'Y'));

        ASSERT_TRUE(channel1_->ScheduleSend(handle1, std::move(msg1)).ok());
        ASSERT_TRUE(channel2_->ScheduleSend(conn_cb_tracker->peer_handle, std::move(msg2)).ok());

        ASSERT_TRUE(received1.WaitForNotificationWithTimeout(kMaxMessageTransmissionTimeout));
        ASSERT_TRUE(received2.WaitForNotificationWithTimeout(kMaxMessageTransmissionTimeout));

        EXPECT_EQ(received_msg_size1, kMaxNotificiationSize);
        EXPECT_EQ(received_msg_size2, kMaxNotificiationSize);
    }

    TEST_F(ControlChannelTest, DisconnectWhileSending) {
        absl::Notification received;
        NotificationCallback cb = [&received](PeerHandle /* unused */,
                                              const AgentAddress & /* unused */,
                                              WorkloadNotificationMessage && /* unused */) {
            received.Notify();
        };
        auto conn_cb_tracker = MakeConnectionCallbackTracker();

        channel2_ =
            test::MakeControlChannel(std::move(conn_cb_tracker->cb), [](...) {}, std::move(cb));
        Listen(*channel1_);
        auto channel2_listen_addr = Listen(*channel2_);
        auto handle = Connect(*channel1_, channel2_listen_addr);
        ASSERT_TRUE(
            conn_cb_tracker->connected.WaitForNotificationWithTimeout(kConnectionReadyTimeout));
        WaitForConnectionReady(*channel1_, handle);

        Message msg;
        msg.mutable_notif()->set_message(std::string(kMaxNotificiationSize, 'X'));

        ASSERT_TRUE(channel1_->ScheduleSend(handle, std::move(msg)).ok());
        EXPECT_TRUE(channel1_->Disconnect(handle).ok());

        // Wait for channel two to detect the drop
        auto start_drop_wait = absl::Now();
        bool dropped = false;
        while (absl::Now() - start_drop_wait < kDisconnectDetectionTimeout) {
            auto ready = channel2_->IsConnectionReady(conn_cb_tracker->peer_handle);
            // It might return an error status (not found) or false if socket disconnected but
            // still tracked
            if (!ready.ok() || !(*ready)) {
                dropped = true;
                break;
            }
            absl::SleepFor(kPollInterval);
        }
        EXPECT_TRUE(dropped);
    }

    TEST_F(ControlChannelTest, MultiThreadedSend) {
        static constexpr int kNumThreads = 10;
        static constexpr int kNumMessagesPerThread = 50;
        static constexpr int kTotalMessages = kNumThreads * kNumMessagesPerThread;

        std::atomic<int> received_count{0};
        absl::Notification all_received;
        NotificationCallback cb = [&received_count,
                                   &all_received](PeerHandle /* unused */,
                                                  const AgentAddress & /* unused */,
                                                  WorkloadNotificationMessage && /* unused */) {
            if (++received_count == kTotalMessages) {
                all_received.Notify();
            }
        };

        channel2_ = test::MakeControlChannel([](...) {}, [](...) {}, std::move(cb));
        Listen(*channel1_);
        auto addr2 = Listen(*channel2_);
        auto handle = ConnectAndWait(*channel1_, addr2);

        auto send_func = [this, handle = handle]() {
            for (int i = 0; i < kNumMessagesPerThread; ++i) {
                Message msg;
                msg.mutable_notif()->set_message("Test message from thread");
                EXPECT_TRUE(channel1_->ScheduleSend(handle, std::move(msg)).ok());
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < kNumThreads; ++i) {
            threads.emplace_back(send_func);
        }
        for (auto &t : threads) {
            t.join();
        }

        ASSERT_TRUE(all_received.WaitForNotificationWithTimeout(kMaxMessageTransmissionTimeout));
        EXPECT_EQ(received_count.load(), kTotalMessages);
    }

    TEST_F(ControlChannelTest, TcpKeepaliveConfiguration) {
        Listen(*channel1_);
        auto channel2_listen_addr = Listen(*channel2_);
        auto handle = ConnectAndWait(*channel1_, channel2_listen_addr);

        // Get the underlying socket for channel1's connection to channel2
        int fd = static_cast<int>(handle);

        int optval = 0;
        socklen_t optlen = sizeof(optval);

        // Verify SO_KEEPALIVE is enabled
        EXPECT_EQ(::getsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &optval, &optlen), 0);
        EXPECT_EQ(optval, 1);

        EXPECT_EQ(::getsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &optval, &optlen), 0);
        int expected_idle = absl::ToInt64Seconds(test::kHeartbeatSendPeriod);
        EXPECT_EQ(optval, expected_idle);

        EXPECT_EQ(::getsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &optval, &optlen), 0);
        int expected_intvl = absl::ToInt64Seconds(test::kHeartbeatSendPeriod);
        EXPECT_EQ(optval, expected_intvl);

        // Verify TCP_KEEPCNT matches the period -> count math
        EXPECT_EQ(::getsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &optval, &optlen), 0);
        int expected_probes =
            std::ceil(static_cast<double>(absl::ToInt64Seconds(test::kHeartbeatTimeout)) /
                      static_cast<double>(expected_intvl));
        EXPECT_EQ(optval, expected_probes);
    }

} // namespace
} // namespace tcpxo

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
