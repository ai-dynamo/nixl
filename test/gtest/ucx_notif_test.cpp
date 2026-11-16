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
#include "common/hw_info.h"
#include "ucx_backend_req.h"
#include "ucx_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t large_payload_size = 1024 * 1024;

// A sender and a receiver worker of one UCX context. A large notification
// stays pending while the receiver does not progress.
class ucxNotification : public ::testing::Test {
protected:
    void
    SetUp() override {
        ASSERT_NO_FATAL_FAILURE(
            createWorkers("tcp", false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE));
    }

    void
    createWorkers(const std::string &tls, bool prog_thread, nixl_thread_sync_t sync_mode) {
        env_.addVar("UCX_TLS", tls);
        env_.addVar("UCX_ZCOPY_THRESH", "0");
        // Small socket buffers keep a large send pending on an established
        // TCP connection while the receiver does not progress
        env_.addVar("UCX_TCP_SNDBUF", "65536");
        env_.addVar("UCX_TCP_RCVBUF", "65536");
        // Two workers, so that tests can progress them from different threads
        context_ = std::make_unique<nixlUcxContext>(std::vector<std::string>{},
                                                    prog_thread,
                                                    2,
                                                    sync_mode,
                                                    0,
                                                    "TLS=" + tls + ",ZCOPY_THRESH=0");
        receiver_ = std::make_unique<nixlUcxWorker>(*context_, UCP_ERR_HANDLING_MODE_PEER);
        ASSERT_EQ(UCS_OK,
                  receiver_->regAmCallback(nixl::ucx::am_cb_op_t::NOTIF_STR, receive, &received_));
        ASSERT_NO_FATAL_FAILURE(reconnect(UCP_ERR_HANDLING_MODE_PEER));
    }

    static ucs_status_t
    receive(void *arg,
            const void *,
            size_t,
            void *data,
            size_t length,
            const ucp_am_recv_param_t *) {
        static_cast<std::string *>(arg)->append(static_cast<const char *>(data), length);
        return UCS_OK;
    }

    template<typename Predicate>
    bool
    waitUntil(Predicate predicate) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        do {
            if (predicate()) {
                return true;
            }
            std::this_thread::yield();
        } while (std::chrono::steady_clock::now() < deadline);
        return false;
    }

    // Replace the sender worker and its endpoint
    void
    reconnect(ucp_err_handling_mode_t mode,
              std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        ep_.reset();
        sender_ = std::make_unique<nixlUcxWorker>(*context_, mode);
        sender_->setAmDrainTimeout(timeout);
        auto address = receiver_->epAddr();
        ep_ = sender_->connect(address.data());
        ASSERT_NE(nullptr, ep_);
    }

    [[nodiscard]] nixl_status_t
    send(std::string payload) {
        return ep_->sendAm(
            nixl::ucx::am_cb_op_t::NOTIF_STR, std::move(payload), UCP_AM_SEND_FLAG_EAGER);
    }

    // Reconnect and exchange a message, then leave the receiver idle
    void
    establish(ucp_err_handling_mode_t mode,
              std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) {
        ASSERT_NO_FATAL_FAILURE(reconnect(mode, timeout));
        const nixl_status_t status = send("ready");
        ASSERT_TRUE(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
        ASSERT_TRUE(waitUntil([&] {
            sender_->progress();
            receiver_->progress();
            return received_ == "ready" && sender_->getPendingAmSends() == 0;
        }));
        // A TCP flush after a send waits for the peer. Flush now, so that a
        // flush with no sends after this one completes without the receiver.
        nixlUcxReq req = nullptr;
        const nixl_status_t flushed = ep_->flushEp(req);
        ASSERT_TRUE(flushed == NIXL_SUCCESS || flushed == NIXL_IN_PROG);
        if (flushed == NIXL_IN_PROG) {
            ASSERT_TRUE(waitUntil([&] {
                sender_->progress();
                receiver_->progress();
                return ucp_request_check_status(req) != UCS_INPROGRESS;
            }));
            EXPECT_EQ(UCS_OK, ucp_request_check_status(req));
            sender_->reqRelease(req);
        }
        received_.clear();
    }

    // Progress both workers until the receiver has the payload
    void
    progressUntilReceived(const std::string &payload) {
        ASSERT_TRUE(waitUntil([&] {
            sender_->progress();
            receiver_->progress();
            return received_.size() == payload.size();
        }));
        EXPECT_TRUE(payload == received_)
            << "expected bytes=" << payload.size() << ", received bytes=" << received_.size();
    }

    // Progress the sender until its send callbacks have run
    void
    progressUntilNoPendingSends() {
        ASSERT_TRUE(waitUntil([&] {
            sender_->progress();
            return sender_->getPendingAmSends() == 0;
        }));
        EXPECT_EQ(0, ep_->getPendingAmSends());
    }

    // Destroy the endpoint and the worker while a send is pending. The pending
    // counters must drop to zero, as the payloads must be released (LeakSanitizer).
    void
    destroyPending(ucp_err_handling_mode_t mode) {
        const gtest::LogIgnoreGuard timeout("UCX AM drain timed out with.*pending sends");
        ASSERT_NO_FATAL_FAILURE(reconnect(mode, std::chrono::milliseconds(0)));
        const auto worker_stats = sender_->getAmSendStats();
        const auto ep_stats = ep_->getAmSendStats();
        ASSERT_EQ(NIXL_IN_PROG, send(std::string(large_payload_size, 'n')));
        ASSERT_EQ(1, worker_stats->pending());
        ASSERT_EQ(1, ep_stats->pending());
        ep_.reset();
        sender_.reset();
        EXPECT_EQ(0, worker_stats->pending());
        EXPECT_EQ(0, ep_stats->pending());
    }

    // Destroy the endpoint while another thread progresses the receiver:
    // the endpoint destructor must drain the send, so that it is delivered.
    void
    drainWithCooperatingReceiver() {
        const std::string payload(large_payload_size, 'n');
        const auto ep_stats = ep_->getAmSendStats();
        ASSERT_EQ(NIXL_IN_PROG, send(payload));
        ASSERT_EQ(1, ep_stats->pending());
        std::atomic<bool> delivered{false};
        std::jthread receiver_progress([&](std::stop_token stop) {
            while (!stop.stop_requested()) {
                receiver_->progress();
                if (received_.size() == payload.size()) {
                    delivered = true;
                    return;
                }
            }
        });
        ep_.reset();
        EXPECT_EQ(0, ep_stats->pending());
        EXPECT_TRUE(waitUntil([&] { return delivered.load(); }));
        receiver_progress.request_stop();
        receiver_progress.join();
        EXPECT_TRUE(payload == received_)
            << "expected bytes=" << payload.size() << ", received bytes=" << received_.size();
    }

    gtest::ScopedEnv env_;
    std::string received_;
    std::unique_ptr<nixlUcxContext> context_;
    std::unique_ptr<nixlUcxWorker> receiver_;
    std::unique_ptr<nixlUcxWorker> sender_;
    std::unique_ptr<nixlUcxEp> ep_;
};

TEST_F(ucxNotification, TransferDoneWhileNotificationSendIsPending) {
    ASSERT_NO_FATAL_FAILURE(establish(UCP_ERR_HANDLING_MODE_PEER, std::chrono::milliseconds(1000)));
    const std::string payload(large_payload_size, 'n');
    nixlUcxBackendReqH transfer(sender_.get());
    transfer.setNotif(std::string(payload));
    transfer.init({}, *ep_);
    ASSERT_EQ(NIXL_SUCCESS, transfer.flush());

    // There are no data requests, and the flush sends the notification. The
    // receiver does not progress, so the notification send is still pending
    // when the transfer completes.
    nixl_status_t status = NIXL_IN_PROG;
    EXPECT_TRUE(waitUntil([&] { return (status = transfer.status()) != NIXL_IN_PROG; }));
    ASSERT_EQ(NIXL_SUCCESS, status);
    EXPECT_EQ(1, ep_->getPendingAmSends());
    transfer.release();

    ASSERT_NO_FATAL_FAILURE(progressUntilReceived(payload));
    ASSERT_NO_FATAL_FAILURE(progressUntilNoPendingSends());
}

TEST_F(ucxNotification, CompletionClearsPendingCounters) {
    const std::string payload(large_payload_size, 'n');
    const auto worker_stats = sender_->getAmSendStats();
    const auto ep_stats = ep_->getAmSendStats();
    ASSERT_EQ(NIXL_IN_PROG, send(payload));
    EXPECT_EQ(1, worker_stats->pending());
    EXPECT_EQ(1, ep_stats->pending());
    ASSERT_NO_FATAL_FAILURE(progressUntilReceived(payload));
    ASSERT_NO_FATAL_FAILURE(progressUntilNoPendingSends());
    EXPECT_EQ(0, worker_stats->pending());
    EXPECT_EQ(0, ep_stats->pending());
}

TEST_F(ucxNotification, PendingWarningDoesNotRejectNotifications) {
    const gtest::LogIgnoreGuard backlog("UCX AM pending sends reached warning threshold 1.*");
    sender_->setAmWarnThreshold(1);
    for (size_t i = 0; i < 2; ++i) {
        ASSERT_EQ(NIXL_IN_PROG, send(std::string(large_payload_size, 'n')));
    }
    EXPECT_EQ(2, sender_->getPendingAmSends());
    EXPECT_EQ(1, backlog.getIgnoredCount());
    ASSERT_NO_FATAL_FAILURE(progressUntilReceived(std::string(2 * large_payload_size, 'n')));
    ASSERT_NO_FATAL_FAILURE(progressUntilNoPendingSends());
}

TEST_F(ucxNotification, FailedEndpointRejectsNotification) {
    ep_->err_cb(ep_->getEp(), UCS_ERR_CONNECTION_RESET);
    EXPECT_EQ(NIXL_ERR_REMOTE_DISCONNECT, send(std::string(large_payload_size, 'n')));
    EXPECT_EQ(0, sender_->getPendingAmSends());
    EXPECT_EQ(0, ep_->getPendingAmSends());
}

TEST_F(ucxNotification, EndpointDestructionDrainsNotification) {
    drainWithCooperatingReceiver();
}

TEST_F(ucxNotification, DrainTimeoutIsBounded) {
    const gtest::LogIgnoreGuard timeout("UCX AM drain timed out with.*pending sends");
    ASSERT_NO_FATAL_FAILURE(reconnect(UCP_ERR_HANDLING_MODE_PEER, std::chrono::milliseconds(5)));
    ASSERT_EQ(NIXL_IN_PROG, send(std::string(large_payload_size, 'n')));
    EXPECT_FALSE(ep_->drainAmSends(std::chrono::milliseconds(0)));
    const auto worker_stats = sender_->getAmSendStats();
    const auto ep_stats = ep_->getAmSendStats();
    const auto start = std::chrono::steady_clock::now();
    ep_.reset();
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(1));
    sender_.reset();
    EXPECT_EQ(0, worker_stats->pending());
    EXPECT_EQ(0, ep_stats->pending());
}

TEST_F(ucxNotification, ProgressNotificationAfterEndpointDestroyed) {
    // No drain budget: a graceful close still delivers the pending send while
    // its worker keeps progressing.
    const gtest::LogIgnoreGuard timeout("UCX AM drain timed out with.*pending sends");
    ASSERT_NO_FATAL_FAILURE(reconnect(UCP_ERR_HANDLING_MODE_NONE, std::chrono::milliseconds(0)));
    const std::string payload(large_payload_size, 'n');
    ASSERT_EQ(NIXL_IN_PROG, send(payload));
    ep_.reset();
    ASSERT_NO_FATAL_FAILURE(progressUntilReceived(payload));
    EXPECT_TRUE(sender_->drainAmSends(std::chrono::milliseconds(1000)));
    EXPECT_EQ(0, sender_->getPendingAmSends());
    EXPECT_EQ(1, timeout.getIgnoredCount());
}

TEST_F(ucxNotification, DestroyWorkerWithPendingNotificationPeer) {
    destroyPending(UCP_ERR_HANDLING_MODE_PEER);
}

TEST_F(ucxNotification, DestroyWorkerWithPendingNotificationNone) {
    destroyPending(UCP_ERR_HANDLING_MODE_NONE);
}

TEST_F(ucxNotification, MaximumTimeoutOnEmptyWorkerDoesNotOverflow) {
    sender_->setAmDrainTimeout(std::chrono::milliseconds::max());
    EXPECT_EQ(0, sender_->getPendingAmSends());
    EXPECT_TRUE(sender_->drainAmSends(std::chrono::milliseconds::max()));
    EXPECT_TRUE(ep_->drainAmSends(std::chrono::milliseconds::max()));
}

TEST_F(ucxNotification, OneShutdownDeadlineCapsMultipleWorkers) {
    const gtest::LogIgnoreGuard timeout("UCX AM drain timed out with.*pending sends");
    std::array<std::unique_ptr<nixlUcxWorker>, 3> workers;
    std::array<std::unique_ptr<nixlUcxEp>, 3> endpoints;
    std::array<std::shared_ptr<const nixlUcxAmSendStats>, 3> stats;
    auto address = receiver_->epAddr();
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::make_unique<nixlUcxWorker>(*context_, UCP_ERR_HANDLING_MODE_PEER);
        workers[i]->setAmDrainTimeout(std::chrono::milliseconds(1000));
        endpoints[i] = workers[i]->connect(address.data());
        ASSERT_NE(nullptr, endpoints[i]);
        stats[i] = workers[i]->getAmSendStats();
        ASSERT_EQ(NIXL_IN_PROG,
                  endpoints[i]->sendAm(nixl::ucx::am_cb_op_t::NOTIF_STR,
                                       std::string(large_payload_size, 'n'),
                                       UCP_AM_SEND_FLAG_EAGER));
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
    for (const auto &worker : workers) {
        worker->beginAmShutdown(deadline);
    }
    EXPECT_FALSE(workers[0]->drainAmSends(std::chrono::milliseconds(1000)));
    // The first worker used up the shared budget. No later worker or endpoint
    // may start another full timeout, even when given a later deadline.
    const auto tail_start = std::chrono::steady_clock::now();
    for (size_t i = 1; i < workers.size(); ++i) {
        workers[i]->beginAmShutdown(tail_start + std::chrono::seconds(5));
        EXPECT_FALSE(workers[i]->drainAmSends(std::chrono::milliseconds(1000)));
    }
    endpoints = {};
    workers = {};
    EXPECT_LT(std::chrono::steady_clock::now() - tail_start, std::chrono::milliseconds(500));
    for (const auto &observer : stats) {
        EXPECT_EQ(0, observer->pending());
    }
}

// Established RC connections: the send is posted to the NIC, not only queued
class ucxNotificationRc : public ucxNotification {
protected:
    // A small receive queue keeps a large AM from completing while the
    // receiver does not progress, even on an established connection.
    virtual bool
    limitReceiveQueue() const {
        return true;
    }

    void
    SetUp() override {
        if (nixl::hwInfo::instance().numIbDevices == 0) {
            GTEST_SKIP() << "Established RC notification tests require an RDMA device";
        }
        if (limitReceiveQueue()) {
            env_.addVar("UCX_RC_RX_QUEUE_LEN", "32");
            env_.addVar("UCX_RC_TX_CQ_LEN", "16");
        } else {
            env_.unsetVar("UCX_RC_RX_QUEUE_LEN");
            env_.unsetVar("UCX_RC_TX_CQ_LEN");
        }
        ASSERT_NO_FATAL_FAILURE(
            createWorkers("rc", false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE));
    }

    void
    destroyEstablished(ucp_err_handling_mode_t mode, bool progress_send) {
        ASSERT_NO_FATAL_FAILURE(establish(mode));
        const gtest::LogIgnoreGuard timeout("UCX AM drain timed out with.*pending sends");
        const auto worker_stats = sender_->getAmSendStats();
        const auto ep_stats = ep_->getAmSendStats();
        ASSERT_EQ(NIXL_IN_PROG, send(std::string(large_payload_size, 'n')));
        if (progress_send) {
            ASSERT_TRUE(waitUntil([&] { return sender_->progress() != 0; }));
        }
        ASSERT_GT(worker_stats->pending(), 0);
        ASSERT_GT(ep_stats->pending(), 0);
        const auto start = std::chrono::steady_clock::now();
        ep_.reset();
        sender_.reset();
        EXPECT_EQ(0, worker_stats->pending());
        EXPECT_EQ(0, ep_stats->pending());
        EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
    }
};

class ucxNotificationRcDelivery : public ucxNotificationRc {
protected:
    bool
    limitReceiveQueue() const override {
        return false;
    }
};

TEST_F(ucxNotificationRc, EstablishedPendingPeerTeardown) {
    destroyEstablished(UCP_ERR_HANDLING_MODE_PEER, false);
}

TEST_F(ucxNotificationRc, EstablishedInflightPeerTeardown) {
    destroyEstablished(UCP_ERR_HANDLING_MODE_PEER, true);
}

TEST_F(ucxNotificationRcDelivery, EstablishedEndpointDrainsNotification) {
    ASSERT_NO_FATAL_FAILURE(establish(UCP_ERR_HANDLING_MODE_PEER, std::chrono::milliseconds(5000)));
    drainWithCooperatingReceiver();
}

// MULTI workers: sends, progress and endpoint teardown race
class ucxNotificationMt : public ucxNotification {
protected:
    void
    SetUp() override {
        if (!nixlUcxMtLevelIsSupported(nixl::ucx::mt_mode_t::WORKER)) {
            GTEST_SKIP() << "Concurrent notification stress requires a MULTI UCX worker";
        }
        ASSERT_NO_FATAL_FAILURE(
            createWorkers("tcp", true, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW));
        // A lifetime stress, not a timeout test: let the whole burst drain,
        // also under sanitizers
        sender_->setAmDrainTimeout(std::chrono::milliseconds(5000));
        ucp_worker_attr_t attributes{};
        attributes.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
        ASSERT_EQ(UCS_OK, ucp_worker_query(sender_->get(), &attributes));
        ASSERT_EQ(UCS_THREAD_MODE_MULTI, attributes.thread_mode);
    }
};

TEST_F(ucxNotificationMt, ConcurrentSendProgressAndTeardown) {
    constexpr size_t producers = 4;
    constexpr size_t per_producer = 128;
    constexpr size_t payload_size = 8192;
    constexpr size_t messages = producers * per_producer;
    constexpr uint32_t seed = 7;
    const auto worker_stats = sender_->getAmSendStats();
    const auto ep_stats = ep_->getAmSendStats();
    std::atomic<size_t> errors{0};
    std::atomic<size_t> received_bytes{0};
    std::atomic<bool> receive_enabled{false};
    std::jthread sender_progress([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            sender_->progress();
            std::this_thread::yield();
        }
    });
    std::jthread receiver_progress([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            if (receive_enabled.load()) {
                receiver_->progress();
                received_bytes = received_.size();
            }
            std::this_thread::yield();
        }
    });
    std::vector<std::thread> sending;
    for (size_t producer = 0; producer < producers; ++producer) {
        sending.emplace_back([&, producer] {
            for (size_t i = 0; i < per_producer; ++i) {
                std::string payload(payload_size, 'n');
                gtest::fillPayload(
                    payload.data(), payload.size(), producer * per_producer + i, seed);
                const nixl_status_t status = send(std::move(payload));
                if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
                    ++errors;
                }
            }
        });
    }
    for (auto &thread : sending) {
        thread.join();
    }
    EXPECT_GT(worker_stats->pending(), 0);
    receive_enabled = true;
    // The senders are joined, but the progress thread still races the send
    // callbacks against the endpoint drain and close.
    ep_.reset();
    const bool delivered =
        waitUntil([&] { return received_bytes.load() == messages * payload_size; });
    sender_progress.request_stop();
    sender_progress.join();
    // Stop progressing the worker before destroying it
    sender_.reset();
    receiver_progress.request_stop();
    receiver_progress.join();
    EXPECT_EQ(0, errors.load());
    EXPECT_TRUE(delivered);
    EXPECT_EQ(0, worker_stats->pending());
    EXPECT_EQ(0, ep_stats->pending());
    ASSERT_EQ(messages * payload_size, received_.size());
    std::array<bool, messages> seen{};
    for (size_t offset = 0; offset < received_.size(); offset += payload_size) {
        uint32_t id;
        std::memcpy(&id, received_.data() + offset, sizeof(id));
        ASSERT_LT(id, messages);
        EXPECT_FALSE(seen[id]);
        seen[id] = true;
        EXPECT_TRUE(gtest::checkPayload(received_.data() + offset, payload_size, id, seed));
    }
}

TEST(ucxNotificationConfig, InvalidDrainAndWarningParametersRejected) {
    nixlBackendInitParams init;
    init.localAgent = "notif-config";
    init.type = "UCX";
    init.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE;
    nixl_b_params_t params;
    init.customParams = &params;
    params["ucx_notif_drain_timeout_ms"] = "-1";
    EXPECT_THROW((void)nixlUcxEngine::create(init), std::invalid_argument);
    for (const auto *name : {"ucx_notif_drain_timeout_ms", "ucx_notif_warn_threshold"}) {
        for (const auto *value : {"invalid", "1.5", "999999999999999999999999999999"}) {
            params = {{name, value}};
            EXPECT_THROW((void)nixlUcxEngine::create(init), std::runtime_error);
        }
    }
    params = {{"ucx_notif_warn_threshold", "-1"}};
    EXPECT_THROW((void)nixlUcxEngine::create(init), std::runtime_error);
}

TEST(ucxNotificationConfig, ZeroAndMaximumDrainTimeoutAreAccepted) {
    gtest::ScopedEnv env;
    env.addVar("UCX_TLS", "tcp");
    const gtest::LogIgnoreGuard gpu_warning(
        "GPU\\(s\\) were detected, but UCX (CUDA|ROCm) support was not found");
    nixl_b_params_t params{{"ucx_notif_drain_timeout_ms", "0"},
                           {"ucx_notif_warn_threshold", "0"},
                           {"engine_config", "TLS=tcp"}};
    nixlBackendInitParams init;
    init.localAgent = "notif-zero-config";
    init.type = "UCX";
    init.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE;
    init.customParams = &params;
    for (const auto timeout : {std::chrono::milliseconds(0), std::chrono::milliseconds::max()}) {
        params["ucx_notif_drain_timeout_ms"] = std::to_string(timeout.count());
        EXPECT_NO_THROW((void)nixlUcxEngine::create(init));
    }
}

TEST(ucxNotificationConfig, DrainDeadlineSaturatesWithoutOverflow) {
    using clock = std::chrono::steady_clock;
    using milliseconds = std::chrono::milliseconds;
    EXPECT_EQ(clock::time_point::max(), nixlUcxAmDrainDeadline(milliseconds::max()));
    for (const auto timeout : {milliseconds::min(), milliseconds(0), milliseconds(20)}) {
        const auto before = clock::now();
        const auto deadline = nixlUcxAmDrainDeadline(timeout);
        const auto after = clock::now();
        const auto increment = std::max(timeout, milliseconds(0));
        EXPECT_LE(before + increment, deadline);
        EXPECT_LE(deadline, after + increment);
    }
}

} // namespace
