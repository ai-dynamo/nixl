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

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct agentNotifParam {
    bool composite;
    bool progressThread;
};

// Transfer notifications through the public agent API, on the default and the
// thread-pool (composite request) engine, with and without a progress thread
// on the initiator.
class ucxNotificationAgent : public ::testing::TestWithParam<agentNotifParam> {
protected:
    using clock = std::chrono::steady_clock;
    static constexpr size_t descriptorCount = 8;
    static constexpr size_t descriptorBytes = 64;
    using buffer = std::array<unsigned char, descriptorBytes>;

    void
    SetUp() override {
        env_.addVar("NIXL_TELEMETRY_ENABLE", "n");
        env_.addVar("NIXL_UCX_SGL_ENABLE", "n");
        env_.addVar("UCX_TLS", "tcp");
        env_.addVar("UCX_ZCOPY_THRESH", "0");
        // Small socket buffers: a large notification needs many progress
        // calls on the initiator to complete.
        env_.addVar("UCX_TCP_SNDBUF", "65536");
        env_.addVar("UCX_TCP_RCVBUF", "65536");

        // Only the initiator's progress thread varies. A thread that polls the
        // target's notifications progresses the target in all cases.
        nixlAgentConfig config;
        config.useProgThread = GetParam().progressThread;
        config.useListenThread = false;
        config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        sender_ = std::make_unique<nixlAgent>(senderName, config);
        config.useProgThread = false;
        receiver_ = std::make_unique<nixlAgent>(receiverName, config);
        // split_batch_size=2 splits every transfer of the composite case into chunks
        nixl_b_params_t params{{"num_workers", GetParam().composite ? "3" : "1"},
                               {"num_threads", GetParam().composite ? "2" : "0"},
                               {"split_batch_size", "2"},
                               {"engine_config", "TLS=tcp,ZCOPY_THRESH=0"}};
        const gtest::LogIgnoreGuard gpu_warning(
            "GPU\\(s\\) were detected, but UCX (CUDA|ROCm) support was not found");
        nixlBackendH *backend = nullptr;
        ASSERT_EQ(NIXL_SUCCESS, sender_->createBackend("UCX", params, backend));
        ASSERT_NE(nullptr, backend);
        ASSERT_EQ(NIXL_SUCCESS, receiver_->createBackend("UCX", params, backend));
        ASSERT_NE(nullptr, backend);

        for (size_t i = 0; i < descriptorCount; ++i) {
            local_.push_back(std::make_unique<buffer>());
            remote_.push_back(std::make_unique<buffer>());
            const auto laddr = reinterpret_cast<uintptr_t>(local_.back()->data());
            const auto raddr = reinterpret_cast<uintptr_t>(remote_.back()->data());
            localReg_.addDesc(nixlBlobDesc(laddr, descriptorBytes, 0));
            remoteReg_.addDesc(nixlBlobDesc(raddr, descriptorBytes, 0));
            localDescs_.addDesc(nixlBasicDesc(laddr, descriptorBytes, 0));
            remoteDescs_.addDesc(nixlBasicDesc(raddr, descriptorBytes, 0));
        }
        ASSERT_EQ(NIXL_SUCCESS, sender_->registerMem(localReg_));
        ASSERT_EQ(NIXL_SUCCESS, receiver_->registerMem(remoteReg_));
        nixl_blob_t metadata;
        std::string name;
        ASSERT_EQ(NIXL_SUCCESS, receiver_->getLocalMD(metadata));
        ASSERT_EQ(NIXL_SUCCESS, sender_->loadRemoteMD(metadata, name));
        ASSERT_EQ(receiverName, name);
        ASSERT_EQ(NIXL_SUCCESS, sender_->getLocalMD(metadata));
        ASSERT_EQ(NIXL_SUCCESS, receiver_->loadRemoteMD(metadata, name));
        ASSERT_EQ(senderName, name);

        receiverProgress_ = std::jthread([this](std::stop_token stop) {
            while (!stop.stop_requested()) {
                nixl_notifs_t notifications;
                const nixl_status_t status = receiver_->getNotifs(notifications);
                if (status != NIXL_SUCCESS) {
                    receiverStatus_ = status;
                    return;
                }
                for (const auto &payload : notifications[senderName]) {
                    const std::lock_guard lock(receiptMutex_);
                    receipts_.push_back(payload);
                }
                std::this_thread::yield();
            }
        });
    }

    void
    TearDown() override {
        // Keep progressing the target while the initiator's teardown drains
        sender_.reset();
        receiverProgress_.request_stop();
        if (receiverProgress_.joinable()) {
            receiverProgress_.join();
        }
        receiver_.reset();
    }

    size_t
    receiptCount() {
        const std::lock_guard lock(receiptMutex_);
        return receipts_.size();
    }

    std::string
    getReceipt(size_t index) {
        const std::lock_guard lock(receiptMutex_);
        return receipts_.at(index);
    }

    template<typename Predicate>
    bool
    waitUntil(Predicate predicate, std::chrono::seconds timeout = std::chrono::seconds(10)) {
        const auto deadline = clock::now() + timeout;
        do {
            if (predicate()) {
                return true;
            }
            std::this_thread::yield();
        } while (clock::now() < deadline);
        return false;
    }

    void
    preparePayload(nixl_xfer_op_t operation, uint32_t seed) {
        auto &source = operation == NIXL_WRITE ? local_ : remote_;
        auto &target = operation == NIXL_WRITE ? remote_ : local_;
        for (size_t i = 0; i < descriptorCount; ++i) {
            gtest::fillPayload(source[i]->data(), descriptorBytes, i, seed);
            gtest::fillPayload(target[i]->data(), descriptorBytes, i, 0);
        }
    }

    bool
    payloadMatches(nixl_xfer_op_t operation, uint32_t seed) {
        const auto &target = operation == NIXL_WRITE ? remote_ : local_;
        for (size_t i = 0; i < descriptorCount; ++i) {
            if (!gtest::checkPayload(target[i]->data(), descriptorBytes, i, seed)) {
                return false;
            }
        }
        return true;
    }

    // Post a transfer and poll it until it is no longer in progress
    nixl_status_t
    transfer(nixl_xfer_op_t operation, const std::string &notification, nixlXferReqH *&request) {
        nixl_opt_args_t options;
        options.notif = notification;
        const auto status = sender_->createXferReq(
            operation, localDescs_, remoteDescs_, receiverName, request, &options);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        nixl_status_t result = sender_->postXferReq(request);
        const auto deadline = clock::now() + std::chrono::seconds(10);
        while (result == NIXL_IN_PROG && clock::now() < deadline) {
            result = sender_->getXferStatus(request);
        }
        return result;
    }

    // Complete a transfer with a large notification, which is still pending
    // at completion, then disconnect or destroy the initiator without any
    // further progress through its API
    void
    doneThenCleanup(bool destroy, bool release_before_disconnect = true) {
        const std::string notification(1024 * 1024, 'n');
        preparePayload(NIXL_WRITE, 17);
        nixlXferReqH *request = nullptr;
        ASSERT_EQ(NIXL_SUCCESS, transfer(NIXL_WRITE, notification, request));
        ASSERT_TRUE(payloadMatches(NIXL_WRITE, 17));
        if (destroy || release_before_disconnect) {
            ASSERT_EQ(NIXL_SUCCESS, sender_->releaseXferReq(request));
            request = nullptr;
        }
        if (destroy) {
            sender_.reset();
        } else {
            EXPECT_EQ(NIXL_SUCCESS, sender_->invalidateRemoteMD(receiverName));
        }
        EXPECT_TRUE(waitUntil([&] { return receiptCount() == 1; }));
        EXPECT_EQ(NIXL_SUCCESS, receiverStatus_.load());
        if (receiptCount() == 1) {
            EXPECT_TRUE(notification == getReceipt(0));
        }
        if (request != nullptr) {
            ASSERT_EQ(NIXL_SUCCESS, sender_->releaseXferReq(request));
        }
    }

    // Buffers outlive the agents and their registered memory
    std::vector<std::unique_ptr<buffer>> local_, remote_;
    nixl_reg_dlist_t localReg_{DRAM_SEG}, remoteReg_{DRAM_SEG};
    nixl_xfer_dlist_t localDescs_{DRAM_SEG}, remoteDescs_{DRAM_SEG};
    gtest::ScopedEnv env_;
    std::unique_ptr<nixlAgent> sender_, receiver_;
    std::jthread receiverProgress_;
    std::mutex receiptMutex_;
    std::vector<std::string> receipts_;
    std::atomic<nixl_status_t> receiverStatus_{NIXL_SUCCESS};
    static constexpr const char *senderName = "notif-agent-source";
    static constexpr const char *receiverName = "notif-agent-target";
};

TEST_P(ucxNotificationAgent, DoneThenDisconnectDeliversNotification) {
    doneThenCleanup(false);
}

TEST_P(ucxNotificationAgent, DoneThenAgentDestructionDeliversNotification) {
    doneThenCleanup(true);
}

TEST_P(ucxNotificationAgent, DoneThenDisconnectWithLiveHandleDeliversNotification) {
    // A completed request keeps its backend connection. Disconnect must drain
    // the connection before it drops it from its map.
    doneThenCleanup(false, false);
}

TEST_P(ucxNotificationAgent, NotificationArrivesWithoutInitiatorCalls) {
    // After the transfer completes, the threads of the initiator have to
    // finish sending its notification: the progress thread or the thread that
    // sent the notification of a split transfer.
    if (!GetParam().composite && !GetParam().progressThread) {
        GTEST_SKIP() << "Without a progress thread, the application progresses the initiator";
    }
    const std::string notification(1024 * 1024, 'n');
    preparePayload(NIXL_WRITE, 23);
    nixlXferReqH *request = nullptr;
    ASSERT_EQ(NIXL_SUCCESS, transfer(NIXL_WRITE, notification, request));
    ASSERT_EQ(NIXL_SUCCESS, sender_->releaseXferReq(request));
    // Many times what the initiator's threads need when they progress it
    // without pause; far less than with a progress call per second.
    EXPECT_TRUE(waitUntil([&] { return receiptCount() == 1; }, std::chrono::seconds(3)));
    if (receiptCount() == 1) {
        EXPECT_TRUE(notification == getReceipt(0));
    }
    EXPECT_EQ(NIXL_SUCCESS, receiverStatus_.load());
}

TEST_P(ucxNotificationAgent, RepeatedReadWriteNotificationsAtDone) {
    const std::string notification(256, 'n');
    size_t expected = 0;
    for (const auto operation : {NIXL_WRITE, NIXL_READ}) {
        for (uint32_t seed = 1; seed <= 32; ++seed) {
            preparePayload(operation, seed);
            nixlXferReqH *request = nullptr;
            ASSERT_EQ(NIXL_SUCCESS, transfer(operation, notification, request));
            ASSERT_TRUE(payloadMatches(operation, seed));
            ASSERT_EQ(NIXL_SUCCESS, sender_->releaseXferReq(request));
            ++expected;
            ASSERT_TRUE(waitUntil([&] {
                nixl_notifs_t unused;
                EXPECT_EQ(NIXL_SUCCESS, sender_->getNotifs(unused));
                return receiptCount() == expected;
            }));
            EXPECT_EQ(notification, getReceipt(expected - 1));
        }
    }
    EXPECT_EQ(NIXL_SUCCESS, receiverStatus_.load());
}

TEST_P(ucxNotificationAgent, ReleaseWhileFlushCallbackMayBePending) {
    const gtest::LogIgnoreGuard canceled_requests(
        "request handle .* released with .* requests in progress");
    const gtest::LogIgnoreGuard release_retry(
        "could not release transfer request and returned error status");
    nixl_opt_args_t options;
    options.notif = "released-transfer";
    // Keep the registered buffers and their source bytes unchanged until all
    // aborted operations have drained; only the request ownership races
    // with progress.
    preparePayload(NIXL_WRITE, 31);
    for (size_t iteration = 0; iteration < 100; ++iteration) {
        nixlXferReqH *request = nullptr;
        ASSERT_EQ(NIXL_SUCCESS,
                  sender_->createXferReq(
                      NIXL_WRITE, localDescs_, remoteDescs_, receiverName, request, &options));
        const auto posted = sender_->postXferReq(request);
        ASSERT_TRUE(posted == NIXL_SUCCESS || posted == NIXL_IN_PROG);
        const auto deadline = clock::now() + std::chrono::seconds(10);
        do {
            const auto released = sender_->releaseXferReq(request);
            if (released == NIXL_SUCCESS) {
                request = nullptr;
                break;
            }
            ASSERT_EQ(NIXL_ERR_REPOST_ACTIVE, released);
            const auto status = sender_->getXferStatus(request);
            ASSERT_TRUE(status == NIXL_SUCCESS || status == NIXL_IN_PROG ||
                        status == NIXL_ERR_CANCELED);
            std::this_thread::yield();
        } while (clock::now() < deadline);
        ASSERT_EQ(nullptr, request) << "active transfer could not be released before the deadline";
    }
    // A completed transfer afterwards shows that the releases neither
    // corrupted the callback state nor broke the connection. Released
    // transfers may or may not have sent their notifications.
    const std::string barrier = "after-released-transfers";
    nixlXferReqH *request = nullptr;
    ASSERT_EQ(NIXL_SUCCESS, transfer(NIXL_WRITE, barrier, request));
    ASSERT_EQ(NIXL_SUCCESS, sender_->releaseXferReq(request));
    ASSERT_TRUE(waitUntil([&] {
        nixl_notifs_t unused;
        EXPECT_EQ(NIXL_SUCCESS, sender_->getNotifs(unused));
        const size_t count = receiptCount();
        for (size_t i = 0; i < count; ++i) {
            if (getReceipt(i) == barrier) {
                return true;
            }
        }
        return false;
    }));
    EXPECT_EQ(NIXL_SUCCESS, receiverStatus_.load());
}

INSTANTIATE_TEST_SUITE_P(ucx,
                         ucxNotificationAgent,
                         ::testing::Values(agentNotifParam{false, false},
                                           agentNotifParam{false, true},
                                           agentNotifParam{true, false},
                                           agentNotifParam{true, true}),
                         [](const ::testing::TestParamInfo<agentNotifParam> &info) {
                             return std::string(info.param.composite ? "Composite" : "Base") +
                                 (info.param.progressThread ? "ProgressThread" : "Polling");
                         });

} // namespace
