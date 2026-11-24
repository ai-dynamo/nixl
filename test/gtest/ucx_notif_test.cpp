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

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

// Far more than the sockets take from an idle receiver: about 256 KiB with the
// 64 KiB buffers set below
constexpr size_t large_payload_size = 16 * 1024 * 1024;

// A sender and a receiver worker of one UCX context, and a transfer with no
// data and a large notification. The flush of the transfer completes without
// the receiver, and the idle receiver then holds back the send of the
// notification.
class ucxNotification : public ::testing::Test {
protected:
    void
    SetUp() override {
        // On a host with NVIDIA GPUs, NIXL rejects transports without CUDA support
        const std::string tls =
            (nixl::hwInfo::instance().numNvidiaGpus == 0) ? "tcp" : "tcp,cuda_copy";
        env_.addVar("UCX_TLS", tls);
        env_.addVar("UCX_ZCOPY_THRESH", zcopyThresh());
        // Small socket buffers keep a large send in progress while the
        // receiver does not progress
        env_.addVar("UCX_TCP_SNDBUF", "65536");
        env_.addVar("UCX_TCP_RCVBUF", "65536");
        context_ = std::make_unique<nixlUcxContext>(
            std::vector<std::string>{}, false, 2, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE, 0, "");
        receiver_ = std::make_unique<nixlUcxWorker>(*context_, UCP_ERR_HANDLING_MODE_PEER);
        ASSERT_EQ(UCS_OK,
                  receiver_->regAmCallback(nixl::ucx::am_cb_op_t::NOTIF_STR, receive, &received_));
        sender_ = std::make_unique<nixlUcxWorker>(*context_, UCP_ERR_HANDLING_MODE_PEER);
        auto address = receiver_->epAddr();
        ep_ = sender_->connect(address.data());
        ASSERT_NE(nullptr, ep_);
        ASSERT_NO_FATAL_FAILURE(establish());
    }

    // UCX_ZCOPY_THRESH of the sends: the notification is sent zero-copy
    [[nodiscard]] virtual std::string
    zcopyThresh() const {
        return "0";
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

    // Whether the sockets hold back a large send while the receiver does not
    // progress, seen on a plain UCP send rather than on the transfer under test
    void
    probeHold() {
        const std::string payload(large_payload_size, 'p');
        ucp_request_param_t param;
        param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        param.flags = UCP_AM_SEND_FLAG_EAGER;
        const ucs_status_ptr_t request = ucp_am_send_nbx(ep_->getEp(),
                                                         unsigned(nixl::ucx::am_cb_op_t::NOTIF_STR),
                                                         nullptr,
                                                         0,
                                                         payload.data(),
                                                         payload.size(),
                                                         &param);
        ASSERT_FALSE(UCS_PTR_IS_ERR(request)) << ucs_status_string(UCS_PTR_STATUS(request));
        const auto in_progress = [request] {
            return UCS_PTR_IS_PTR(request) && (ucp_request_check_status(request) == UCS_INPROGRESS);
        };
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (in_progress() && (std::chrono::steady_clock::now() < deadline)) {
            sender_->progress();
        }
        heldBack_ = in_progress();
        progressUntilReceived(payload);
        if (UCS_PTR_IS_PTR(request)) {
            EXPECT_EQ(UCS_OK, ucp_request_check_status(request));
            ucp_request_free(request);
        }
        received_.clear();
    }

    // Exchange a message, see whether the sockets hold back a large send, then
    // leave the receiver idle
    void
    establish() {
        const nixl_status_t status =
            ep_->sendAm(nixl::ucx::am_cb_op_t::NOTIF_STR, "ready", UCP_AM_SEND_FLAG_EAGER);
        ASSERT_TRUE(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
        ASSERT_NO_FATAL_FAILURE(progressUntilReceived("ready"));
        received_.clear();
        ASSERT_NO_FATAL_FAILURE(probeHold());
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

    // Start the transfer, and wait until only the send of its notification
    // is in progress. Skips the test where the sockets take a large send
    // without the receiver: there is nothing to hold.
    void
    holdNotification(const std::string &payload) {
        ASSERT_EQ(large_payload_size, payload.size());
        if (!heldBack_) {
            GTEST_SKIP() << "the sockets take a send of " << payload.size()
                         << " bytes without the receiver";
        }
        transfer_.reset(new nixlUcxBackendReqH(sender_.get()));
        transfer_->setNotif(std::string(payload));
        transfer_->init({}, *ep_);
        ASSERT_EQ(NIXL_SUCCESS, transfer_->flush());
        ASSERT_TRUE(waitUntil([&] {
            return (transfer_->status() == NIXL_IN_PROG) && !transfer_->hasPendingRequests();
        }));
    }

    // Releases a transfer and then destroys it, as the engine does
    struct releaseTransfer {
        void
        operator()(nixlUcxBackendReqH *transfer) const {
            transfer->release();
            delete transfer;
        }
    };

    gtest::ScopedEnv env_;
    std::string received_;
    // Whether the sockets hold back a large send while the receiver does not
    // progress
    bool heldBack_ = false;
    std::unique_ptr<nixlUcxContext> context_;
    std::unique_ptr<nixlUcxWorker> receiver_;
    std::unique_ptr<nixlUcxWorker> sender_;
    std::unique_ptr<nixlUcxEp> ep_;
    // Released before the endpoint and the worker go, also when a failed
    // assertion ends a test early: a flush in progress must not call back
    // into a destroyed transfer
    std::unique_ptr<nixlUcxBackendReqH, releaseTransfer> transfer_;
};

TEST_F(ucxNotification, TransferIsDoneOnceNotificationIsSent) {
    const std::string payload(large_payload_size, 'n');
    ASSERT_NO_FATAL_FAILURE(holdNotification(payload));
    if (IsSkipped()) {
        return;
    }

    nixl_status_t status = NIXL_IN_PROG;
    ASSERT_TRUE(waitUntil([&] {
        receiver_->progress();
        return (status = transfer_->status()) != NIXL_IN_PROG;
    }));
    EXPECT_EQ(NIXL_SUCCESS, status);
    transfer_.reset();
    ASSERT_NO_FATAL_FAILURE(progressUntilReceived(payload));
}

TEST_F(ucxNotification, ReleaseLeavesNotificationSendToItsCallback) {
    const std::string payload(large_payload_size, 'n');
    ASSERT_NO_FATAL_FAILURE(holdNotification(payload));
    if (IsSkipped()) {
        return;
    }

    // Release the transfer before it is done, as the engine does. Only the
    // send of its notification is in progress: it goes on, and its callback
    // releases the request and the payload.
    const gtest::LogIgnoreGuard in_progress(
        "request handle .* released with .* requests in progress");
    transfer_.reset();
    ASSERT_NO_FATAL_FAILURE(progressUntilReceived(payload));
    EXPECT_EQ(0, in_progress.getIgnoredCount());
}

// The notification is sent with bcopy: when the connection fails while a
// zero-copy send is partly written, UCX TCP can complete it twice
// (openucx/ucx#12376)
class ucxNotificationBcopy : public ucxNotification {
protected:
    [[nodiscard]] std::string
    zcopyThresh() const override {
        return "inf";
    }
};

TEST_F(ucxNotificationBcopy, FailedNotificationSendFailsTransfer) {
    ASSERT_NO_FATAL_FAILURE(holdNotification(std::string(large_payload_size, 'f')));
    if (IsSkipped()) {
        return;
    }

    // The receiver goes away while it holds back the notification. The send
    // fails, and so does the transfer.
    const gtest::LogIgnoreGuard send_failed("UCX AM send failed with status");
    receiver_.reset();
    nixl_status_t status = NIXL_IN_PROG;
    ASSERT_TRUE(waitUntil([&] { return (status = transfer_->status()) != NIXL_IN_PROG; }));
    EXPECT_EQ(NIXL_ERR_REMOTE_DISCONNECT, status);
}

} // namespace
