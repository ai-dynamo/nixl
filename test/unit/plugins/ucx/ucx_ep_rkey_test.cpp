/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <vector>

#include <gtest/gtest.h>

#include "ucx_utils.h"

namespace {

class CoordinatedEpOps {
public:
    nixlUcxEpOps
    ops() {
        return {
            .context = this,
            .rkeyUnpack = [](void *context,
                             ucp_ep_h,
                             const void *,
                             ucp_rkey_h *rkey) {
                return static_cast<CoordinatedEpOps *>(context)->unpack(rkey);
            },
            .closeNb = [](void *context, ucp_ep_h ep, unsigned mode) {
                return static_cast<CoordinatedEpOps *>(context)->close(ep, mode);
            },
        };
    }

    ucs_status_t
    unpack(ucp_rkey_h *rkey) {
        std::unique_lock<std::mutex> lock(mutex_);
        ++unpackCalls_;
        unpackActive_ = true;
        unpackEntered_.notify_one();
        releaseUnpack_.wait(lock, [this] { return releaseUnpackRequested_; });
        unpackActive_ = false;
        *rkey = reinterpret_cast<ucp_rkey_h>(1);
        return UCS_OK;
    }

    ucs_status_ptr_t
    close(ucp_ep_h ep, unsigned mode) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closeEntered_ = true;
            closeOverlappedUnpack_ = unpackActive_;
        }
        closeEnteredCv_.notify_one();
        return ucp_ep_close_nb(ep, mode);
    }

    void
    waitForUnpack() {
        std::unique_lock<std::mutex> lock(mutex_);
        unpackEntered_.wait(lock, [this] { return unpackActive_; });
    }

    bool
    waitForClose(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return closeEnteredCv_.wait_for(lock, timeout, [this] { return closeEntered_; });
    }

    void
    releaseUnpack() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            releaseUnpackRequested_ = true;
        }
        releaseUnpack_.notify_one();
    }

    bool
    closeOverlappedUnpack() {
        std::lock_guard<std::mutex> lock(mutex_);
        return closeOverlappedUnpack_;
    }

    size_t
    unpackCalls() {
        std::lock_guard<std::mutex> lock(mutex_);
        return unpackCalls_;
    }

private:
    std::mutex mutex_;
    std::condition_variable unpackEntered_;
    std::condition_variable releaseUnpack_;
    std::condition_variable closeEnteredCv_;
    bool unpackActive_ = false;
    bool releaseUnpackRequested_ = false;
    bool closeEntered_ = false;
    bool closeOverlappedUnpack_ = false;
    size_t unpackCalls_ = 0;
};

TEST(UcxEndpoint, CloseDuringRkeyUnpackIsSerialized) {
    std::vector<std::string> devices;
    nixlUcxContext contexts[] = {
        {devices, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 1},
        {devices, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 1},
    };
    CoordinatedEpOps coordinated;
    nixlUcxWorker consumer(
        contexts[0], UCP_ERR_HANDLING_MODE_PEER, 0, coordinated.ops());
    nixlUcxWorker producer(contexts[1], UCP_ERR_HANDLING_MODE_PEER);
    std::string producerAddress = producer.epAddr();
    auto endpoint = consumer.connect(producerAddress.data(), producerAddress.size());
    ASSERT_NE(endpoint, nullptr);

    ucp_rkey_h rkey = nullptr;
    auto unpack = std::async(std::launch::async, [&] {
        return endpoint->unpackRkey(nullptr, &rkey);
    });
    coordinated.waitForUnpack();

    const ucp_ep_h nativeEndpoint = endpoint->getEp();
    auto reportError = std::async(std::launch::async, [&] {
        endpoint->err_cb(nativeEndpoint, UCS_ERR_CONNECTION_RESET);
    });

    EXPECT_FALSE(coordinated.waitForClose(std::chrono::milliseconds(20)))
        << "endpoint close overlapped active rkey unpack";
    coordinated.releaseUnpack();

    EXPECT_EQ(unpack.get(), NIXL_SUCCESS);
    reportError.get();
    EXPECT_TRUE(coordinated.waitForClose(std::chrono::seconds(1)));
    EXPECT_FALSE(coordinated.closeOverlappedUnpack())
        << "endpoint close overlapped active rkey unpack";

    ucp_rkey_h unused = nullptr;
    EXPECT_EQ(endpoint->unpackRkey(nullptr, &unused), NIXL_ERR_REMOTE_DISCONNECT);
}

TEST(UcxEndpoint, RkeyUnpackAfterEndpointFailureReturnsRemoteDisconnect) {
    std::vector<std::string> devices;
    nixlUcxContext contexts[] = {
        {devices, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 1},
        {devices, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 1},
    };
    CoordinatedEpOps coordinated;
    nixlUcxWorker consumer(
        contexts[0], UCP_ERR_HANDLING_MODE_PEER, 0, coordinated.ops());
    nixlUcxWorker producer(contexts[1], UCP_ERR_HANDLING_MODE_PEER);
    std::string producerAddress = producer.epAddr();
    auto endpoint = consumer.connect(producerAddress.data(), producerAddress.size());
    ASSERT_NE(endpoint, nullptr);

    endpoint->err_cb(endpoint->getEp(), UCS_ERR_CONNECTION_RESET);
    ASSERT_TRUE(coordinated.waitForClose(std::chrono::seconds(1)));

    ucp_rkey_h rkey = nullptr;
    EXPECT_EQ(endpoint->unpackRkey(nullptr, &rkey), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(rkey, nullptr);
    EXPECT_EQ(coordinated.unpackCalls(), 0);
}

} // namespace

int
main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
