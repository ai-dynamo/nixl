/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "redis_client.h"
#include "common/scoped_fd.h"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <functional>
#include <future>
#include <netinet/in.h>
#include <sys/socket.h>
#include <system_error>
#include <thread>

namespace {

using namespace std::chrono_literals;

// These wrappers only control the submitting thread and observe the real libevent loop.
thread_local std::function<void()> before_submit;
std::promise<int> *loop_result = nullptr;

class eventLoopResult {
public:
    eventLoopResult() : result_(promise_.get_future()) {
        loop_result = &promise_;
    }

    ~eventLoopResult() {
        loop_result = nullptr;
    }

    std::future<int> &
    result() {
        return result_;
    }

private:
    std::promise<int> promise_;
    std::future<int> result_;
};

class pausedSubmission {
public:
    explicit pausedSubmission(std::function<void()> submit)
        : paused_(pausedPromise_.get_future()),
          resume_(resumePromise_.get_future()),
          thread_([this, submit = std::move(submit)] {
              before_submit = [this] {
                  pausedPromise_.set_value();
                  resume_.wait();
              };
              submit();
              before_submit = {};
          }) {}

    ~pausedSubmission() {
        resume();
    }

    bool
    wait() {
        return paused_.wait_for(5s) == std::future_status::ready;
    }

    void
    resume() {
        if (thread_.joinable()) {
            resumePromise_.set_value();
            thread_.join();
        }
    }

private:
    std::promise<void> pausedPromise_;
    std::promise<void> resumePromise_;
    std::future<void> paused_;
    std::future<void> resume_;
    std::thread thread_;
};

class loopbackPeer {
public:
    loopbackPeer() : listener_(::socket(AF_INET, SOCK_STREAM, 0)) {
        if (!listener_.valid()) {
            throw std::system_error(errno, std::generic_category(), "socket");
        }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(listener_.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
            throw std::system_error(errno, std::generic_category(), "bind");
        }
        socklen_t size = sizeof(address);
        if (::getsockname(listener_.get(), reinterpret_cast<sockaddr *>(&address), &size) != 0) {
            throw std::system_error(errno, std::generic_category(), "getsockname");
        }
        config_.host = "127.0.0.1";
        config_.port = ntohs(address.sin_port);
        config_.pool_size = 1;
    }

    void
    listen() {
        if (::listen(listener_.get(), 2) != 0) {
            throw std::system_error(errno, std::generic_category(), "listen");
        }
    }

    void
    acceptAsyncConnection() {
        connection_ = nixl::scopedFd(::accept(listener_.get(), nullptr, nullptr));
        if (!connection_.valid()) {
            throw std::system_error(errno, std::generic_category(), "accept");
        }
    }

    void
    disconnect() {
        ::shutdown(connection_.get(), SHUT_RDWR);
        connection_.reset();
    }

    const RedisConfig &
    config() const {
        return config_;
    }

private:
    nixl::scopedFd listener_;
    nixl::scopedFd connection_;
    RedisConfig config_;
};

class redisEventLoopTest : public ::testing::TestWithParam<nixl_xfer_op_t> {
protected:
    void
    submit(RedisConnectionPool &pool, std::shared_ptr<std::promise<nixl_status_t>> promise) {
        const auto address = reinterpret_cast<uintptr_t>(buffer_.data());
        if (GetParam() == NIXL_READ) {
            pool.getKeyAsync("key", address, buffer_.size(), std::move(promise));
        } else {
            pool.putKeyAsync("key", address, buffer_.size(), std::move(promise));
        }
    }

    std::array<char, 8> buffer_{};
};

TEST_P(redisEventLoopTest, CompletesSubmissionQueuedAfterLastConnectionCloses) {
    eventLoopResult loop;
    loopbackPeer peer;
    peer.listen();
    auto pool = std::make_unique<RedisConnectionPool>(peer.config());
    peer.acceptAsyncConnection();

    auto in_flight = std::make_shared<std::promise<nixl_status_t>>();
    auto in_flight_result = in_flight->get_future();
    submit(*pool, in_flight);

    auto pending = std::make_shared<std::promise<nixl_status_t>>();
    auto pending_result = pending->get_future();
    pausedSubmission submission([&] { submit(*pool, pending); });
    ASSERT_TRUE(submission.wait());

    peer.disconnect();
    ASSERT_EQ(in_flight_result.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(in_flight_result.get(), NIXL_ERR_BACKEND);
    // Dispatch without NO_EXIT_ON_EMPTY returns here once hiredis removes its socket events.
    // The persistent loop stays available to consume the submission below.
    EXPECT_EQ(loop.result().wait_for(1s), std::future_status::timeout);

    submission.resume();
    pending.reset();
    ASSERT_EQ(pending_result.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(pending_result.get(), NIXL_ERR_BACKEND);

    pool.reset();
    ASSERT_EQ(loop.result().wait_for(0s), std::future_status::ready);
    EXPECT_EQ(loop.result().get(), 0);
}

TEST_P(redisEventLoopTest, ShutdownCompletesOutstandingCommands) {
    eventLoopResult loop;
    loopbackPeer peer;
    peer.listen();
    auto pool = std::make_unique<RedisConnectionPool>(peer.config());
    peer.acceptAsyncConnection();

    auto pending = std::make_shared<std::promise<nixl_status_t>>();
    auto result = pending->get_future();
    submit(*pool, pending);
    pending.reset();
    EXPECT_EQ(result.wait_for(0s), std::future_status::timeout);

    pool.reset();
    ASSERT_EQ(result.wait_for(0s), std::future_status::ready);
    EXPECT_EQ(result.get(), NIXL_ERR_BACKEND);
    ASSERT_EQ(loop.result().wait_for(0s), std::future_status::ready);
    EXPECT_EQ(loop.result().get(), 0);
}

INSTANTIATE_TEST_SUITE_P(Transfers, redisEventLoopTest, ::testing::Values(NIXL_READ, NIXL_WRITE));

TEST(redisEventLoopFailureTest, ConnectionRefusedStopsTheLoop) {
    eventLoopResult loop;
    loopbackPeer peer; // The bound port stays reserved without accepting connections.
    EXPECT_THROW(RedisConnectionPool{peer.config()}, std::runtime_error);
    EXPECT_EQ(loop.result().wait_for(0s), std::future_status::ready);
}

} // namespace

extern "C" {

int
__real_event_base_once(event_base *,
                       evutil_socket_t,
                       short,
                       event_callback_fn,
                       void *,
                       const timeval *);
int
__real_event_base_dispatch(event_base *);
int
__real_event_base_loop(event_base *, int);

int
__wrap_event_base_once(event_base *base,
                       evutil_socket_t fd,
                       short events,
                       event_callback_fn callback,
                       void *arg,
                       const timeval *timeout) {
    if (before_submit) {
        before_submit();
    }
    return __real_event_base_once(base, fd, events, callback, arg, timeout);
}

int
__wrap_event_base_dispatch(event_base *base) {
    const int result = __real_event_base_dispatch(base);
    loop_result->set_value(result);
    return result;
}

int
__wrap_event_base_loop(event_base *base, int flags) {
    const int result = __real_event_base_loop(base, flags);
    loop_result->set_value(result);
    return result;
}

} // extern "C"
