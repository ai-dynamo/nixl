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

// restClient concurrency, teardown and descriptor exhaustion. A single poller
// thread drives every request, so the number of open connections does not depend
// on the thread count; a loopback server that holds connections open before
// replying makes that observable.

#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "nixl_types.h"
#include "rest_accel/scality_ai_connector/client.h"

namespace gtest::obj {

// A loopback HTTP server that accepts many connections concurrently, reads each
// request, and HOLDS the connection open (no reply) until release() is called.
// Tracks the peak number of simultaneously-held connections.
class holdingTcpServer {
public:
    /// @param close_connection answer with "Connection: close", so the client
    ///        keeps no idle connection to this server.
    explicit holdingTcpServer(bool close_connection = false) : closeConnection_(close_connection) {
        listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
        EXPECT_GE(listenFd_, 0) << "socket() failed: " << strerror(errno);

        int opt = 1;
        setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};

        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0; // OS-assigned free port

        EXPECT_EQ(bind(listenFd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0)
            << "bind() failed: " << strerror(errno);
        EXPECT_EQ(listen(listenFd_, 128), 0) << "listen() failed: " << strerror(errno);

        socklen_t len = sizeof(addr);
        getsockname(listenFd_, reinterpret_cast<struct sockaddr *>(&addr), &len);
        port_ = ntohs(addr.sin_port);

        acceptThread_ = std::thread(&holdingTcpServer::acceptLoop, this);
    }

    ~holdingTcpServer() {
        stop_.store(true);
        release(); // unblock any held handlers
        if (listenFd_ >= 0) {
            ::shutdown(listenFd_, SHUT_RDWR);
            close(listenFd_);
            listenFd_ = -1;
        }
        if (acceptThread_.joinable()) {
            acceptThread_.join();
        }
        std::vector<std::thread> conns;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            conns.swap(conns_);
        }
        for (auto &t : conns) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    int
    port() const {
        return port_;
    }

    int
    maxConcurrent() const {
        return maxConcurrent_.load();
    }

    // Wait until at least n connections are simultaneously held (or timeout).
    bool
    waitUntilHeld(int n, int timeout_ms = 5000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (concurrent_.load() < n) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return true;
    }

    // Wait until at least n connections have been answered and closed.
    bool
    waitUntilClosed(int n, int timeout_ms = 5000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (closed_.load() < n) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return true;
    }

    // Let all held (and future) connections send "200 OK" and close.
    void
    release() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            released_ = true;
        }
        cv_.notify_all();
    }

private:
    void
    acceptLoop() {
        while (!stop_.load()) {
            int client_fd = accept(listenFd_, nullptr, nullptr);
            if (client_fd < 0) {
                // Out of descriptors is a test condition, not a shutdown: the
                // connection stays in the backlog until one is free again.
                if (!stop_.load() && (errno == EMFILE || errno == ENFILE)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                break; // listen socket closed
            }
            std::lock_guard<std::mutex> lk(mtx_);
            conns_.emplace_back(&holdingTcpServer::handleConn, this, client_fd);
        }
    }

    void
    handleConn(int fd) {
        std::string request;
        char buf[4096];
        while (request.find("\r\n\r\n") == std::string::npos) {
            ssize_t n = recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                close(fd);
                return;
            }
            request.append(buf, static_cast<std::string::size_type>(n));
        }

        int now = concurrent_.fetch_add(1) + 1;
        int prev_max = maxConcurrent_.load();
        while (now > prev_max && !maxConcurrent_.compare_exchange_weak(prev_max, now)) {}

        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return released_; });
        }

        const char *resp = closeConnection_ ?
            "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n" :
            "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        // The peer may already be gone (client destroyed mid-flight); avoid a
        // SIGPIPE killing the test process.
        (void)send(fd, resp, strlen(resp), MSG_NOSIGNAL);
        close(fd);
        concurrent_.fetch_sub(1);
        closed_.fetch_add(1);
    }

    const bool closeConnection_;
    int listenFd_ = -1;
    int port_ = 0;
    std::thread acceptThread_;
    std::vector<std::thread> conns_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool released_ = false;
    std::atomic<bool> stop_{false};
    std::atomic<int> concurrent_{0};
    std::atomic<int> maxConcurrent_{0};
    std::atomic<int> closed_{0};
};

// Held requests must not hit request_timeout_ms while a test is still counting
// them, so these tests set it well above anything they wait for.
static nixl_b_params_t
makeRestParams(int port, const std::string &num_threads, const std::string &max_inflight = "") {
    nixl_b_params_t params = {{"endpoint_override", "http://127.0.0.1:" + std::to_string(port)},
                              {"num_threads", num_threads},
                              {"request_timeout_ms", "60000"}};
    if (!max_inflight.empty()) {
        params["max_inflight"] = max_inflight;
    }
    return params;
}

// Submit n RDMA GETs against the client, counting completions and successes.
static void
submitGets(restClient &client,
           int n,
           std::vector<char> &buf,
           std::atomic<int> &done,
           std::atomic<int> &ok) {
    for (int i = 0; i < n; i++) {
        client.getObjectRdmaAsync("key" + std::to_string(i),
                                  reinterpret_cast<uintptr_t>(buf.data()),
                                  buf.size(),
                                  0,
                                  "rdma-descriptor",
                                  /*past_end_ok=*/false,
                                  [&](bool success) {
                                      if (success) {
                                          ok.fetch_add(1);
                                      }
                                      done.fetch_add(1);
                                  });
    }
}

static bool
waitForCount(const std::atomic<int> &count, int n, int timeout_sec) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (count.load() < n && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return count.load() >= n;
}

class scalityRestClientConcurrencyTest : public testing::Test {};

// Many requests are in flight at once even though the client has a single poller
// thread and a tiny callback pool, proving connection count is decoupled from
// thread count.
TEST_F(scalityRestClientConcurrencyTest, ManyConcurrentInFlightExceedThreadCount) {
    constexpr int num_requests = 16;
    constexpr int threshold = 8; // comfortably above 1 poller + 2 callback threads

    holdingTcpServer server;
    // num_threads sizes only the callback pool (2); it must NOT gate in-flight.
    nixl_b_params_t params = makeRestParams(server.port(), "2");
    restClient client(&params);

    std::vector<char> buf(1024);
    std::atomic<int> done{0};
    std::atomic<int> ok{0};
    submitGets(client, num_requests, buf, done, ok);

    EXPECT_TRUE(server.waitUntilHeld(threshold))
        << "only reached " << server.maxConcurrent() << " concurrent connections";
    EXPECT_GE(server.maxConcurrent(), threshold);

    server.release();

    EXPECT_TRUE(waitForCount(done, num_requests, 10)) << "not all callbacks fired";
    EXPECT_EQ(ok.load(), num_requests) << "some requests did not succeed";
}

// max_inflight bounds simultaneously-running requests. Excess requests wait in
// the pending queue and start as slots free, so all of them still complete.
TEST_F(scalityRestClientConcurrencyTest, MaxInflightCapsConcurrentConnections) {
    constexpr int num_requests = 16;
    constexpr int cap = 4;

    holdingTcpServer server;
    nixl_b_params_t params = makeRestParams(server.port(), "2", std::to_string(cap));
    restClient client(&params);

    std::vector<char> buf(1024);
    std::atomic<int> done{0};
    std::atomic<int> ok{0};
    submitGets(client, num_requests, buf, done, ok);

    // The cap should be reached but never exceeded. Give any surplus request time
    // to arrive so the upper-bound assertion is meaningful rather than racy.
    EXPECT_TRUE(server.waitUntilHeld(cap))
        << "never reached the cap; only " << server.maxConcurrent() << " concurrent";
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_LE(server.maxConcurrent(), cap)
        << "exceeded max_inflight=" << cap << " with " << server.maxConcurrent();

    // Releasing lets the held requests finish, which frees slots for the queued
    // remainder; every request must eventually complete.
    server.release();

    EXPECT_TRUE(waitForCount(done, num_requests, 20)) << "queued requests never ran";
    EXPECT_EQ(ok.load(), num_requests) << "some requests did not succeed";
    EXPECT_LE(server.maxConcurrent(), cap) << "cap was exceeded while draining";
}

// Teardown must also fail requests that are still waiting for an in-flight slot,
// not just the ones already running.
TEST_F(scalityRestClientConcurrencyTest, TeardownWithPendingRequestsFiresAllCallbacks) {
    constexpr int num_requests = 16;
    constexpr int cap = 2;

    holdingTcpServer server; // never released: the running requests stay stuck
    nixl_b_params_t params = makeRestParams(server.port(), "2", std::to_string(cap));

    std::vector<char> buf(1024);
    std::atomic<int> done{0};
    std::atomic<int> ok{0};

    {
        restClient client(&params);
        submitGets(client, num_requests, buf, done, ok);
        // cap requests are running; the other num_requests - cap are queued.
        ASSERT_TRUE(server.waitUntilHeld(cap)) << "no request reached the server";
    }

    EXPECT_EQ(done.load(), num_requests) << "pending requests were dropped without a callback";
    EXPECT_EQ(ok.load(), 0) << "aborted requests should report failure";
}

// Destroying the client while requests are stuck in flight must return promptly
// and fire every callback exactly once, with failure.
TEST_F(scalityRestClientConcurrencyTest, TeardownWithOutstandingRequestsFiresAllCallbacks) {
    constexpr int num_requests = 8;

    holdingTcpServer server; // never released: requests stay in flight
    nixl_b_params_t params = makeRestParams(server.port(), "2");

    std::vector<char> buf(1024);
    std::atomic<int> done{0};
    std::atomic<int> ok{0};

    auto client = std::make_unique<restClient>(&params);
    submitGets(*client, num_requests, buf, done, ok);
    ASSERT_TRUE(server.waitUntilHeld(1)) << "no request reached the server";

    const auto start = std::chrono::steady_clock::now();
    client.reset();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    // The destructor drains the callback pool, so every callback has fired.
    EXPECT_LT(elapsed.count(), 2000) << "teardown waited on the stuck requests";
    EXPECT_EQ(done.load(), num_requests) << "not all callbacks fired on teardown";
    EXPECT_EQ(ok.load(), 0) << "aborted requests should report failure";
}

// Holds every file descriptor the process may still open, under a soft
// RLIMIT_NOFILE lowered to just above the current count. Restores both on
// destruction, so a failing assertion cannot leak the state into other tests.
class descriptorSqueeze {
public:
    descriptorSqueeze() {
        getrlimit(RLIMIT_NOFILE, &saved_);
        rlimit squeezed = saved_;
        squeezed.rlim_cur = openCount() + 16;
        setrlimit(RLIMIT_NOFILE, &squeezed);
        for (int fd; (fd = open("/dev/null", O_RDONLY)) >= 0;) {
            held_.push_back(fd);
        }
    }

    ~descriptorSqueeze() {
        releaseAll();
        setrlimit(RLIMIT_NOFILE, &saved_);
    }

    void
    releaseAll() {
        for (int fd : held_) {
            close(fd);
        }
        held_.clear();
    }

private:
    static rlim_t
    openCount() {
        rlim_t n = 0;
        if (DIR *dir = opendir("/proc/self/fd")) {
            while (readdir(dir) != nullptr) {
                n++;
            }
            closedir(dir);
        }
        return n;
    }

    rlimit saved_{};
    std::vector<int> held_;
};

static bool
bodyReadSucceeds(restClient &client, int port) {
    std::vector<char> buf(16);
    std::promise<bool> result;
    auto future = result.get_future();
    client.getObjectBodyAsync(
        "k", buf.data(), buf.size(), 0, [&](bool ok) { result.set_value(ok); });
    if (future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
        ADD_FAILURE() << "callback never fired (port " << port << ")";
        return false;
    }
    return future.get();
}

// Running out of file descriptors fails requests at connect, and the client
// recovers once descriptors are free again instead of failing every later
// request without trying.
TEST_F(scalityRestClientConcurrencyTest, RecoversAfterDescriptorsAreReleased) {
    // Connection: close, so the client keeps no idle socket that it would free
    // during the squeeze, and waiting for the server to close its side means no
    // descriptor is released behind the squeeze's back either.
    holdingTcpServer server(/*close_connection=*/true);
    server.release(); // answer at once
    nixl_b_params_t params = makeRestParams(server.port(), "2", "4");
    restClient client(&params);

    ASSERT_TRUE(bodyReadSucceeds(client, server.port())) << "baseline read failed";
    ASSERT_TRUE(server.waitUntilClosed(1)) << "the server never closed the baseline connection";

    {
        descriptorSqueeze squeeze;
        EXPECT_FALSE(bodyReadSucceeds(client, server.port()))
            << "a read with no descriptor left must fail at connect";
        squeeze.releaseAll();
        for (int i = 0; i < 3; ++i) {
            EXPECT_TRUE(bodyReadSucceeds(client, server.port()))
                << "read " << i << " after the descriptors were released failed";
        }
    }
}

} // namespace gtest::obj
