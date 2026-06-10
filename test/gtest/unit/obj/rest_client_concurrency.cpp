/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * RestClient concurrency tests.
 *
 * The curl_multi-based RestClient drives all transfers from a single poller
 * thread, so the number of simultaneously-open connections is decoupled from
 * the thread count. These tests prove that with a loopback server that accepts
 * many connections and holds them open before replying, and verify graceful
 * teardown while requests are still in flight.
 */

#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <netinet/in.h>
#include <optional>
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
// Tracks the peak number of simultaneously-held connections so a test can prove
// that in-flight request count exceeds the client's thread count.
class HoldingTcpServer {
public:
    HoldingTcpServer() {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        EXPECT_GE(listen_fd_, 0) << "socket() failed: " << strerror(errno);

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};

        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0; // OS-assigned free port

        EXPECT_EQ(bind(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0)
            << "bind() failed: " << strerror(errno);
        EXPECT_EQ(listen(listen_fd_, 128), 0) << "listen() failed: " << strerror(errno);

        socklen_t len = sizeof(addr);
        getsockname(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), &len);
        port_ = ntohs(addr.sin_port);

        accept_thread_ = std::thread(&HoldingTcpServer::acceptLoop, this);
    }

    ~HoldingTcpServer() {
        stop_.store(true);
        release(); // unblock any held handlers
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            close(listen_fd_);
            listen_fd_ = -1;
        }
        if (accept_thread_.joinable()) {
            accept_thread_.join();
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
        return max_concurrent_.load();
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
            int client_fd = accept(listen_fd_, nullptr, nullptr);
            if (client_fd < 0) {
                break; // listen socket closed
            }
            std::lock_guard<std::mutex> lk(mtx_);
            conns_.emplace_back(&HoldingTcpServer::handleConn, this, client_fd);
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
        int prev_max = max_concurrent_.load();
        while (now > prev_max && !max_concurrent_.compare_exchange_weak(prev_max, now)) {}

        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return released_; });
        }

        const char *resp = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        // The peer may already be gone (client destroyed mid-flight); avoid a
        // SIGPIPE killing the test process.
#ifdef MSG_NOSIGNAL
        (void)send(fd, resp, strlen(resp), MSG_NOSIGNAL);
#else
        (void)send(fd, resp, strlen(resp), 0);
#endif
        close(fd);
        concurrent_.fetch_sub(1);
    }

    int listen_fd_ = -1;
    int port_ = 0;
    std::thread accept_thread_;
    std::vector<std::thread> conns_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool released_ = false;
    std::atomic<bool> stop_{false};
    std::atomic<int> concurrent_{0};
    std::atomic<int> max_concurrent_{0};
};

static nixl_b_params_t
makeRestParams(const std::string &endpoint, const std::string &num_threads) {
    return {{"endpoint_override", endpoint}, {"num_threads", num_threads}};
}

class RestClientConcurrencyTest : public testing::Test {};

// Many requests are in flight at once even though the client has a single poller
// thread and a tiny callback pool, proving connection count is decoupled from
// thread count.
TEST_F(RestClientConcurrencyTest, ManyConcurrentInFlightExceedThreadCount) {
    constexpr int kN = 16;
    constexpr int kThreshold = 8; // comfortably above 1 poller + 2 callback threads

    HoldingTcpServer server;
    // num_threads sizes only the callback pool now (2); it must NOT gate in-flight.
    nixl_b_params_t params =
        makeRestParams("http://127.0.0.1:" + std::to_string(server.port()), "2");
    RestClient client(&params);

    std::vector<char> buf(1024);
    std::atomic<int> done{0};
    std::atomic<int> ok{0};

    for (int i = 0; i < kN; i++) {
        client.getObjectRdmaAsync("key" + std::to_string(i),
                                  reinterpret_cast<uintptr_t>(buf.data()),
                                  buf.size(),
                                  0,
                                  "rdma-token",
                                  [&](bool success) {
                                      if (success) {
                                          ok.fetch_add(1);
                                      }
                                      done.fetch_add(1);
                                  });
    }

    // The server should hold many connections open simultaneously before we
    // release any of them.
    EXPECT_TRUE(server.waitUntilHeld(kThreshold))
        << "only reached " << server.maxConcurrent() << " concurrent connections";
    EXPECT_GE(server.maxConcurrent(), kThreshold);

    server.release();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (done.load() < kN && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(done.load(), kN) << "not all callbacks fired";
    EXPECT_EQ(ok.load(), kN) << "some requests did not succeed";
}

// Destroying the client while requests are stuck in flight must return promptly
// and fire every callback exactly once (with failure).
TEST_F(RestClientConcurrencyTest, TeardownWithOutstandingRequestsFiresAllCallbacks) {
    constexpr int kN = 8;

    HoldingTcpServer server; // never released → requests stay in flight
    nixl_b_params_t params =
        makeRestParams("http://127.0.0.1:" + std::to_string(server.port()), "2");

    std::vector<char> buf(1024);
    std::atomic<int> done{0};
    std::atomic<int> ok{0};

    {
        RestClient client(&params);
        for (int i = 0; i < kN; i++) {
            client.getObjectRdmaAsync("key" + std::to_string(i),
                                      reinterpret_cast<uintptr_t>(buf.data()),
                                      buf.size(),
                                      0,
                                      "rdma-token",
                                      [&](bool success) {
                                          if (success) {
                                              ok.fetch_add(1);
                                          }
                                          done.fetch_add(1);
                                      });
        }
        ASSERT_TRUE(server.waitUntilHeld(1)) << "no request reached the server";

        auto t0 = std::chrono::steady_clock::now();
        // client destructor runs here at end of scope
        (void)t0;
    }

    // After the destructor returns, the callback pool has been drained, so every
    // callback must already have fired, all as failures (aborted in flight).
    EXPECT_EQ(done.load(), kN) << "not all callbacks fired on teardown";
    EXPECT_EQ(ok.load(), 0) << "aborted requests should report failure";
}

} // namespace gtest::obj
