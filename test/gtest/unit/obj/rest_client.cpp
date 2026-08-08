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
 * RestClient HTTP wire-format tests.
 *
 * Confirms that RestClient sends correctly-formed HTTP requests to the
 * connector endpoint without any RDMA hardware, cuObject, or nvidia-fs: each
 * test starts a one-shot loopback TCP server, points a RestClient at it, issues
 * a request with a fake RDMA descriptor, and asserts on the raw HTTP captured.
 */

#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <netinet/in.h>
#include <optional>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "nixl_types.h"
#include "rest_accel/scality_ai_connector/rest_client.h"

namespace gtest::obj {

// A throwaway single-request HTTP server: binds to localhost:0, reads one full
// HTTP request, replies "200 OK", and hands the raw request text back.
class TcpServer {
public:
    explicit TcpServer(int status_code = 200, std::string reason = "OK")
        : status_code_(status_code),
          reason_(std::move(reason)) {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        EXPECT_GE(listen_fd_, 0) << "socket() failed: " << strerror(errno);

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};

        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0; // let the OS assign a free port

        EXPECT_EQ(bind(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0)
            << "bind() failed: " << strerror(errno);
        EXPECT_EQ(listen(listen_fd_, 1), 0) << "listen() failed: " << strerror(errno);

        socklen_t len = sizeof(addr);
        getsockname(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), &len);
        port_ = ntohs(addr.sin_port);

        future_ = promise_.get_future();
        thread_ = std::thread(&TcpServer::acceptAndRead, this);
    }

    ~TcpServer() {
        if (listen_fd_ >= 0) {
            close(listen_fd_);
        }
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    int
    port() const {
        return port_;
    }

    // Block until one full HTTP request is captured (or timeout); "" on timeout.
    std::string
    capturedRequest(int timeout_sec = 5) {
        if (future_.wait_for(std::chrono::seconds(timeout_sec)) == std::future_status::timeout) {
            return "";
        }
        return future_.get();
    }

private:
    void
    acceptAndRead() {
        int client_fd = accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            promise_.set_value("");
            return;
        }

        std::string request;
        char buf[4096];
        while (request.find("\r\n\r\n") == std::string::npos) {
            ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                break;
            }
            request.append(buf, static_cast<std::string::size_type>(n));
        }

        std::string response = "HTTP/1.1 " + std::to_string(status_code_) + " " + reason_ +
            "\r\nContent-Length: 0\r\n\r\n";
        send(client_fd, response.data(), response.size(), 0);
        close(client_fd);

        promise_.set_value(request);
    }

    int status_code_;
    std::string reason_;
    int listen_fd_ = -1;
    int port_ = 0;
    std::thread thread_;
    std::promise<std::string> promise_;
    std::future<std::string> future_;
};

static nixl_b_params_t
makeRestParams(const std::string &endpoint) {
    return {{"endpoint_override", endpoint}};
}

// The async calls dispatch to an internal thread pool and return immediately;
// spin-wait until the callback fires (or timeout).
static void
waitForCallback(const std::atomic<bool> &done, int timeout_ms = 5000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!done.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

class RestClientTest : public testing::Test {};

TEST_F(RestClientTest, PutSendsCorrectUrlAndHeaders) {
    TcpServer server;
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    const std::string rdma_desc = "fake-rdma-descriptor-for-testing";
    const size_t offset = 512;
    const size_t data_len = 1024;
    std::vector<char> buf(data_len);

    std::atomic<bool> done{false}, success{false};
    client.putObjectRdmaAsync("mykey",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              data_len,
                              offset,
                              rdma_desc,
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    std::string req = server.capturedRequest();
    waitForCallback(done);

    EXPECT_NE(req.find("PUT /mykey"), std::string::npos) << "Expected 'PUT /mykey' in request:\n"
                                                         << req;
    EXPECT_NE(req.find("x-scal-rdma: " + rdma_desc), std::string::npos)
        << "x-scal-rdma header missing or wrong in:\n"
        << req;
    // Body must be empty; the data moves via RDMA, not the HTTP body.
    EXPECT_NE(req.find("Content-Length: 0"), std::string::npos) << "Content-Length: 0 missing in:\n"
                                                                << req;
    EXPECT_TRUE(success.load()) << "putObjectRdmaAsync reported failure";
}

TEST_F(RestClientTest, GetSendsCorrectUrlAndHeaders) {
    TcpServer server;
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    const size_t offset = 128;
    std::vector<char> buf(512);

    std::atomic<bool> done{false}, success{false};
    client.getObjectRdmaAsync("readkey",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              offset,
                              "rdma-get-token",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    std::string req = server.capturedRequest();
    waitForCallback(done);

    EXPECT_NE(req.find("GET /readkey"), std::string::npos) << "Expected 'GET /readkey' in:\n"
                                                           << req;
    EXPECT_EQ(req.find("PUT"), std::string::npos) << "GET request must not contain PUT method in:\n"
                                                  << req;
    EXPECT_NE(req.find("x-scal-rdma: rdma-get-token"), std::string::npos)
        << "x-scal-rdma missing in:\n"
        << req;
    EXPECT_TRUE(success.load()) << "getObjectRdmaAsync reported failure";
}

TEST_F(RestClientTest, PutRejectsEmptyRdmaDesc) {
    // Port 1 won't accept connections, so a stray connection attempt fails loudly.
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    RestClient client(&params);

    std::vector<char> buf(64);
    std::atomic<bool> done{false}, success{true};

    client.putObjectRdmaAsync("k",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              0,
                              /*rdma_desc=*/"",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with empty rdma_desc";
}

TEST_F(RestClientTest, PutRejectsZeroDataLen) {
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    RestClient client(&params);

    std::atomic<bool> done{false}, success{true};
    client.putObjectRdmaAsync("k",
                              /*data_ptr=*/0,
                              /*data_len=*/0,
                              0,
                              "some-token",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with data_len=0";
}

TEST_F(RestClientTest, GetRejectsEmptyRdmaDesc) {
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    RestClient client(&params);

    std::vector<char> buf(64);
    std::atomic<bool> done{false}, success{true};
    client.getObjectRdmaAsync("k",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              0,
                              /*rdma_desc=*/"",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with empty rdma_desc";
}

TEST_F(RestClientTest, ConstructorThrowsOnMissingEndpoint) {
    nixl_b_params_t params = {}; // intentionally empty
    EXPECT_THROW({ RestClient c(&params); }, std::invalid_argument);
}

TEST_F(RestClientTest, ConstructorThrowsOnNullParams) {
    EXPECT_THROW({ RestClient c(nullptr); }, std::invalid_argument);
}

// ─── Existence check (HTTP HEAD) tests ────────────────────────────────────────

TEST_F(RestClientTest, CheckExistsReturnsTrueOn200) {
    TcpServer server(200, "OK");
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    std::atomic<bool> done{false};
    std::optional<bool> result;
    client.checkObjectExistsAsync("mykey", [&](std::optional<bool> exists) {
        result = exists;
        done = true;
    });

    std::string req = server.capturedRequest();
    waitForCallback(done);

    EXPECT_NE(req.find("HEAD /mykey"), std::string::npos) << "Expected 'HEAD /mykey' in:\n" << req;
    ASSERT_TRUE(result.has_value()) << "Expected a definite result, got error";
    EXPECT_TRUE(*result) << "Expected exists=true for HTTP 200";
}

TEST_F(RestClientTest, CheckExistsReturnsFalseOn404) {
    TcpServer server(404, "Not Found");
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    std::atomic<bool> done{false};
    std::optional<bool> result;
    client.checkObjectExistsAsync("missing", [&](std::optional<bool> exists) {
        result = exists;
        done = true;
    });

    server.capturedRequest();
    waitForCallback(done);

    ASSERT_TRUE(result.has_value()) << "Expected a definite result, got error";
    EXPECT_FALSE(*result) << "Expected exists=false for HTTP 404";
}

TEST_F(RestClientTest, CheckExistsReturnsErrorOn500) {
    TcpServer server(500, "Internal Server Error");
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    std::atomic<bool> done{false};
    std::optional<bool> result{false}; // sentinel; a 500 must reset this to nullopt
    client.checkObjectExistsAsync("boom", [&](std::optional<bool> exists) {
        result = exists;
        done = true;
    });

    server.capturedRequest();
    waitForCallback(done);

    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(result.has_value()) << "Expected error (nullopt) for HTTP 500";
}

} // namespace gtest::obj
