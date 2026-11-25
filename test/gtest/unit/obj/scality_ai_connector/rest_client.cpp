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

// restClient HTTP wire format and response handling, without any RDMA hardware:
// each test starts a loopback TCP server, points a restClient at it, issues a
// request with a fake RDMA descriptor, and checks the raw HTTP it captured or the
// outcome the client reported.

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
#include "rest_accel/scality_ai_connector/client.h"

namespace gtest::obj {

// A throwaway single-request HTTP server: binds to localhost:0, reads one full
// HTTP request, replies with the configured status (and optionally a body), and
// hands the raw request text back.
//
// setBody() makes it serve bytes, which is what the plain-HTTP read path needs:
// the RDMA path only ever cares about the request, but a body read has to be
// checked against what actually lands in the caller's buffer.
class tcpServer {
public:
    explicit tcpServer(int status_code = 200, std::string reason = "OK")
        : statusCode_(status_code),
          reason_(std::move(reason)) {
        listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
        EXPECT_GE(listenFd_, 0) << "socket() failed: " << strerror(errno);

        int opt = 1;
        setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};

        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0; // let the OS assign a free port

        EXPECT_EQ(bind(listenFd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0)
            << "bind() failed: " << strerror(errno);
        EXPECT_EQ(listen(listenFd_, 1), 0) << "listen() failed: " << strerror(errno);

        socklen_t len = sizeof(addr);
        getsockname(listenFd_, reinterpret_cast<struct sockaddr *>(&addr), &len);
        port_ = ntohs(addr.sin_port);

        future_ = promise_.get_future();
        thread_ = std::thread(&tcpServer::acceptAndRead, this);
    }

    ~tcpServer() {
        if (listenFd_ >= 0) {
            close(listenFd_);
        }
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    int
    port() const {
        return port_;
    }

    /// Serve `body` after the status line. `declared_length` overrides the
    /// Content-Length header so a truncated response can be simulated: claiming
    /// more than is sent is what libcurl reports as CURLE_PARTIAL_FILE.
    /// Must be called before the client connects.
    void
    setBody(std::string body, long declared_length = -1) {
        body_ = std::move(body);
        declaredLength_ = (declared_length < 0) ? static_cast<long>(body_.size()) : declared_length;
        hasBody_ = true;
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
        int client_fd = accept(listenFd_, nullptr, nullptr);
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

        const long content_length = hasBody_ ? declaredLength_ : 0;
        std::string response = "HTTP/1.1 " + std::to_string(statusCode_) + " " + reason_ +
            "\r\nContent-Length: " + std::to_string(content_length) + "\r\n\r\n";
        if (hasBody_) {
            response += body_;
        }
        send(client_fd, response.data(), response.size(), 0);
        close(client_fd);

        promise_.set_value(request);
    }

    int statusCode_;
    std::string reason_;
    bool hasBody_ = false;
    std::string body_;
    long declaredLength_ = 0;
    int listenFd_ = -1;
    int port_ = 0;
    std::thread thread_;
    std::promise<std::string> promise_;
    std::future<std::string> future_;
};

// A listening socket that never accepts. The kernel still completes the TCP
// handshake from its backlog, so a client connects, sends its request, and waits
// for a response that never comes.
class silentListener {
public:
    silentListener() {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        EXPECT_GE(fd_, 0) << "socket() failed: " << strerror(errno);

        struct sockaddr_in addr{};

        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        EXPECT_EQ(bind(fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0);
        EXPECT_EQ(listen(fd_, 16), 0);

        socklen_t len = sizeof(addr);
        getsockname(fd_, reinterpret_cast<struct sockaddr *>(&addr), &len);
        port_ = ntohs(addr.sin_port);
    }

    ~silentListener() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    int
    port() const {
        return port_;
    }

private:
    int fd_ = -1;
    int port_ = 0;
};

static nixl_b_params_t
makeRestParams(const std::string &endpoint) {
    return {{"endpoint_override", endpoint}};
}

static std::string
localEndpoint(int port) {
    return "http://127.0.0.1:" + std::to_string(port);
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

class scalityRestClientTest : public testing::Test {};

// ---------------------------------------------------------------------------
// RDMA PUT/GET: header-only requests carrying the RDMA descriptor
// ---------------------------------------------------------------------------

TEST_F(scalityRestClientTest, PutSendsCorrectUrlAndHeaders) {
    tcpServer server;
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

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
    // The body must be empty: the data moves by RDMA, not in the HTTP body.
    EXPECT_NE(req.find("Content-Length: 0"), std::string::npos) << "Content-Length: 0 missing in:\n"
                                                                << req;
    EXPECT_TRUE(success.load()) << "putObjectRdmaAsync reported failure";
}

TEST_F(scalityRestClientTest, GetSendsCorrectUrlAndHeaders) {
    tcpServer server;
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

    const size_t offset = 128;
    std::vector<char> buf(512);

    std::atomic<bool> done{false}, success{false};
    client.getObjectRdmaAsync("readkey",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              offset,
                              "rdma-get-descriptor",
                              /*past_end_ok=*/false,
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
    EXPECT_NE(req.find("x-scal-rdma: rdma-get-descriptor"), std::string::npos)
        << "x-scal-rdma missing in:\n"
        << req;
    EXPECT_NE(req.find("Range: bytes=128-639"), std::string::npos) << "Range missing in:\n" << req;
    EXPECT_TRUE(success.load()) << "getObjectRdmaAsync reported failure";
}

// A sized read at offset 0 must still be ranged, else the server sends the whole
// object into a buffer registered for only data_len bytes.
TEST_F(scalityRestClientTest, GetAtOffsetZeroSendsRangeHeader) {
    tcpServer server;
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

    std::vector<char> buf(8);

    std::atomic<bool> done{false}, success{false};
    client.getObjectRdmaAsync("headerprobe",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              /*offset=*/0,
                              "rdma-get-descriptor",
                              /*past_end_ok=*/false,
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    std::string req = server.capturedRequest();
    waitForCallback(done);

    EXPECT_NE(req.find("GET /headerprobe"), std::string::npos)
        << "Expected 'GET /headerprobe' in:\n"
        << req;
    EXPECT_NE(req.find("Range: bytes=0-7"), std::string::npos)
        << "offset-0 read must be ranged in:\n"
        << req;
    EXPECT_TRUE(success.load()) << "getObjectRdmaAsync reported failure";
}

// A 416 on a piece of a split read that starts past the end of the object means
// there is nothing left to read: success. On any other request it is an error.
TEST_F(scalityRestClientTest, GetTreats416AsSuccessOnlyWhenPastEndIsOk) {
    for (const bool past_end_ok : {true, false}) {
        tcpServer server(416, "Range Not Satisfiable");
        nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
        restClient client(&params);

        std::vector<char> buf(64);
        std::atomic<bool> done{false}, success{!past_end_ok};
        client.getObjectRdmaAsync("small",
                                  reinterpret_cast<uintptr_t>(buf.data()),
                                  buf.size(),
                                  /*offset=*/8u << 20,
                                  "rdma-get-descriptor",
                                  past_end_ok,
                                  [&](bool ok) {
                                      success = ok;
                                      done = true;
                                  });

        server.capturedRequest();
        waitForCallback(done);

        ASSERT_TRUE(done.load()) << "Callback was never invoked";
        EXPECT_EQ(success.load(), past_end_ok)
            << "416 with past_end_ok=" << past_end_ok << " reported " << success.load();
    }
}

TEST_F(scalityRestClientTest, PutRejectsEmptyRdmaDesc) {
    // Port 1 won't accept connections, so a stray connection attempt fails loudly.
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    restClient client(&params);

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

TEST_F(scalityRestClientTest, PutRejectsZeroDataLen) {
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    restClient client(&params);

    std::atomic<bool> done{false}, success{true};
    client.putObjectRdmaAsync("k",
                              /*data_ptr=*/0,
                              /*data_len=*/0,
                              0,
                              "some-descriptor",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with data_len=0";
}

TEST_F(scalityRestClientTest, GetRejectsEmptyRdmaDesc) {
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    restClient client(&params);

    std::vector<char> buf(64);
    std::atomic<bool> done{false}, success{true};
    client.getObjectRdmaAsync("k",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              0,
                              /*rdma_desc=*/"",
                              /*past_end_ok=*/false,
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with empty rdma_desc";
}

// ---------------------------------------------------------------------------
// Construction and timeouts
// ---------------------------------------------------------------------------

TEST_F(scalityRestClientTest, ConstructorThrowsOnMissingEndpoint) {
    nixl_b_params_t params = {}; // intentionally empty
    EXPECT_THROW({ restClient c(&params); }, std::invalid_argument);
}

TEST_F(scalityRestClientTest, ConstructorThrowsOnNullParams) {
    EXPECT_THROW({ restClient c(nullptr); }, std::invalid_argument);
}

TEST_F(scalityRestClientTest, ConstructorRejectsInvalidRequestTimeout) {
    for (const char *value : {"0", "fast", "100ms"}) {
        nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
        params["request_timeout_ms"] = value;
        EXPECT_THROW(
            { restClient c(&params); }, std::invalid_argument)
            << "request_timeout_ms=" << value << " was accepted";
    }
}

// std::stoul alone would accept "-1" (wrapping to a huge value) and " 5".
TEST_F(scalityRestClientTest, ConstructorRejectsNonDigitNumbers) {
    for (const char *key : {"num_threads", "request_timeout_ms", "max_inflight"}) {
        for (const char *value : {"-1", "+5", " 5", "", "99999999999999999999999"}) {
            nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
            params[key] = value;
            EXPECT_THROW(
                { restClient c(&params); }, std::invalid_argument)
                << key << "='" << value << "' was accepted";
        }
    }
}

// A request the endpoint never answers fails after request_timeout_ms instead of
// holding the transfer.
TEST_F(scalityRestClientTest, StalledRequestFailsAfterRequestTimeout) {
    silentListener listener;
    nixl_b_params_t params = makeRestParams(localEndpoint(listener.port()));
    params["request_timeout_ms"] = "300";
    restClient client(&params);

    std::vector<char> buf(64);
    std::atomic<bool> done{false}, success{true};
    const auto start = std::chrono::steady_clock::now();
    client.getObjectRdmaAsync("stalled",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              0,
                              "rdma-get-descriptor",
                              /*past_end_ok=*/false,
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    waitForCallback(done, 5000);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    ASSERT_TRUE(done.load()) << "a stalled request never timed out";
    EXPECT_FALSE(success.load()) << "a stalled request must fail";
    EXPECT_GE(elapsed.count(), 250) << "failed before request_timeout_ms";
    EXPECT_LT(elapsed.count(), 2000) << "ignored request_timeout_ms";
}

// ---------------------------------------------------------------------------
// Existence check (HTTP HEAD)
// ---------------------------------------------------------------------------

TEST_F(scalityRestClientTest, CheckExistsReturnsTrueOn200) {
    tcpServer server(200, "OK");
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

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

TEST_F(scalityRestClientTest, CheckExistsReturnsFalseOn404) {
    tcpServer server(404, "Not Found");
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

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

TEST_F(scalityRestClientTest, CheckExistsReturnsErrorOn500) {
    tcpServer server(500, "Internal Server Error");
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

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

// ---------------------------------------------------------------------------
// getObjectBodyAsync: plain HTTP into the caller's buffer, no RDMA
// ---------------------------------------------------------------------------

TEST_F(scalityRestClientTest, BodyGetSendsRangeAndNoRdmaHeader) {
    tcpServer server(206, "Partial Content");
    server.setBody("01234567");
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

    char buf[8] = {};
    std::atomic<bool> done{false};
    bool ok = false;
    client.getObjectBodyAsync("shard.safetensors", buf, sizeof(buf), 0, [&](bool success) {
        ok = success;
        done = true;
    });

    const std::string request = server.capturedRequest();
    waitForCallback(done);

    ASSERT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_TRUE(ok);
    EXPECT_NE(request.find("GET /shard.safetensors "), std::string::npos) << request;
    // Offset 0 included: an unranged GET would pull the whole multi-GB shard.
    EXPECT_NE(request.find("Range: bytes=0-7"), std::string::npos) << request;
    // The absence of this header is what makes the endpoint answer with a body.
    EXPECT_EQ(request.find("x-scal-rdma"), std::string::npos)
        << "body read must not carry an RDMA descriptor: " << request;
    EXPECT_EQ(std::string(buf, sizeof(buf)), "01234567");
}

TEST_F(scalityRestClientTest, BodyGetHonoursOffset) {
    tcpServer server(206, "Partial Content");
    server.setBody("abcd");
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

    char buf[4] = {};
    std::atomic<bool> done{false};
    bool ok = false;
    client.getObjectBodyAsync("k", buf, sizeof(buf), 4096, [&](bool success) {
        ok = success;
        done = true;
    });

    const std::string request = server.capturedRequest();
    waitForCallback(done);

    EXPECT_NE(request.find("Range: bytes=4096-4099"), std::string::npos) << request;
    EXPECT_TRUE(ok);
    EXPECT_EQ(std::string(buf, sizeof(buf)), "abcd");
}

// Past offset 0, a 200 is the object from its first byte: the Range header was
// ignored, and the bytes in the buffer are the wrong ones even though they fit.
TEST_F(scalityRestClientTest, BodyGetAtOffsetRejects200) {
    tcpServer server(200, "OK");
    server.setBody("abcd");
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

    char buf[4] = {};
    std::atomic<bool> done{false};
    bool ok = true;
    client.getObjectBodyAsync("k", buf, sizeof(buf), 4096, [&](bool success) {
        ok = success;
        done = true;
    });

    server.capturedRequest();
    waitForCallback(done);

    ASSERT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(ok) << "a 200 for a range past offset 0 must fail";
}

TEST_F(scalityRestClientTest, BodyGetRejectsResponseLongerThanRequested) {
    // A server that ignores Range answers an 8-byte request with the whole object.
    // Without the write bound that is a heap overflow, so this must fail instead.
    tcpServer server(200, "OK");
    server.setBody(std::string(64 * 1024, 'x'));
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

    struct guardedBuffer {
        char buf[8];
        char canary[8];
    } guarded;

    std::memset(&guarded, 0, sizeof(guarded));

    std::atomic<bool> done{false};
    bool ok = true;
    client.getObjectBodyAsync("k", guarded.buf, sizeof(guarded.buf), 0, [&](bool success) {
        ok = success;
        done = true;
    });

    server.capturedRequest();
    waitForCallback(done);

    ASSERT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(ok) << "an over-long response must fail the read";
    EXPECT_EQ(std::string(guarded.canary, sizeof(guarded.canary)), std::string(8, '\0'))
        << "wrote past the end of the caller's buffer";
}

TEST_F(scalityRestClientTest, BodyGetAcceptsCompleteButShortResponse) {
    // A range reaching past the end of the object is answered as a complete 206
    // with fewer bytes. Those bytes are what the caller wanted.
    tcpServer server(206, "Partial Content");
    server.setBody("short");
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

    char buf[64] = {};
    std::atomic<bool> done{false};
    bool ok = false;
    client.getObjectBodyAsync("k", buf, sizeof(buf), 0, [&](bool success) {
        ok = success;
        done = true;
    });

    server.capturedRequest();
    waitForCallback(done);

    ASSERT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_TRUE(ok) << "a complete but shorter response is not an error";
    EXPECT_EQ(std::string(buf, 5), "short");
    EXPECT_EQ(buf[5], '\0') << "the tail past the object's end must stay untouched";
}

TEST_F(scalityRestClientTest, BodyGetFailsWhenTruncatedAgainstContentLength) {
    // Distinct from the case above: the response promises 64 bytes and delivers 5,
    // which is a transfer cut short. libcurl reports CURLE_PARTIAL_FILE.
    tcpServer server(206, "Partial Content");
    server.setBody("short", /*declared_length=*/64);
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

    char buf[64] = {};
    std::atomic<bool> done{false};
    bool ok = true;
    client.getObjectBodyAsync("k", buf, sizeof(buf), 0, [&](bool success) {
        ok = success;
        done = true;
    });

    server.capturedRequest();
    waitForCallback(done, 10000);

    ASSERT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(ok) << "a body cut short against its own Content-Length must fail";
}

TEST_F(scalityRestClientTest, BodyGetRejectsNullDestinationAndZeroLength) {
    // No tcpServer: both rejections happen before a request is built, and a server
    // that never gets one would block in accept() until its destructor joined it.
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    restClient client(&params);

    char buf[8] = {};
    bool null_ok = true;
    client.getObjectBodyAsync("k", nullptr, sizeof(buf), 0, [&](bool s) { null_ok = s; });
    EXPECT_FALSE(null_ok) << "a null destination must fail without a request";

    bool zero_ok = true;
    client.getObjectBodyAsync("k", buf, 0, 0, [&](bool s) { zero_ok = s; });
    EXPECT_FALSE(zero_ok) << "a zero-length read must fail without a request";
}

TEST_F(scalityRestClientTest, BodyGetKeepsErrorResponseOutOfTheCallerBuffer) {
    // A 404 carries an explanation, not data. Writing it into the destination would
    // corrupt the buffer of a read that failed; the text belongs in the failure log.
    tcpServer server(404, "Not Found");
    server.setBody("no such object");
    nixl_b_params_t params = makeRestParams(localEndpoint(server.port()));
    restClient client(&params);

    char buf[64];
    std::memset(buf, 0xAB, sizeof(buf)); // poison: must survive untouched
    std::atomic<bool> done{false};
    bool ok = true;
    client.getObjectBodyAsync("missing", buf, sizeof(buf), 0, [&](bool success) {
        ok = success;
        done = true;
    });

    server.capturedRequest();
    waitForCallback(done);

    ASSERT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(ok) << "a 404 must fail the read";
    for (size_t i = 0; i < sizeof(buf); ++i) {
        EXPECT_EQ(static_cast<unsigned char>(buf[i]), 0xABu)
            << "error body was written into the caller's buffer at byte " << i;
    }
}

} // namespace gtest::obj
