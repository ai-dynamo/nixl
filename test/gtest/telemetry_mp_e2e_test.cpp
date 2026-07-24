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
#include "prometheus_mp_exporter.h"

#include "common.h"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;

[[nodiscard]] nixlTelemetryExporterInitParams
initParams(const std::string &agent) {
    return nixlTelemetryExporterInitParams{agent, 4096};
}

// Minimal HTTP/1.1 GET over 127.0.0.1:<port>; returns the response body (empty on
// failure). Self-contained to keep the test free of an HTTP client dependency.
[[nodiscard]] std::string
httpGet(uint16_t port, const std::string &path) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return {};
    }
    const struct timeval tv{3, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = ::inet_addr("127.0.0.1");
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return {};
    }

    const std::string req =
        "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    ::send(fd, req.data(), req.size(), 0);

    std::string response;
    char buf[4096];
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        response.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fd);

    const auto pos = response.find("\r\n\r\n");
    return pos == std::string::npos ? std::string{} : response.substr(pos + 4);
}

[[nodiscard]] std::string
scrapeMetrics(uint16_t port) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    do {
        const std::string body = httpGet(port, "/metrics");
        if (!body.empty()) {
            return body;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    } while (std::chrono::steady_clock::now() < deadline);
    return {};
}

// Parses "<metric>{...agent_name="X"...} <value>" lines into agent_name -> value.
[[nodiscard]] std::map<std::string, double>
parseSeriesByAgent(const std::string &body, const std::string &metric) {
    std::map<std::string, double> out;
    const std::string prefix = metric + "{";
    const std::string key = "agent_name=\"";
    std::istringstream iss(body);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.rfind(prefix, 0) != 0) {
            continue;
        }
        const auto ap = line.find(key);
        const auto brace = line.rfind('}');
        if (ap == std::string::npos || brace == std::string::npos) {
            continue;
        }
        const auto vstart = ap + key.size();
        const auto vend = line.find('"', vstart);
        if (vend == std::string::npos) {
            continue;
        }
        try {
            out[line.substr(vstart, vend - vstart)] = std::stod(line.substr(brace + 1));
        }
        catch (const std::exception &) {
        }
    }
    return out;
}

void
runWriterChild(int go_fd, int ready_fd, int quit_fd, const std::string &agent, uint64_t tx_value) {
    char c = 0;
    while (::read(go_fd, &c, 1) > 0) {}

    int rc = 0;
    try {
        nixlTelemetryPrometheusMpExporter exporter(initParams(agent));
        exporter.exportEvent({TX_BYTES, tx_value});
        const char ok = 1;
        ::write(ready_fd, &ok, 1);
        // Block until the parent closes the quit pipe.
        while (::read(quit_fd, &c, 1) > 0) {}
    }
    catch (...) {
        rc = 3;
    }
    ::_exit(rc);
}

class MpE2ETest : public ::testing::Test {
protected:
    void
    SetUp() override {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = std::filesystem::path(::testing::TempDir()) /
            ("nixl_mp_e2e_" + std::to_string(::getpid()) + "_" + info->name());
        std::filesystem::create_directories(dir_);
        port_ = gtest::PortAllocator::next_tcp_port();
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_LOCAL", "y");
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_PORT", std::to_string(port_));
        env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", dir_.string());
        // Dead processes become stale immediately so the reaping check is prompt.
        env_.addVar("NIXL_TELEMETRY_MP_STALE_TTL", "0");
    }

    void
    TearDown() override {
        for (int i = 0; i < 4; ++i) {
            env_.popVar();
        }
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    gtest::ScopedEnv env_;
    uint16_t port_ = 0;
    std::filesystem::path dir_;
};

TEST_F(MpE2ETest, AllRankProcessesAggregateBehindOneEndpointAndStaleAreDropped) {
    constexpr int kChildren = 3;

    int go_pipe[2];
    int ready_pipe[2];
    int quit_pipe[2];
    ASSERT_EQ(::pipe(go_pipe), 0);
    ASSERT_EQ(::pipe(ready_pipe), 0);
    ASSERT_EQ(::pipe(quit_pipe), 0);

    // Fork children while the parent is still single-threaded (before it builds
    // the owner exporter, which starts civetweb threads).
    std::vector<pid_t> children;
    for (int i = 0; i < kChildren; ++i) {
        const pid_t pid = ::fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            ::close(go_pipe[1]);
            ::close(ready_pipe[0]);
            ::close(quit_pipe[1]);
            runWriterChild(go_pipe[0],
                           ready_pipe[1],
                           quit_pipe[0],
                           "agent-" + std::to_string(i),
                           static_cast<uint64_t>((i + 1) * 100));
        }
        children.push_back(pid);
    }

    ::close(go_pipe[0]);
    ::close(ready_pipe[1]);
    ::close(quit_pipe[0]);

    // Parent becomes the bind-race owner and serves the endpoint.
    nixlTelemetryPrometheusMpExporter owner(initParams("agent-parent"));
    ASSERT_TRUE(owner.isExporter());
    owner.exportEvent({TX_BYTES, 999});

    // Release the children (they now become writers) and wait for readiness.
    ::close(go_pipe[1]);
    for (int i = 0; i < kChildren; ++i) {
        char c = 0;
        ASSERT_EQ(::read(ready_pipe[0], &c, 1), 1);
    }

    // Phase 1: every process must appear behind the single owner endpoint.
    const auto phase1 = parseSeriesByAgent(scrapeMetrics(port_), "agent_tx_bytes_total");
    EXPECT_EQ(phase1.size(), static_cast<std::size_t>(kChildren + 1));
    EXPECT_DOUBLE_EQ(phase1.at("agent-parent"), 999.0);
    EXPECT_DOUBLE_EQ(phase1.at("agent-0"), 100.0);
    EXPECT_DOUBLE_EQ(phase1.at("agent-1"), 200.0);
    EXPECT_DOUBLE_EQ(phase1.at("agent-2"), 300.0);

    // Kill one child and reap it so its pid is truly gone before the next scrape.
    ASSERT_EQ(::kill(children[0], SIGKILL), 0);
    int status = 0;
    ASSERT_EQ(::waitpid(children[0], &status, 0), children[0]);

    // Phase 2: the dead child's series is dropped (and its store reaped).
    const auto phase2 = parseSeriesByAgent(scrapeMetrics(port_), "agent_tx_bytes_total");
    EXPECT_EQ(phase2.count("agent-0"), 0u);
    EXPECT_EQ(phase2.count("agent-1"), 1u);
    EXPECT_EQ(phase2.count("agent-2"), 1u);
    EXPECT_EQ(phase2.count("agent-parent"), 1u);

    // Release the remaining children and reap them.
    ::close(quit_pipe[1]);
    for (int i = 1; i < kChildren; ++i) {
        ::waitpid(children[i], &status, 0);
    }
    ::close(ready_pipe[0]);
}

} // namespace
