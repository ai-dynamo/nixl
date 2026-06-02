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
#ifndef NIXL_TEST_DOCA_TELEMETRY_SCRAPE_UTIL_H
#define NIXL_TEST_DOCA_TELEMETRY_SCRAPE_UTIL_H

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>

namespace nixl::doca_test {

// Minimal HTTP/1.1 GET over 127.0.0.1:<port>; returns the response body (empty
// on failure).
inline std::string
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
        response.append(buf, n);
    }
    ::close(fd);

    const auto pos = response.find("\r\n\r\n");
    return pos == std::string::npos ? std::string{} : response.substr(pos + 4);
}

// Poll /metrics until it contains `needle`, or timeout.
inline std::string
scrapeUntil(uint16_t port, const std::string &needle, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::string body;
    do {
        body = httpGet(port, "/metrics");
        if (body.find(needle) != std::string::npos) {
            return body;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    } while (std::chrono::steady_clock::now() < deadline);
    return body;
}

// Value on the first non-comment exposition line that starts with `metric`.
// Exposition format is: name{labels} VALUE [TIMESTAMP]  (or  name VALUE [TS]).
// The value is the token right after the label set, NOT the trailing timestamp.
inline double
metricValue(const std::string &body, const std::string &metric) {
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.rfind(metric, 0) != 0) {
            continue;
        }

        size_t value_start;
        const auto labels_end = line.find("} ");
        if (labels_end != std::string::npos) {
            value_start = labels_end + 2;
        } else {
            const auto sp = line.find(' ');
            if (sp == std::string::npos) {
                continue;
            }
            value_start = sp + 1;
        }

        const auto value_end = line.find(' ', value_start);
        const std::string token = line.substr(
            value_start,
            value_end == std::string::npos ? std::string::npos : value_end - value_start);
        try {
            return std::stod(token);
        }
        catch (const std::exception &) {
        }
    }
    return -1.0;
}

// Ask the OS for a free TCP port on the loopback interface.
inline uint16_t
findFreePort() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return 0;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ::inet_addr("127.0.0.1");
    addr.sin_port = 0;
    uint16_t port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
        socklen_t len = sizeof(addr);
        if (::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0) {
            port = ntohs(addr.sin_port);
        }
    }
    ::close(fd);
    return port;
}

} // namespace nixl::doca_test

#endif // NIXL_TEST_DOCA_TELEMETRY_SCRAPE_UTIL_H
