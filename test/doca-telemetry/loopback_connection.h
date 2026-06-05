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
#ifndef NIXL_TEST_DOCA_TELEMETRY_LOOPBACK_CONNECTION_H
#define NIXL_TEST_DOCA_TELEMETRY_LOOPBACK_CONNECTION_H

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

namespace nixl::doca_test {

// RAII blocking TCP client to 127.0.0.1:<port>: connects (with send/recv
// timeouts) on construction and closes the socket on destruction. Single-use
// and scope-bound, so it is neither copyable nor movable.
class loopbackConnection {
public:
    explicit loopbackConnection(uint16_t port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) {
            return;
        }

        const struct timeval tv{3, 0};
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = ::inet_addr(loopbackAddr);
        if (::connect(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    ~loopbackConnection() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    loopbackConnection(const loopbackConnection &) = delete;
    loopbackConnection &
    operator=(const loopbackConnection &) = delete;

    bool
    connected() const {
        return fd_ >= 0;
    }

    // Write the whole buffer, retrying short writes and transient errors
    // (EINTR/EAGAIN/EWOULDBLOCK) after a short sleep. MSG_NOSIGNAL avoids a
    // SIGPIPE if the peer closed early. A write deadline bounds the retries.
    void
    send(const std::string &data) const {
        const auto write_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        size_t off = 0;
        while (off < data.size() && std::chrono::steady_clock::now() < write_deadline) {
            const ssize_t n = ::send(fd_, data.data() + off, data.size() - off, MSG_NOSIGNAL);
            if (n > 0) {
                off += static_cast<size_t>(n);
                continue;
            }
            if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            break; // hard error
        }
    }

    // Read until the peer closes the connection (Connection: close). Under
    // SO_RCVTIMEO a slow/not-yet-ready endpoint makes recv() return -1 with
    // EAGAIN/EWOULDBLOCK (or EINTR on a signal); treat those as transient and
    // retry after a short sleep instead of bailing out with a truncated body.
    // A read deadline bounds the retries so the loop can never hang.
    std::string
    recvUntilClosed() const {
        std::string response;
        char buf[4096];
        const auto read_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < read_deadline) {
            const ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
            if (n > 0) {
                response.append(buf, n);
                continue;
            }
            if (n == 0) {
                break; // peer closed: response complete
            }
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            break; // hard error
        }
        return response;
    }

    // Ask the OS for a free TCP port on the loopback interface.
    static uint16_t
    findFreePort() {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return 0;
        }
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = ::inet_addr(loopbackAddr);
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

    // Minimal HTTP/1.1 GET over 127.0.0.1:<port>; returns the response body
    // (empty on failure). One-shot: opens, sends, reads, and closes.
    static std::string
    httpGet(uint16_t port, const std::string &path) {
        const loopbackConnection conn(port);
        if (!conn.connected()) {
            return {};
        }

        conn.send("GET " + path + " HTTP/1.1\r\nHost: " + loopbackAddr +
                  "\r\nConnection: close\r\n\r\n");

        const std::string response = conn.recvUntilClosed();
        const auto pos = response.find("\r\n\r\n");
        return pos == std::string::npos ? std::string{} : response.substr(pos + 4);
    }

private:
    static constexpr const char *loopbackAddr = "127.0.0.1";

    int fd_ = -1;
};

} // namespace nixl::doca_test

#endif // NIXL_TEST_DOCA_TELEMETRY_LOOPBACK_CONNECTION_H
