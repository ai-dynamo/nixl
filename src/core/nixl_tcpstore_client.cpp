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
#include "nixl_tcpstore_client.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

// c10d TCPStore wire constants (torch 2.x). Keys carry the same "/" prefix the
// reference client prepends so we share the keyspace with torch clients.
// Integers go on the wire in native byte order (matching c10d/torch), so this
// assumes the store server runs on the same endianness as the client.
constexpr std::uint32_t validation_magic = 0x3C85F7CE;
constexpr char key_prefix[] = "/";

// Upper bound on a single value read from the store; metadata blobs are far
// smaller, so anything larger means a corrupt or desynced response.
constexpr std::uint64_t max_blob_bytes = 1ULL << 30; // 1 GiB

// Subset of c10d::detail::QueryType we use; values are the enum ordinals.
enum class query_type_t : std::uint8_t {
    VALIDATE = 0,
    SET = 1,
    GET = 3,
    CHECK = 5,
    DELETE_KEY = 8,
    PING = 13,
};

// c10d::detail::CheckResponseType.
enum class check_response_t : std::uint8_t { READY = 0, NOT_READY = 1 };

template<typename T>
void
appendValue(std::vector<std::uint8_t> &buf, T value) {
    static_assert(std::is_integral_v<T>, "appendValue writes an object representation");
    const auto *begin = reinterpret_cast<const std::uint8_t *>(&value);
    buf.insert(buf.end(), begin, begin + sizeof(T));
}

void
appendString(std::vector<std::uint8_t> &buf, const std::string &str) {
    appendValue<std::uint64_t>(buf, str.size());
    buf.insert(buf.end(), str.begin(), str.end());
}

[[noreturn]] void
throwErrno(const std::string &what) {
    throw std::runtime_error("TCPStore client: " + what + ": " + std::strerror(errno));
}

struct addrInfoDeleter {
    void
    operator()(addrinfo *ai) const noexcept {
        ::freeaddrinfo(ai);
    }
};

using addr_info_ptr_t = std::unique_ptr<addrinfo, addrInfoDeleter>;

[[nodiscard]] addr_info_ptr_t
resolveHost(const std::string &host, const std::string &port_str) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *results = nullptr;
    if (const int err = ::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &results); err != 0) {
        throw std::runtime_error("TCPStore client: cannot resolve " + host + ":" + port_str + ": " +
                                 ::gai_strerror(err));
    }
    return addr_info_ptr_t{results};
}

// Bounded non-blocking connect to one address. Returns an empty owner when the
// address cannot be reached within timeout, so the caller can try the next one.
[[nodiscard]] nixl::scopedFd
connectOne(const addrinfo &ai, std::chrono::milliseconds timeout) {
    nixl::scopedFd fd{::socket(ai.ai_family, ai.ai_socktype, ai.ai_protocol)};
    if (!fd.valid()) {
        return {};
    }

    // A failed mode switch makes the bounded connect unreliable; drop the address.
    const int flags = ::fcntl(fd.get(), F_GETFL, 0);
    if (flags < 0 || ::fcntl(fd.get(), F_SETFL, flags | O_NONBLOCK) < 0) {
        return {};
    }

    int rc = ::connect(fd.get(), ai.ai_addr, ai.ai_addrlen);
    if (rc < 0 && errno == EINPROGRESS) {
        pollfd pfd{fd.get(), POLLOUT, 0};
        rc = -1;
        if (::poll(&pfd, 1, static_cast<int>(timeout.count())) > 0) {
            int so_err = 0;
            socklen_t len = sizeof(so_err);
            if (::getsockopt(fd.get(), SOL_SOCKET, SO_ERROR, &so_err, &len) == 0 && so_err == 0) {
                rc = 0;
            }
        }
    }

    if (rc != 0 || ::fcntl(fd.get(), F_SETFL, flags) != 0) {
        return {};
    }
    return fd;
}

// Bounds each blocking syscall; the sendAll/recvAll deadlines sit on top.
void
armTimeouts(int fd, std::chrono::milliseconds timeout) {
    timeval tv{};
    tv.tv_sec = timeout.count() / 1000;
    tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0 ||
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) {
        throwErrno("failed to set socket I/O timeout");
    }
}

} // namespace

nixlTcpStoreClient::nixlTcpStoreClient(std::string host,
                                       std::uint16_t port,
                                       std::chrono::milliseconds connect_timeout,
                                       std::chrono::milliseconds op_timeout) noexcept
    : host_(std::move(host)),
      port_(port),
      connectTimeout_(connect_timeout),
      opTimeout_(op_timeout) {}

void
nixlTcpStoreClient::connect(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    const std::string port_str = std::to_string(port_);
    const addr_info_ptr_t results = resolveHost(host_, port_str);

    nixl::scopedFd fd;
    for (const addrinfo *ai = results.get(); ai != nullptr && !fd.valid(); ai = ai->ai_next) {
        // Whatever is left of the budget, so a multi-address host (dual stack,
        // one address blackholed) cannot cost timeout per address.
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (left <= std::chrono::milliseconds::zero()) {
            break;
        }
        fd = connectOne(*ai, left);
    }

    if (!fd.valid()) {
        throw std::runtime_error("TCPStore client: failed to connect to " + host_ + ":" + port_str);
    }

    // Arm before publishing to fd_: a failure here must not leave an unbounded
    // socket behind for the next operation to pick up as "already connected".
    // The steady-state op timeout, since this socket outlives bring-up; the
    // handshake stays inside the bring-up budget through its deadline.
    armTimeouts(fd.get(), opTimeout_);
    fd_ = std::move(fd);
    handshake(deadline);
}

void
nixlTcpStoreClient::ensureConnected() {
    if (!fd_.valid()) {
        connect(std::exchange(bringUp_, false) ? connectTimeout_ : opTimeout_);
    }
}

void
nixlTcpStoreClient::handshake(deadline_t deadline) {
    // VALIDATE must be the first query; the server drops unvalidated peers.
    std::vector<std::uint8_t> buf = {static_cast<std::uint8_t>(query_type_t::VALIDATE)};
    appendValue<std::uint32_t>(buf, validation_magic);
    sendAll(buf.data(), buf.size(), deadline);

    // PING round-trips a nonce, confirming the server is responsive.
    const auto nonce = static_cast<std::uint32_t>(::getpid());
    buf = {static_cast<std::uint8_t>(query_type_t::PING)};
    appendValue<std::uint32_t>(buf, nonce);
    sendAll(buf.data(), buf.size(), deadline);

    std::uint32_t echoed = 0;
    recvAll(&echoed, sizeof(echoed), deadline);
    if (echoed != nonce) {
        fd_.reset();
        throw std::runtime_error("TCPStore client: ping nonce mismatch");
    }
}

void
nixlTcpStoreClient::sendAll(const void *data, std::size_t len, deadline_t deadline) {
    const auto *p = static_cast<const char *>(data);
    while (len > 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fd_.reset();
            throw std::runtime_error("TCPStore client: send deadline exceeded");
        }
        const ssize_t n = ::send(fd_.get(), p, len, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) {
            continue; // interrupted by a signal before any bytes moved; retry
        }
        if (n <= 0) {
            // Half-written request; close so the desynced socket is not reused.
            const int saved = errno;
            fd_.reset();
            errno = saved;
            throwErrno("send failed");
        }
        p += n;
        len -= static_cast<std::size_t>(n);
    }
}

void
nixlTcpStoreClient::recvAll(void *data, std::size_t len, deadline_t deadline) {
    auto *p = static_cast<char *>(data);
    while (len > 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fd_.reset();
            throw std::runtime_error("TCPStore client: recv deadline exceeded");
        }
        const ssize_t n = ::recv(fd_.get(), p, len, 0);
        if (n < 0 && errno == EINTR) {
            continue; // interrupted by a signal before any bytes moved; retry
        }
        if (n <= 0) {
            // Partial response; the next op would parse leftover bytes.
            const int saved = errno;
            fd_.reset();
            errno = saved;
            throwErrno("recv failed");
        }
        p += n;
        len -= static_cast<std::size_t>(n);
    }
}

std::string
nixlTcpStoreClient::recvBlob(deadline_t deadline) {
    std::uint64_t len = 0;
    recvAll(&len, sizeof(len), deadline);
    if (len > max_blob_bytes) {
        // The stream is now desynced (the body won't be consumed); drop it.
        fd_.reset();
        throw std::runtime_error("TCPStore client: response length " + std::to_string(len) +
                                 " exceeds cap");
    }
    std::string value(len, '\0');
    recvAll(value.data(), len, deadline);
    return value;
}

void
nixlTcpStoreClient::set(const std::string &key, const std::string &value) {
    std::vector<std::uint8_t> buf = {static_cast<std::uint8_t>(query_type_t::SET)};
    appendString(buf, key_prefix + key);
    appendString(buf, value);

    ensureConnected();
    sendAll(buf.data(), buf.size(), opDeadline());
}

std::optional<std::string>
nixlTcpStoreClient::get(const std::string &key) {
    const std::string full_key = key_prefix + key;

    std::vector<std::uint8_t> check_buf = {static_cast<std::uint8_t>(query_type_t::CHECK)};
    appendValue<std::uint64_t>(check_buf, 1);
    appendString(check_buf, full_key);

    std::vector<std::uint8_t> get_buf = {static_cast<std::uint8_t>(query_type_t::GET)};
    appendString(get_buf, full_key);

    ensureConnected();
    // One deadline for the CHECK and the GET together: they are one operation
    // from the caller's point of view.
    const deadline_t deadline = opDeadline();

    sendAll(check_buf.data(), check_buf.size(), deadline);
    auto response = check_response_t::NOT_READY;
    recvAll(&response, sizeof(response), deadline);
    if (response != check_response_t::READY) {
        return std::nullopt;
    }

    sendAll(get_buf.data(), get_buf.size(), deadline);
    std::string value = recvBlob(deadline);
    if (value.empty()) {
        // Deleted between the CHECK and the GET (raced invalidate); the store
        // has no empty values of its own.
        return std::nullopt;
    }
    return value;
}

bool
nixlTcpStoreClient::deleteKey(const std::string &key) {
    std::vector<std::uint8_t> buf = {static_cast<std::uint8_t>(query_type_t::DELETE_KEY)};
    appendString(buf, key_prefix + key);

    ensureConnected();
    const deadline_t deadline = opDeadline();
    sendAll(buf.data(), buf.size(), deadline);

    std::int64_t num_deleted = 0;
    recvAll(&num_deleted, sizeof(num_deleted), deadline);
    return num_deleted == 1;
}
