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

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

// c10d TCPStore wire constants (torch 2.x). Keys carry the same "/" prefix the
// reference client prepends so we share the keyspace with torch clients.
constexpr std::uint32_t kValidationMagic = 0x3C85F7CE;
constexpr char kKeyPrefix[] = "/";

// Upper bound on a single value read from the store; metadata blobs are far
// smaller, so anything larger means a corrupt or desynced response.
constexpr std::uint64_t kMaxBlobBytes = 1ULL << 30; // 1 GiB

// Subset of c10d::detail::QueryType we use; values are the enum ordinals.
enum class query_type_t : std::uint8_t {
    VALIDATE = 0,
    SET = 1,
    GET = 3,
    CHECK = 5,
    DELETE_KEY = 8,
    MULTI_GET = 10,
    PING = 13,
};

// c10d::detail::CheckResponseType.
enum class check_response_t : std::uint8_t { READY = 0, NOT_READY = 1 };

template<typename T>
void
appendValue(std::vector<std::uint8_t> &buf, T value) {
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

} // namespace

nixlTcpStoreClient::nixlTcpStoreClient(const std::string &host,
                                       std::uint16_t port,
                                       std::chrono::milliseconds timeout) {
    if (timeout <= std::chrono::milliseconds::zero()) {
        timeout = std::chrono::milliseconds(5000);
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *results = nullptr;
    const std::string port_str = std::to_string(port);
    if (const int err = ::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &results); err != 0) {
        throw std::runtime_error("TCPStore client: cannot resolve " + host + ":" + port_str + ": " +
                                 ::gai_strerror(err));
    }

    int fd = -1;
    for (addrinfo *ai = results; ai != nullptr; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }
        // Non-blocking connect bounded by the timeout, then restore blocking.
        const int flags = ::fcntl(fd, F_GETFL, 0);
        ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = ::connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc < 0 && errno == EINPROGRESS) {
            pollfd pfd{fd, POLLOUT, 0};
            rc = ::poll(&pfd, 1, static_cast<int>(timeout.count()));
            if (rc > 0) {
                int so_err = 0;
                socklen_t len = sizeof(so_err);
                if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &len) == 0 && so_err == 0) {
                    rc = 0;
                } else {
                    rc = -1;
                }
            } else {
                rc = -1;
            }
        }
        ::fcntl(fd, F_SETFL, flags);
        if (rc == 0) {
            break;
        }
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(results);

    if (fd < 0) {
        throw std::runtime_error("TCPStore client: failed to connect to " + host + ":" + port_str);
    }
    fd_ = fd;

    // Bound every blocking send/recv by the same timeout.
    timeval tv{};
    tv.tv_sec = timeout.count() / 1000;
    tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    try {
        // VALIDATE must be the first query; the server drops unvalidated peers.
        std::vector<std::uint8_t> buf{static_cast<std::uint8_t>(query_type_t::VALIDATE)};
        appendValue<std::uint32_t>(buf, kValidationMagic);
        sendAll(buf.data(), buf.size());

        // PING round-trips a nonce, confirming the server is responsive.
        const auto nonce = static_cast<std::uint32_t>(::getpid());
        buf = {static_cast<std::uint8_t>(query_type_t::PING)};
        appendValue<std::uint32_t>(buf, nonce);
        sendAll(buf.data(), buf.size());

        std::uint32_t echoed = 0;
        recvAll(&echoed, sizeof(echoed));
        if (echoed != nonce) {
            throw std::runtime_error("TCPStore client: ping nonce mismatch");
        }
    }
    catch (...) {
        ::close(fd_);
        fd_ = -1;
        throw;
    }
}

nixlTcpStoreClient::~nixlTcpStoreClient() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void
nixlTcpStoreClient::sendAll(const void *data, std::size_t len) {
    const auto *p = static_cast<const char *>(data);
    while (len > 0) {
        const ssize_t n = ::send(fd_, p, len, MSG_NOSIGNAL);
        if (n <= 0) {
            // A failed/partial send leaves the request half-written; close so
            // the desynced connection cannot be reused for later ops.
            const int saved = errno;
            ::close(fd_);
            fd_ = -1;
            errno = saved;
            throwErrno("send failed");
        }
        p += n;
        len -= static_cast<std::size_t>(n);
    }
}

void
nixlTcpStoreClient::recvAll(void *data, std::size_t len) {
    auto *p = static_cast<char *>(data);
    while (len > 0) {
        const ssize_t n = ::recv(fd_, p, len, 0);
        if (n <= 0) {
            // A timeout/short read mid-response desyncs the stream (the next op
            // would parse leftover bytes); close so it cannot be reused.
            const int saved = errno;
            ::close(fd_);
            fd_ = -1;
            errno = saved;
            throwErrno("recv failed");
        }
        p += n;
        len -= static_cast<std::size_t>(n);
    }
}

void
nixlTcpStoreClient::set(const std::string &key, const std::string &value) {
    const std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::uint8_t> buf{static_cast<std::uint8_t>(query_type_t::SET)};
    appendString(buf, kKeyPrefix + key);
    appendString(buf, value);
    sendAll(buf.data(), buf.size());
}

bool
nixlTcpStoreClient::check(const std::string &key) {
    const std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::uint8_t> buf{static_cast<std::uint8_t>(query_type_t::CHECK)};
    appendValue<std::uint64_t>(buf, 1);
    appendString(buf, kKeyPrefix + key);
    sendAll(buf.data(), buf.size());

    auto response = check_response_t::NOT_READY;
    recvAll(&response, sizeof(response));
    return response == check_response_t::READY;
}

std::string
nixlTcpStoreClient::get(const std::string &key) {
    const std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::uint8_t> buf{static_cast<std::uint8_t>(query_type_t::GET)};
    appendString(buf, kKeyPrefix + key);
    sendAll(buf.data(), buf.size());
    return recvBlob();
}

std::vector<std::string>
nixlTcpStoreClient::multiGet(const std::vector<std::string> &keys) {
    const std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::uint8_t> buf{static_cast<std::uint8_t>(query_type_t::MULTI_GET)};
    appendValue<std::uint64_t>(buf, keys.size());
    for (const auto &key : keys) {
        appendString(buf, kKeyPrefix + key);
    }
    sendAll(buf.data(), buf.size());

    std::vector<std::string> values;
    values.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        values.push_back(recvBlob());
    }
    return values;
}

std::string
nixlTcpStoreClient::recvBlob() {
    std::uint64_t len = 0;
    recvAll(&len, sizeof(len));
    if (len > kMaxBlobBytes) {
        // The stream is now desynced (the body won't be consumed); drop it.
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("TCPStore client: response length " + std::to_string(len) +
                                 " exceeds cap");
    }
    std::string value(len, '\0');
    recvAll(value.data(), len);
    return value;
}

bool
nixlTcpStoreClient::deleteKey(const std::string &key) {
    const std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::uint8_t> buf{static_cast<std::uint8_t>(query_type_t::DELETE_KEY)};
    appendString(buf, kKeyPrefix + key);
    sendAll(buf.data(), buf.size());

    std::int64_t num_deleted = 0;
    recvAll(&num_deleted, sizeof(num_deleted));
    return num_deleted == 1;
}
