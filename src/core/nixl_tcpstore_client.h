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
/**
 * @file nixl_tcpstore_client.h
 * @brief Minimal in-house client for the PyTorch c10d TCPStore wire protocol.
 *
 * Core-internal: speaks to the same server torch.distributed.TCPStore connects
 * to (no libtorch dependency). Only the subset nixlTcpStoreMetadataBackend
 * needs is implemented. Values are opaque byte blobs; the framing matches c10d
 * (uint64 length prefixes in host byte order), so it interoperates on
 * same-endian hosts.
 */
#ifndef NIXL_SRC_CORE_NIXL_TCPSTORE_CLIENT_H
#define NIXL_SRC_CORE_NIXL_TCPSTORE_CLIENT_H

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/** Move-only socket descriptor owner; closes on destruction, no-op when empty. */
class nixlSocketFd {
public:
    nixlSocketFd() = default;

    explicit nixlSocketFd(int fd) noexcept : fd_(fd) {}

    ~nixlSocketFd() {
        reset();
    }

    nixlSocketFd(nixlSocketFd &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

    nixlSocketFd &
    operator=(nixlSocketFd &&other) noexcept {
        if (this != &other) {
            reset();
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }

    nixlSocketFd(const nixlSocketFd &) = delete;
    nixlSocketFd &
    operator=(const nixlSocketFd &) = delete;

    [[nodiscard]] int
    get() const noexcept {
        return fd_;
    }

    [[nodiscard]] explicit
    operator bool() const noexcept {
        return fd_ >= 0;
    }

    void
    reset() noexcept;

private:
    int fd_ = -1;
};

class nixlTcpStoreClient {
public:
    // Connects to host:port, runs the c10d VALIDATE/PING handshake, and arms
    // the socket send/recv timeout. Throws std::runtime_error on any failure,
    // so a constructed client is a connected one (health gate). connect_timeout
    // is the bring-up budget; op_timeout bounds each operation once running.
    nixlTcpStoreClient(std::string host,
                       std::uint16_t port,
                       std::chrono::milliseconds connect_timeout,
                       std::chrono::milliseconds op_timeout);

    ~nixlTcpStoreClient() = default;

    nixlTcpStoreClient(const nixlTcpStoreClient &) = delete;
    nixlTcpStoreClient &
    operator=(const nixlTcpStoreClient &) = delete;

    // Upsert (last-writer-wins).
    void
    set(const std::string &key, const std::string &value);

    // Value for the key, or nullopt when it is absent. Presence is resolved
    // internally with the c10d CHECK query, whose answer for a missing key is
    // defined (a bare GET is not).
    [[nodiscard]] std::optional<std::string>
    get(const std::string &key);

    // Returns true when exactly one key was deleted.
    bool
    deleteKey(const std::string &key);

private:
    // Resolve, connect one address, arm the I/O timeouts, handshake.
    void
    connect(std::chrono::milliseconds timeout);

    // Reconnect when a previous operation dropped the socket. A partial
    // exchange desyncs the framing, so it is closed rather than reused; ops are
    // re-issued whole, so a fresh connection is equivalent.
    void
    ensureConnected();

    // VALIDATE (required first query) followed by a PING round-trip.
    void
    handshake();

    // Both bound the whole call by a deadline taken on entry: SO_SNDTIMEO /
    // SO_RCVTIMEO only bound one syscall, so a peer dribbling bytes could
    // stretch a single transfer to len times the socket timeout.
    void
    sendAll(const void *data, std::size_t len);

    void
    recvAll(void *data, std::size_t len);

    // Rejects absurd lengths so a desynced response cannot trigger an
    // unbounded allocation.
    [[nodiscard]] std::string
    recvBlob();

    const std::string host_;
    const std::uint16_t port_;
    const std::chrono::milliseconds opTimeout_;

    nixlSocketFd fd_;
    std::mutex mutex_; // serializes each request/response exchange on the socket
};

#endif // NIXL_SRC_CORE_NIXL_TCPSTORE_CLIENT_H
