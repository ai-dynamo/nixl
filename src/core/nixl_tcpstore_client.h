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

#include "common/scoped_fd.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// All operations run on the worker thread of the backend that owns this client,
// so nothing here is synchronized.
class nixlTcpStoreClient {
public:
    // Does no I/O: the connection and the c10d VALIDATE/PING handshake happen on
    // the first ensureConnected(), so a store that is briefly unreachable does
    // not fail agent construction. connect_timeout is the bring-up budget for
    // that first connection; op_timeout bounds each operation, and each later
    // reconnect, once running.
    nixlTcpStoreClient(std::string host,
                       std::uint16_t port,
                       std::chrono::milliseconds connect_timeout,
                       std::chrono::milliseconds op_timeout) noexcept;

    ~nixlTcpStoreClient() = default;

    nixlTcpStoreClient(const nixlTcpStoreClient &) = delete;
    nixlTcpStoreClient &
    operator=(const nixlTcpStoreClient &) = delete;

    // Connect on first use, and reconnect when a previous operation dropped the
    // socket. A partial exchange desyncs the framing, so it is closed rather
    // than reused; ops are re-issued whole, so a fresh connection is equivalent.
    // Every operation calls it; the owning backend also calls it at start-up so
    // an unreachable store is reported then and not at the first operation.
    void
    ensureConnected();

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
    using deadline_t = std::chrono::steady_clock::time_point;

    // Bring-up within a single budget: resolve, try each address with whatever
    // time is left, arm the I/O timeouts, handshake against the same deadline.
    void
    connect(std::chrono::milliseconds timeout);

    // VALIDATE (required first query) followed by a PING round-trip.
    void
    handshake(deadline_t deadline);

    // The deadline covers the whole operation, not one transfer: SO_SNDTIMEO /
    // SO_RCVTIMEO only bound a single syscall, so a peer dribbling bytes could
    // otherwise stretch an exchange to many times the socket timeout.
    void
    sendAll(const void *data, std::size_t len, deadline_t deadline);

    void
    recvAll(void *data, std::size_t len, deadline_t deadline);

    // Rejects absurd lengths so a desynced response cannot trigger an
    // unbounded allocation.
    [[nodiscard]] std::string
    recvBlob(deadline_t deadline);

    // Deadline for one operation, taken when it starts.
    [[nodiscard]] deadline_t
    opDeadline() const noexcept {
        return std::chrono::steady_clock::now() + opTimeout_;
    }

    const std::string host_;
    const std::uint16_t port_;
    const std::chrono::milliseconds connectTimeout_;
    const std::chrono::milliseconds opTimeout_;

    nixl::scopedFd fd_;
    // Only the first connection gets the bring-up budget; later reconnects use
    // the op budget so a retry cannot hold the worker for the whole window.
    bool bringUp_ = true;
};

#endif // NIXL_SRC_CORE_NIXL_TCPSTORE_CLIENT_H
