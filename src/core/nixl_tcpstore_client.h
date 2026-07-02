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
#include <string>
#include <vector>

class nixlTcpStoreClient {
public:
    // Connects to host:port, runs the c10d VALIDATE/PING handshake, and arms
    // the socket send/recv timeout. Throws std::runtime_error on any failure,
    // so a constructed client is a connected one (health gate).
    nixlTcpStoreClient(const std::string &host,
                       std::uint16_t port,
                       std::chrono::milliseconds timeout);

    ~nixlTcpStoreClient();

    nixlTcpStoreClient(const nixlTcpStoreClient &) = delete;
    nixlTcpStoreClient &
    operator=(const nixlTcpStoreClient &) = delete;

    // Upsert (last-writer-wins).
    void
    set(const std::string &key, const std::string &value);

    // Presence check; does not block waiting for the key to appear.
    [[nodiscard]] bool
    check(const std::string &key);

    // Value for an existing key (call check() first; absent keys read empty).
    [[nodiscard]] std::string
    get(const std::string &key);

    // Batched read; callers must ensure every key exists (server reads block
    // otherwise). Returns one value per key, in order.
    [[nodiscard]] std::vector<std::string>
    multiGet(const std::vector<std::string> &keys);

    // Returns true when exactly one key was deleted.
    bool
    deleteKey(const std::string &key);

private:
    void
    sendAll(const void *data, std::size_t len);

    void
    recvAll(void *data, std::size_t len);

    // Reads a uint64 length-prefixed value, rejecting absurd lengths so a
    // corrupt/desynced response cannot trigger an unbounded allocation.
    [[nodiscard]] std::string
    recvBlob();

    int fd_ = -1;
    std::mutex mutex_; // serializes each request/response exchange on the socket
};

#endif // NIXL_SRC_CORE_NIXL_TCPSTORE_CLIENT_H
