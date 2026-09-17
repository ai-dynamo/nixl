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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_CONN_INFO_H
#define NIXL_SRC_PLUGINS_UCX_UCX_CONN_INFO_H

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <netinet/in.h>
#include <sys/socket.h>

namespace nixl::ucx {

/**
 * Connection info exchanged between agents when the UCX backend runs in
 * connection_mode=sockaddr: the address of the local UCP listener.
 *
 * Wire format is a versioned, human readable string:
 *     "NIXLUCXSA/1 <family> <address> <port>"
 * e.g. "NIXLUCXSA/1 inet 192.168.10.27 18515"
 *      "NIXLUCXSA/1 inet6 fe80::1 18515"
 *
 * The magic prefix lets the peer tell a sockaddr blob apart from the raw
 * binary UCX worker address used by connection_mode=worker_address, so a
 * mode mismatch between two agents fails loudly instead of mysteriously.
 */
class sockaddrConnInfo {
public:
    static constexpr std::string_view magic = "NIXLUCXSA/1";

    sockaddrConnInfo() = default;

    /** Build from an already resolved socket address (e.g. from ucp_listener_query). */
    sockaddrConnInfo(const sockaddr *addr, socklen_t addrlen);

    /**
     * Resolve host (numeric IPv4/IPv6 address or host name) and port.
     * Returns std::nullopt when the address cannot be resolved.
     */
    [[nodiscard]] static std::optional<sockaddrConnInfo>
    resolve(const std::string &host, uint16_t port);

    /** True if the blob was produced by serialize(), i.e. carries the magic prefix. */
    [[nodiscard]] static bool
    isSockaddrBlob(std::string_view blob) noexcept;

    /** Parse a blob produced by serialize(). Returns std::nullopt on any malformed input. */
    [[nodiscard]] static std::optional<sockaddrConnInfo>
    deserialize(std::string_view blob);

    [[nodiscard]] std::string
    serialize() const;

    /** "<address>:<port>", for logs and error messages. */
    [[nodiscard]] std::string
    str() const;

    [[nodiscard]] const sockaddr *
    addr() const noexcept {
        return reinterpret_cast<const sockaddr *>(&storage_);
    }

    [[nodiscard]] socklen_t
    addrlen() const noexcept {
        return addrlen_;
    }

    [[nodiscard]] bool
    valid() const noexcept {
        return addrlen_ != 0;
    }

    /** True for INADDR_ANY / in6addr_any, which must never be advertised to a peer. */
    [[nodiscard]] bool
    isWildcard() const noexcept;

    [[nodiscard]] uint16_t
    port() const noexcept;

    void
    setPort(uint16_t port) noexcept;

private:
    sockaddr_storage storage_{};
    socklen_t addrlen_{0};
};

} // namespace nixl::ucx

#endif
