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
#include "ucx_conn_info.h"

#include <cstring>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

#include "common/nixl_log.h"

namespace nixl::ucx {

namespace {

constexpr std::string_view family_inet = "inet";
constexpr std::string_view family_inet6 = "inet6";

[[nodiscard]] socklen_t
addrlenOf(int family) noexcept {
    switch (family) {
    case AF_INET:
        return sizeof(sockaddr_in);
    case AF_INET6:
        return sizeof(sockaddr_in6);
    default:
        return 0;
    }
}

} // namespace

sockaddrConnInfo::sockaddrConnInfo(const sockaddr *addr, socklen_t addrlen) {
    if (addr == nullptr || addrlen == 0 || addrlen > socklen_t(sizeof(storage_))) {
        return;
    }
    std::memcpy(&storage_, addr, addrlen);

    /* Callers such as ucp_listener_query() hand out a full sockaddr_storage;
     * normalize to the length that matches the address family. */
    const socklen_t family_len = addrlenOf(storage_.ss_family);
    addrlen_ = (family_len != 0 && family_len <= addrlen) ? family_len : addrlen;
}

std::optional<sockaddrConnInfo>
sockaddrConnInfo::resolve(const std::string &host, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    const std::string service = std::to_string(port);
    addrinfo *result = nullptr;
    const int rc = getaddrinfo(host.c_str(), service.c_str(), &hints, &result);
    if (rc != 0 || result == nullptr) {
        NIXL_ERROR << "Failed to resolve address " << host << ":" << port << ": "
                   << gai_strerror(rc);
        return std::nullopt;
    }

    // Prefer IPv4 when the name resolves to both, to keep the default path simple.
    const addrinfo *chosen = result;
    for (const addrinfo *it = result; it != nullptr; it = it->ai_next) {
        if (it->ai_family == AF_INET) {
            chosen = it;
            break;
        }
    }

    sockaddrConnInfo info(chosen->ai_addr, chosen->ai_addrlen);
    freeaddrinfo(result);

    if (!info.valid()) {
        return std::nullopt;
    }
    return info;
}

bool
sockaddrConnInfo::isSockaddrBlob(std::string_view blob) noexcept {
    return blob.size() > magic.size() && blob.substr(0, magic.size()) == magic;
}

std::optional<sockaddrConnInfo>
sockaddrConnInfo::deserialize(std::string_view blob) {
    if (!isSockaddrBlob(blob)) {
        return std::nullopt;
    }

    const std::vector<std::string> tokens = absl::StrSplit(blob, ' ');
    if (tokens.size() != 4) {
        NIXL_ERROR << "Malformed UCX sockaddr connection info: " << blob;
        return std::nullopt;
    }

    const std::string &family_str = tokens[1];
    const std::string &addr_str = tokens[2];

    uint32_t port = 0;
    if (!absl::SimpleAtoi(tokens[3], &port) || port > UINT16_MAX) {
        NIXL_ERROR << "Malformed port in UCX sockaddr connection info: " << blob;
        return std::nullopt;
    }

    sockaddrConnInfo info;
    if (family_str == family_inet) {
        auto *sin = reinterpret_cast<sockaddr_in *>(&info.storage_);
        sin->sin_family = AF_INET;
        sin->sin_port = htons(uint16_t(port));
        if (inet_pton(AF_INET, addr_str.c_str(), &sin->sin_addr) != 1) {
            NIXL_ERROR << "Malformed IPv4 address in UCX sockaddr connection info: " << blob;
            return std::nullopt;
        }
        info.addrlen_ = sizeof(sockaddr_in);
    } else if (family_str == family_inet6) {
        auto *sin6 = reinterpret_cast<sockaddr_in6 *>(&info.storage_);
        sin6->sin6_family = AF_INET6;
        sin6->sin6_port = htons(uint16_t(port));
        if (inet_pton(AF_INET6, addr_str.c_str(), &sin6->sin6_addr) != 1) {
            NIXL_ERROR << "Malformed IPv6 address in UCX sockaddr connection info: " << blob;
            return std::nullopt;
        }
        info.addrlen_ = sizeof(sockaddr_in6);
    } else {
        NIXL_ERROR << "Unsupported address family in UCX sockaddr connection info: " << blob;
        return std::nullopt;
    }

    return info;
}

std::string
sockaddrConnInfo::serialize() const {
    if (!valid()) {
        return {};
    }

    char buf[INET6_ADDRSTRLEN] = {};
    std::string_view family;

    if (storage_.ss_family == AF_INET) {
        const auto *sin = reinterpret_cast<const sockaddr_in *>(&storage_);
        inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
        family = family_inet;
    } else if (storage_.ss_family == AF_INET6) {
        const auto *sin6 = reinterpret_cast<const sockaddr_in6 *>(&storage_);
        inet_ntop(AF_INET6, &sin6->sin6_addr, buf, sizeof(buf));
        family = family_inet6;
    } else {
        return {};
    }

    return std::string(magic) + " " + std::string(family) + " " + buf + " " +
        std::to_string(port());
}

std::string
sockaddrConnInfo::str() const {
    const std::string serialized = serialize();
    if (serialized.empty()) {
        return "<invalid>";
    }
    // Strip "<magic> <family> " and render as address:port.
    const std::vector<std::string> tokens = absl::StrSplit(serialized, ' ');
    return tokens[2] + ":" + tokens[3];
}

bool
sockaddrConnInfo::isWildcard() const noexcept {
    if (storage_.ss_family == AF_INET) {
        const auto *sin = reinterpret_cast<const sockaddr_in *>(&storage_);
        return sin->sin_addr.s_addr == htonl(INADDR_ANY);
    }
    if (storage_.ss_family == AF_INET6) {
        const auto *sin6 = reinterpret_cast<const sockaddr_in6 *>(&storage_);
        return std::memcmp(&sin6->sin6_addr, &in6addr_any, sizeof(in6addr_any)) == 0;
    }
    return false;
}

uint16_t
sockaddrConnInfo::port() const noexcept {
    if (storage_.ss_family == AF_INET) {
        return ntohs(reinterpret_cast<const sockaddr_in *>(&storage_)->sin_port);
    }
    if (storage_.ss_family == AF_INET6) {
        return ntohs(reinterpret_cast<const sockaddr_in6 *>(&storage_)->sin6_port);
    }
    return 0;
}

void
sockaddrConnInfo::setPort(uint16_t port) noexcept {
    if (storage_.ss_family == AF_INET) {
        reinterpret_cast<sockaddr_in *>(&storage_)->sin_port = htons(port);
    } else if (storage_.ss_family == AF_INET6) {
        reinterpret_cast<sockaddr_in6 *>(&storage_)->sin6_port = htons(port);
    }
}

} // namespace nixl::ucx
