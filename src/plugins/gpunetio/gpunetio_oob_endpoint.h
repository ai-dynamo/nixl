/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_OOB_ENDPOINT_H
#define NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_OOB_ENDPOINT_H

#include <arpa/inet.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

constexpr uint16_t GPUNETIO_DEFAULT_OOB_PORT = 6544;
constexpr std::size_t GPUNETIO_MAX_AGENT_NAME_SIZE = 4096;

/** Return whether an OOB agent-name payload length is safe to allocate. */
inline bool
isValidGpunetioAgentNameSize(std::size_t size) noexcept {
    return size > 0 && size <= GPUNETIO_MAX_AGENT_NAME_SIZE;
}

/** Parsed GPUNETIO IPv4 endpoint and TCP port. */
struct GpunetioOobEndpoint {
    std::string ipv4;
    uint16_t port;
};

/**
 * Parse a decimal TCP port in [1, 65535]. An empty value selects the default port.
 *
 * @throws std::invalid_argument if the value is not a valid TCP port.
 */
inline uint16_t
parseGpunetioOobPort(std::string_view value) {
    if (value.empty()) {
        return GPUNETIO_DEFAULT_OOB_PORT;
    }

    uint32_t port = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), port);
    if (error != std::errc() || end != value.data() + value.size() || port == 0 || port > 65535) {
        throw std::invalid_argument("oob_port must be an integer in the range [1, 65535]");
    }
    return static_cast<uint16_t>(port);
}

/**
 * Parse connection metadata encoded as IPv4 or IPv4:port.
 *
 * IPv4-only metadata selects the default port for backward compatibility.
 * @throws std::invalid_argument if the metadata or port is malformed.
 */
inline GpunetioOobEndpoint
parseGpunetioOobEndpoint(std::string_view metadata) {
    if (metadata.empty() || metadata.find('\0') != std::string_view::npos) {
        throw std::invalid_argument("GPUNETIO connection metadata is empty or malformed");
    }

    const size_t separator = metadata.find(':');
    if (separator != std::string_view::npos &&
        (separator == 0 || separator + 1 == metadata.size() ||
         metadata.find(':', separator + 1) != std::string_view::npos)) {
        throw std::invalid_argument("GPUNETIO connection metadata must be IPv4 or IPv4:port");
    }

    const std::string ipv4(metadata.substr(0, separator));
    in_addr address{};
    if (inet_pton(AF_INET, ipv4.c_str(), &address) != 1) {
        throw std::invalid_argument(
            "GPUNETIO connection metadata contains an invalid IPv4 address");
    }

    const uint16_t port = separator == std::string_view::npos ?
        GPUNETIO_DEFAULT_OOB_PORT :
        parseGpunetioOobPort(metadata.substr(separator + 1));
    return {ipv4, port};
}

/** Format an IPv4 endpoint, omitting the port when it is the default. */
inline std::string
formatGpunetioOobEndpoint(std::string_view ipv4, uint16_t port) {
    if (port == GPUNETIO_DEFAULT_OOB_PORT) {
        return std::string(ipv4);
    }
    return std::string(ipv4) + ":" + std::to_string(port);
}

#endif // NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_OOB_ENDPOINT_H
