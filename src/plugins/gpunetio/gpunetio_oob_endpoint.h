/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GPUNETIO_OOB_ENDPOINT_H
#define GPUNETIO_OOB_ENDPOINT_H

#include <arpa/inet.h>

#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

constexpr uint16_t GPUNETIO_DEFAULT_OOB_PORT = 6544;

struct GpunetioOobEndpoint {
    std::string ipv4;
    uint16_t port;
};

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

inline std::string
formatGpunetioOobEndpoint(std::string_view ipv4, uint16_t port) {
    if (port == GPUNETIO_DEFAULT_OOB_PORT) {
        return std::string(ipv4);
    }
    return std::string(ipv4) + ":" + std::to_string(port);
}

#endif
