/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_CORE_TELEMETRY_TELEMETRY_EVENT_H
#define NIXL_SRC_CORE_TELEMETRY_TELEMETRY_EVENT_H

#include <array>
#include <cstdint>
#include <string_view>

#include "nixl_types.h"

constexpr char TELEMETRY_BUFFER_SIZE_VAR[] = "NIXL_TELEMETRY_BUFFER_SIZE";
constexpr char TELEMETRY_RUN_INTERVAL_VAR[] = "NIXL_TELEMETRY_RUN_INTERVAL";

constexpr inline int TELEMETRY_VERSION = 3;

/**
 * @enum nixl_telemetry_event_type_t
 * @brief Enumerates all known telemetry event types.
 */
enum class nixl_telemetry_event_type_t : uint32_t {
    AGENT_TX_BYTES = 0,
    AGENT_RX_BYTES = 1,
    AGENT_TX_REQUESTS_NUM = 2,
    AGENT_RX_REQUESTS_NUM = 3,
    AGENT_MEMORY_REGISTERED = 4,
    AGENT_MEMORY_DEREGISTERED = 5,
    AGENT_XFER_TIME = 6,
    AGENT_XFER_POST_TIME = 7,
    AGENT_ERR_NOT_POSTED = 8,
    AGENT_ERR_INVALID_PARAM = 9,
    AGENT_ERR_BACKEND = 10,
    AGENT_ERR_NOT_FOUND = 11,
    AGENT_ERR_MISMATCH = 12,
    AGENT_ERR_NOT_ALLOWED = 13,
    AGENT_ERR_REPOST_ACTIVE = 14,
    AGENT_ERR_UNKNOWN = 15,
    AGENT_ERR_NOT_SUPPORTED = 16,
    AGENT_ERR_REMOTE_DISCONNECT = 17,
    AGENT_ERR_CANCELED = 18,
    AGENT_ERR_NO_TELEMETRY = 19,
};

[[nodiscard]] nixl_telemetry_event_type_t
nixlTelemetryEventTypeForStatus(nixl_status_t s);

namespace nixlEnumStrings {
[[nodiscard]] constexpr std::string_view
telemetryEventTypeStr(const nixl_telemetry_event_type_t type) noexcept {
    switch (type) {
    case nixl_telemetry_event_type_t::AGENT_TX_BYTES:
        return "agent_tx_bytes";
    case nixl_telemetry_event_type_t::AGENT_RX_BYTES:
        return "agent_rx_bytes";
    case nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM:
        return "agent_tx_requests_num";
    case nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM:
        return "agent_rx_requests_num";
    case nixl_telemetry_event_type_t::AGENT_MEMORY_REGISTERED:
        return "agent_memory_registered";
    case nixl_telemetry_event_type_t::AGENT_MEMORY_DEREGISTERED:
        return "agent_memory_deregistered";
    case nixl_telemetry_event_type_t::AGENT_XFER_TIME:
        return "agent_xfer_time";
    case nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME:
        return "agent_xfer_post_time";
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_POSTED:
        return "agent_err_not_posted";
    case nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM:
        return "agent_err_invalid_param";
    case nixl_telemetry_event_type_t::AGENT_ERR_BACKEND:
        return "agent_err_backend";
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_FOUND:
        return "agent_err_not_found";
    case nixl_telemetry_event_type_t::AGENT_ERR_MISMATCH:
        return "agent_err_mismatch";
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_ALLOWED:
        return "agent_err_not_allowed";
    case nixl_telemetry_event_type_t::AGENT_ERR_REPOST_ACTIVE:
        return "agent_err_repost_active";
    case nixl_telemetry_event_type_t::AGENT_ERR_UNKNOWN:
        return "agent_err_unknown";
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_SUPPORTED:
        return "agent_err_not_supported";
    case nixl_telemetry_event_type_t::AGENT_ERR_REMOTE_DISCONNECT:
        return "agent_err_remote_disconnect";
    case nixl_telemetry_event_type_t::AGENT_ERR_CANCELED:
        return "agent_err_canceled";
    case nixl_telemetry_event_type_t::AGENT_ERR_NO_TELEMETRY:
        return "agent_err_no_telemetry";
    }
    return "unknown_event";
}

constexpr std::array<nixl_telemetry_event_type_t, 12> telemetry_error_event_types{{
    nixl_telemetry_event_type_t::AGENT_ERR_NOT_POSTED,
    nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM,
    nixl_telemetry_event_type_t::AGENT_ERR_BACKEND,
    nixl_telemetry_event_type_t::AGENT_ERR_NOT_FOUND,
    nixl_telemetry_event_type_t::AGENT_ERR_MISMATCH,
    nixl_telemetry_event_type_t::AGENT_ERR_NOT_ALLOWED,
    nixl_telemetry_event_type_t::AGENT_ERR_REPOST_ACTIVE,
    nixl_telemetry_event_type_t::AGENT_ERR_UNKNOWN,
    nixl_telemetry_event_type_t::AGENT_ERR_NOT_SUPPORTED,
    nixl_telemetry_event_type_t::AGENT_ERR_REMOTE_DISCONNECT,
    nixl_telemetry_event_type_t::AGENT_ERR_CANCELED,
    nixl_telemetry_event_type_t::AGENT_ERR_NO_TELEMETRY,
}};

[[nodiscard]] constexpr const char *
telemetryErrorStatusLabel(const nixl_telemetry_event_type_t type) noexcept {
    switch (type) {
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_POSTED:
        return "not_posted";
    case nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM:
        return "invalid_param";
    case nixl_telemetry_event_type_t::AGENT_ERR_BACKEND:
        return "backend";
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_FOUND:
        return "not_found";
    case nixl_telemetry_event_type_t::AGENT_ERR_MISMATCH:
        return "mismatch";
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_ALLOWED:
        return "not_allowed";
    case nixl_telemetry_event_type_t::AGENT_ERR_REPOST_ACTIVE:
        return "repost_active";
    case nixl_telemetry_event_type_t::AGENT_ERR_UNKNOWN:
        return "unknown";
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_SUPPORTED:
        return "not_supported";
    case nixl_telemetry_event_type_t::AGENT_ERR_REMOTE_DISCONNECT:
        return "remote_disconnect";
    case nixl_telemetry_event_type_t::AGENT_ERR_CANCELED:
        return "canceled";
    case nixl_telemetry_event_type_t::AGENT_ERR_NO_TELEMETRY:
        return "no_telemetry";
    case nixl_telemetry_event_type_t::AGENT_TX_BYTES:
    case nixl_telemetry_event_type_t::AGENT_RX_BYTES:
    case nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM:
    case nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM:
    case nixl_telemetry_event_type_t::AGENT_MEMORY_REGISTERED:
    case nixl_telemetry_event_type_t::AGENT_MEMORY_DEREGISTERED:
    case nixl_telemetry_event_type_t::AGENT_XFER_TIME:
    case nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME:
        return nullptr;
    }
    return nullptr;
}
} // namespace nixlEnumStrings

/**
 * @struct nixlTelemetryEvent
 * @brief A structure to hold individual telemetry event data for cyclic buffer storage
 */
struct nixlTelemetryEvent {
    nixl_telemetry_event_type_t eventType_; // Detailed event type/identifier
    uint64_t value_; // Numeric value associated with the event

    nixlTelemetryEvent() noexcept = default;

    nixlTelemetryEvent(nixl_telemetry_event_type_t event_type, uint64_t value) noexcept
        : eventType_(event_type),
          value_(value) {}
};

#endif
