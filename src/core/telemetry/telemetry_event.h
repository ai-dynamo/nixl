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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "nixl_types.h"

constexpr char TELEMETRY_BUFFER_SIZE_VAR[] = "NIXL_TELEMETRY_BUFFER_SIZE";
constexpr char TELEMETRY_RUN_INTERVAL_VAR[] = "NIXL_TELEMETRY_RUN_INTERVAL";

constexpr inline int TELEMETRY_VERSION = 2;

/**
 * @enum nixl_telemetry_category_t
 * @brief An enumeration of main telemetry event categories for easy filtering and aggregation
 */
enum class nixl_telemetry_category_t {
    NIXL_TELEMETRY_MEMORY = 0, // Memory operations (register, deregister, allocation)
    NIXL_TELEMETRY_TRANSFER = 1, // Data transfer operations (read, write)
    NIXL_TELEMETRY_CONNECTION = 2, // Connection management (connect, disconnect)
    NIXL_TELEMETRY_BACKEND = 3, // Backend-specific operations
    NIXL_TELEMETRY_ERROR = 4, // Error events
    NIXL_TELEMETRY_PERFORMANCE = 5, // Performance metrics
    NIXL_TELEMETRY_SYSTEM = 6, // System-level events
    NIXL_TELEMETRY_CUSTOM = 7, // Custom/user-defined events
};

/**
 * @enum nixl_telemetry_event_type_t
 * @brief Enumerates all known telemetry event types.
 */
enum class nixl_telemetry_event_type_t : uint8_t {
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
    AGENT_BACKEND = 20,
};

[[nodiscard]] std::optional<nixl_telemetry_event_type_t>
nixlTelemetryEventTypeForStatus(nixl_status_t s);

namespace nixlEnumStrings {
std::string
telemetryCategoryStr(const nixl_telemetry_category_t &category);

[[nodiscard]] constexpr std::string_view
telemetryEventTypeStr(nixl_telemetry_event_type_t type) noexcept {
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
    case nixl_telemetry_event_type_t::AGENT_BACKEND:
        return "agent_backend";
    }
    return "unknown_event";
}
}

/**
 * @struct nixlTelemetryEvent
 * @brief A structure to hold individual telemetry event data for cyclic buffer storage
 */
struct nixlTelemetryEvent {
    nixl_telemetry_category_t category_; // Main event category for filtering
    nixl_telemetry_event_type_t eventType_; // Detailed event type/identifier
    uint64_t value_; // Numeric value associated with the event

    nixlTelemetryEvent() noexcept = default;

    nixlTelemetryEvent(nixl_telemetry_category_t category,
                       nixl_telemetry_event_type_t event_type,
                       uint64_t value) noexcept
        : category_(category),
          eventType_(event_type),
          value_(value) {}
};

#endif
