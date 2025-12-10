/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _NIXL_TELEMETRY_H
#define _NIXL_TELEMETRY_H

#include <cstdint>
#include <cstring>

#include "nixl_types.h"

constexpr char TELEMETRY_BUFFER_SIZE_VAR[] = "NIXL_TELEMETRY_BUFFER_SIZE";
constexpr char TELEMETRY_RUN_INTERVAL_VAR[] = "NIXL_TELEMETRY_RUN_INTERVAL";

constexpr int TELEMETRY_VERSION = 1;
constexpr size_t MAX_EVENT_NAME_LEN = 32;

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

namespace nixlEnumStrings {
std::string
telemetryCategoryStr(const nixl_telemetry_category_t &category);
}

/**
 * @struct nixlTelemetryEvent
 * @brief A structure to hold individual telemetry event data for cyclic buffer storage
 */
struct nixlTelemetryEvent {
    uint64_t timestampUs_;
    nixl_telemetry_category_t category_; // Main event category for filtering
    char eventName_[MAX_EVENT_NAME_LEN]; // Detailed event name/identifier
    uint64_t value_; // Numeric value associated with the event
    nixlTelemetryEvent() = default;

    nixlTelemetryEvent(uint64_t timestamp_us,
                       nixl_telemetry_category_t category,
                       const std::string &event_name,
                       uint64_t value)
        : timestampUs_(timestamp_us),
          category_(category),
          value_(value) {
        strncpy(eventName_, event_name.c_str(), MAX_EVENT_NAME_LEN - 1);
        eventName_[MAX_EVENT_NAME_LEN - 1] = '\0';
    }
};

#endif // _NIXL_TELEMETRY_H
