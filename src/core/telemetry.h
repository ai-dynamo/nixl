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

#include "common/cyclic_buffer.h"
#include "nixl_types.h"
#include <string>
#include <vector>

#ifdef NIXL_ENABLE_TELEMETRY

class nixlTelemetry {
    nixlTelemetry();

public:
    static nixlTelemetry *
    getInstance() {
        static nixlTelemetry instance;
        return &instance;
    }
    void
    writeEvent();
    bool
    isEnabled() const {
        return true;
    }
    void
    updateTxBytes(uint64_t tx_bytes);
    void
    updateRxBytes(uint64_t rx_bytes);
    void
    updateTxRequestsNum(uint32_t num);
    void
    updateRxRequestsNum(uint32_t num);
    void
    updateErrorCount(nixl_status_t error_type);
    void
    updateMemoryRegistered(int64_t memory_registered);

    void
    addRequestTime(std::chrono::microseconds transaction_time);

    // Transaction time tracking methods
    void
    updateTransactionTime(std::chrono::microseconds transaction_time);
    void
    resetTransactionTimeStats();

    // Plugin telemetry methods (new)
    void
    updatePluginTelemetry(const std::string &plugin_name,
                          const std::vector<nixlPluginTelemetryMetric> &metrics);

    ~nixlTelemetry();

private:
    SharedRingBuffer<nixlTelemetryEvent, TELEMETRY_BUFFER_SIZE> buffer_;
    nixlTelemetryEvent event_;
    uint8_t last_requests_index_;
    bool updated_;
    bool enabled_;
};

#else // Telemetry disabled

// Telemetry disabled at compile time - provide no-op implementations
class nixlTelemetry {
public:
    static nixlTelemetry *
    getInstance() {
        static nixlTelemetry instance;
        return &instance;
    }

    // No-op methods when telemetry is disabled
    void
    writeEvent() {}
    bool
    isEnabled() const {
        return false;
    }
    void
    updateTxBytes(uint64_t) {}
    void
    updateRxBytes(uint64_t) {}
    void
    updateTxRequestsNum(uint32_t) {}
    void
    updateRxRequestsNum(uint32_t) {}
    void
    updateErrorCount(nixl_status_t error_type) {}
    void
    updateMemoryRegistered(int64_t) {}

    void
    updateTransactionTime(std::chrono::microseconds) {}

    void
    addRequestTime(std::chrono::microseconds) {}

    void
    updatePluginTelemetry(const std::string &, const std::vector<nixlPluginTelemetryMetric> &) {}

    ~nixlTelemetry() = default;
};

#endif // Telemetry enabled/disabled

#endif // _NIXL_TELEMETRY_H
