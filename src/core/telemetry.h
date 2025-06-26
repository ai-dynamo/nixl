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
#include <mutex>
#include <memory>

#include <asio.hpp>

constexpr size_t DEFAULT_TELEMETRY_BUFFER_SIZE = 4096;
constexpr size_t DEFAULT_TELEMETRY_RUN_INTERVAL = 100;

#ifdef NIXL_ENABLE_TELEMETRY

class nixlTelemetry {
public:
    nixlTelemetry(const std::string file);

    ~nixlTelemetry();

    bool
    isEnabled() const {
        return enabled_;
    }

    void
    writeEvent();
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
    updateMemoryRegistered(uint64_t memory_registered);
    void
    updateMemoryDeregistered(uint64_t memory_deregistered);
    void
    addTransactionTime(std::chrono::microseconds transaction_time);
    void
    addGeneralTelemetry(const std::string &event_name, uint64_t value);

private:
    void
    updateData(const std::string &event_name, nixl_telemetry_category_t category, uint64_t value);
    void
    writeEventHelper();
    std::unique_ptr<sharedRingBuffer<nixlTelemetryEvent>> buffer_;
    std::vector<nixlTelemetryEvent> events_;
    std::size_t bufferSize_;
    std::mutex pluginTelemetryMutex_;
    asio::thread_pool pool_;
    std::shared_ptr<asio::steady_timer> timer_;
    std::size_t runInterval_;
    bool enabled_;
};

#else // Telemetry disabled

// Telemetry disabled at compile time - provide no-op implementations
class nixlTelemetry {
public:
    nixlTelemetry(const std::string file) {}

    ~nixlTelemetry() = default;

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
    updateMemoryRegistered(uint64_t memory_registered) {}

    void
    updateMemoryDeregistered(uint64_t memory_deregistered) {}

    void
    addTransactionTime(std::chrono::microseconds transaction_time) {}

    void
    addGeneralTelemetry(const std::string &event_name, uint64_t value) {}
};

#endif // Telemetry enabled/disabled

#endif // _NIXL_TELEMETRY_H
