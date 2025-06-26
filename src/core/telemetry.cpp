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
#include <chrono>
#include <sstream>
#include <thread>
#include <filesystem>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "common/nixl_log.h"
#include "telemetry.h"
#include "nixl_types.h"
#include "util.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

nixlTelemetry::nixlTelemetry(const std::string file) : pool_(1) {
    enabled_ = std::getenv("NIXL_ENABLE_TELEMETRY") != nullptr;
    if (!enabled_) {
        NIXL_INFO << "Telemetry disabled via NIXL_ENABLE_TELEMETRY environment variable";
        return;
    }
    bufferSize_ = std::getenv("NIXL_TELEMETRY_BUFFER_SIZE") ?
        std::stoul(std::getenv("NIXL_TELEMETRY_BUFFER_SIZE")) :
        DEFAULT_TELEMETRY_BUFFER_SIZE;

    auto folder_path =
        std::getenv("NIXL_TELEMETRY_DIR") ? std::getenv("NIXL_TELEMETRY_DIR") : "/tmp";

    auto file_name =
        file.empty() ? TELEMETRY_PREFIX + std::string(".") + std::to_string(getpid()) : file;

    auto full_file_path = fs::path(folder_path) / file_name;

    if (bufferSize_ == 0) {
        throw std::invalid_argument("Telemetry buffer size cannot be 0");
    }

    NIXL_INFO << "Telemetry enabled, using buffer path: " << full_file_path
              << " with size: " << bufferSize_;

    buffer_ = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        full_file_path, true, bufferSize_, TELEMETRY_VERSION);
    runInterval_ = std::getenv("NIXL_TELEMETRY_RUN_INTERVAL") ?
        std::stoul(std::getenv("NIXL_TELEMETRY_RUN_INTERVAL")) :
        DEFAULT_TELEMETRY_RUN_INTERVAL;
    timer_ = std::make_shared<asio::steady_timer>(pool_.get_executor());
    writeEvent();
}

nixlTelemetry::~nixlTelemetry() {
    if (timer_) {
        timer_->cancel();
    }
    // flush last events
    if (buffer_) writeEventHelper();
}

void
nixlTelemetry::writeEventHelper() {
    auto max_events_to_dump = buffer_->capacity();
    std::vector<nixlTelemetryEvent> events_to_dump;
    events_to_dump.reserve(buffer_->capacity() / 2);
    events_.swap(events_to_dump);
    for (size_t i = 0; i < std::min(max_events_to_dump, events_to_dump.size()); i++) {
        if (!buffer_->push(events_to_dump[i])) {
            NIXL_DEBUG << "Failed to push event to buffer, buffer is full";
        }
    }
}

void
nixlTelemetry::writeEvent() {
    timer_->expires_after(std::chrono::milliseconds(runInterval_));
    timer_->async_wait([this](const asio::error_code &ec) {
        if (ec != asio::error::operation_aborted) {
            auto start_time = std::chrono::steady_clock::now();

            writeEventHelper();

            auto end_time = std::chrono::steady_clock::now();
            auto execution_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // Schedule next execution with adjusted interval
            auto next_interval = std::chrono::milliseconds(runInterval_) - execution_time;
            if (next_interval.count() < 0) {
                next_interval = std::chrono::milliseconds(0);
            }

            // Schedule the next write operation
            writeEvent();
        }
    });
}

void
nixlTelemetry::updateData(const std::string &event_name,
                          nixl_telemetry_category_t category,
                          uint64_t value) {
    if (!enabled_) return;
    nixlTelemetryEvent event(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count(),
                             category,
                             event_name,
                             value);

    // agent can be multi-threaded
    std::lock_guard<std::mutex> lock(pluginTelemetryMutex_);
    events_.push_back(event);
}

void
nixlTelemetry::updateTxBytes(uint64_t tx_bytes) {
    updateData("agent_tx_bytes", nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, tx_bytes);
}

void
nixlTelemetry::updateRxBytes(uint64_t rx_bytes) {
    updateData("agent_rx_bytes", nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, rx_bytes);
}

void
nixlTelemetry::updateTxRequestsNum(uint32_t tx_requests_num) {
    updateData("agent_tx_requests_num",
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
               tx_requests_num);
}

void
nixlTelemetry::updateRxRequestsNum(uint32_t rx_requests_num) {
    updateData("agent_rx_requests_num",
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
               rx_requests_num);
}

void
nixlTelemetry::updateErrorCount(nixl_status_t error_type) {
    updateData(
        nixlEnumStrings::statusStr(error_type), nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR, 1);
}

void
nixlTelemetry::updateMemoryRegistered(uint64_t memory_registered) {
    updateData("agent_memory_registered",
               nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
               memory_registered);
}

void
nixlTelemetry::updateMemoryDeregistered(uint64_t memory_deregistered) {
    updateData("agent_memory_deregistered",
               nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
               memory_deregistered);
}

void
nixlTelemetry::addTransactionTime(std::chrono::microseconds transaction_time) {
    updateData("agent_transaction_time",
               nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
               transaction_time.count());
}

void
nixlTelemetry::addGeneralTelemetry(const std::string &event_name, uint64_t value) {
    updateData(event_name, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND, value);
}
