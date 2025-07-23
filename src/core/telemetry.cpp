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

#include "common/nixl_log.h"
#include "telemetry.h"
#include "nixl_types.h"
#include "common/thread_executor.h"
#include "util.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

#ifdef NIXL_ENABLE_TELEMETRY

std::string
getTelemetryBufferName() {
    std::stringstream ss;
    std::string telemetry_dir =
        std::getenv("NIXL_TELEMETRY_DIR") ? std::getenv("NIXL_TELEMETRY_DIR") : "/tmp";
    ss << telemetry_dir << "/" << TELEMETRY_PREFIX << "." << getpid();
    return ss.str();
}

nixlTelemetry::nixlTelemetry() : buffer_(), update_index_(0), enabled_(false) {
    // Check environment variable for runtime disable
}

bool
nixlTelemetry::initialize(const std::string file, size_t buffer_size) {
    enabled_ = std::getenv("NIXL_ENABLE_TELEMETRY") != nullptr;
    if (!enabled_) {
        NIXL_INFO << "Telemetry disabled via NIXL_ENABLE_TELEMETRY environment variable";
        return false;
    }

    auto folder_path =
        std::getenv("NIXL_TELEMETRY_DIR") ? std::getenv("NIXL_TELEMETRY_DIR") : "/tmp";

    auto file_name =
        file.empty() ? TELEMETRY_PREFIX + std::string(".") + std::to_string(getpid()) : file;

    auto file_path = fs::path(folder_path) / file_name;

    buffer_size = std::getenv("NIXL_TELEMETRY_BUFFER_SIZE") ?
        std::stoul(std::getenv("NIXL_TELEMETRY_BUFFER_SIZE")) :
        buffer_size;

    // Buffer size validation
    if (buffer_size == 0) {
        NIXL_WARN << "Buffer size cannot be 0, using default size";
        buffer_size = DEFAULT_TELEMETRY_BUFFER_SIZE;
    }
    NIXL_INFO << "Telemetry enabled, using buffer path: " << file_path
              << " with size: " << buffer_size;
    buffer_.initialize(file_path.c_str(), buffer_size, true, TELEMETRY_VERSION);
    auto run_interval = std::getenv("NIXL_TELEMETRY_RUN_INTERVAL") ?
        std::stoul(std::getenv("NIXL_TELEMETRY_RUN_INTERVAL")) :
        100;
    nixl::ThreadExecutor::getInstance().registerPeriodicTask(
        "telemetry", [this]() { this->writeEvent(); }, std::chrono::milliseconds(run_interval));
    return true;
}

nixlTelemetry::~nixlTelemetry() {
    if (enabled_) {
        nixl::ThreadExecutor::getInstance().unregisterTask("telemetry");
    }
}

void
nixlTelemetry::writeEvent() {
    // move update_index_ to the other index and update current index
    bool index_to_dump = update_index_;
    {
        std::lock_guard<std::mutex> lock(plugin_telemetry_mutex_);
        update_index_ = !update_index_;
    }
    // dump the index to dump
    auto &index_to_dump_events = plugin_telemetry_[index_to_dump];
    for (const auto &event : index_to_dump_events) {
        buffer_.push(event);
    }
    index_to_dump_events.clear();
}

void
nixlTelemetry::update_data(const std::string &event_name,
                           nixl_telemetry_category_t category,
                           uint64_t value) {
    if (!enabled_) {
        return;
    }
    nixlTelemetryEvent event;
    event.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
    event.category = category;
    strncpy(event.event_name, event_name.c_str(), MAX_EVENT_NAME_LEN - 1);
    event.event_name[MAX_EVENT_NAME_LEN - 1] = '\0';
    event.value = value;
    std::lock_guard<std::mutex> lock(plugin_telemetry_mutex_);
    plugin_telemetry_[update_index_].push_back(event);
}

void
nixlTelemetry::updateTxBytes(uint64_t tx_bytes) {
    update_data("agent_tx_bytes", nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, tx_bytes);
}

void
nixlTelemetry::updateRxBytes(uint64_t rx_bytes) {
    update_data("agent_rx_bytes", nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, rx_bytes);
}

void
nixlTelemetry::updateTxRequestsNum(uint32_t tx_requests_num) {
    update_data("agent_tx_requests_num",
                nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
                tx_requests_num);
}

void
nixlTelemetry::updateRxRequestsNum(uint32_t rx_requests_num) {
    update_data("agent_rx_requests_num",
                nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
                rx_requests_num);
}

void
nixlTelemetry::updateErrorCount(nixl_status_t error_type) {
    update_data("agent_error_" + nixlEnumStrings::statusStr(error_type),
                nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR,
                1);
}

void
nixlTelemetry::updateMemoryRegistered(uint64_t memory_registered) {
    update_data("agent_memory_registered",
                nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
                memory_registered);
}

void
nixlTelemetry::updateMemoryDeregistered(uint64_t memory_deregistered) {
    update_data("agent_memory_deregistered",
                nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
                memory_deregistered);
}

void
nixlTelemetry::addTransactionTime(std::chrono::microseconds transaction_time) {
    update_data("agent_transaction_time",
                nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
                transaction_time.count());
}

void
nixlTelemetry::addGeneralTelemetry(const std::string &event_name, uint64_t value) {
    update_data(event_name, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND, value);
}

#endif // NIXL_ENABLE_TELEMETRY
