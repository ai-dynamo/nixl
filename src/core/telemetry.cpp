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
#include "telemetry_event.h"
#include "util.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

constexpr std::chrono::milliseconds DEFAULT_TELEMETRY_RUN_INTERVAL = 100ms;
constexpr size_t DEFAULT_TELEMETRY_BUFFER_SIZE = 4096;

nixlTelemetry::nixlTelemetry(const std::string &file_path, backend_map_t &backend_map)
    : pool_(1),
      writeTask_(pool_.get_executor(), DEFAULT_TELEMETRY_RUN_INTERVAL),
      file_(file_path),
      backendMap_(backend_map) {
    if (file_path.empty()) {
        throw std::invalid_argument("Telemetry file path cannot be empty");
    }
    initializeTelemetry();
}

nixlTelemetry::~nixlTelemetry() {
    try {
        writeTask_.callback_ = nullptr;
        writeTask_.timer_.cancel();
    }
    catch (const asio::system_error &e) {
        NIXL_DEBUG << "Failed to cancel telemetry write timer: " << e.what();
        // continue anyway since it's not critical
    }
    if (buffer_) {
        writeEventHelper();
        buffer_.reset();
    }
}

void
nixlTelemetry::initializeTelemetry() {
    auto buffer_size = std::getenv(TELEMETRY_BUFFER_SIZE_VAR) ?
        std::stoul(std::getenv(TELEMETRY_BUFFER_SIZE_VAR)) :
        DEFAULT_TELEMETRY_BUFFER_SIZE;

    auto full_file_path = fs::path(file_);

    if (buffer_size == 0) {
        throw std::invalid_argument("Telemetry buffer size cannot be 0");
    }

    NIXL_INFO << "Telemetry enabled, using buffer path: " << full_file_path
              << " with size: " << buffer_size;

    buffer_ = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        full_file_path, true, TELEMETRY_VERSION, buffer_size);

    auto run_interval = std::getenv(TELEMETRY_RUN_INTERVAL_VAR) ?
        std::chrono::milliseconds(std::stoul(std::getenv(TELEMETRY_RUN_INTERVAL_VAR))) :
        DEFAULT_TELEMETRY_RUN_INTERVAL;

    // Update write task interval and start it
    writeTask_.callback_ = [this]() { return writeEventHelper(); };
    writeTask_.interval_ = run_interval;
    registerPeriodicTask(writeTask_);
}

bool
nixlTelemetry::writeEventHelper() {
    std::vector<nixlTelemetryEvent> next_queue;
    // assume next buffer will be the same size as the current one
    next_queue.reserve(buffer_->capacity());
    {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.swap(next_queue);
    }
    for (auto &event : next_queue) {
        // if full, ignore
        buffer_->push(event);
    }
    // collect all events and sort them by timestamp
    std::vector<nixlTelemetryEvent> all_events;
    for (auto &backend : backendMap_) {
        auto backend_events = backend.second->getTelemetryEvents();
        for (auto &event : backend_events) {
            // don't trust enum value coming from backend,
            // as it might be different from the one in agent
            event.category_ = nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND;
            all_events.push_back(event);
        }
    }
    std::sort(all_events.begin(),
              all_events.end(),
              [](const nixlTelemetryEvent &a, const nixlTelemetryEvent &b) {
                  return a.timestampUs_ < b.timestampUs_;
              });
    for (auto &event : all_events) {
        buffer_->push(event);
    }
    return true;
}

void
nixlTelemetry::registerPeriodicTask(periodicTask &task) {
    task.timer_.expires_after(task.interval_);
    task.timer_.async_wait([this, &task](const asio::error_code &ec) {
        if (ec != asio::error::operation_aborted) {
            auto start_time = std::chrono::steady_clock::now();

            if (!task.callback_ || !task.callback_()) {
                // if return false, stop the task
                return;
            }

            auto end_time = std::chrono::steady_clock::now();
            auto execution_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // Schedule next execution with adjusted interval
            auto next_interval = std::chrono::milliseconds(task.interval_) - execution_time;
            if (next_interval.count() < 0) {
                next_interval = std::chrono::milliseconds(0);
            }

            // Schedule the next operation
            task.interval_ = next_interval;
            registerPeriodicTask(task);
        }
    });
}

void
nixlTelemetry::updateData(const std::string &event_name,
                          nixl_telemetry_category_t category,
                          uint64_t value) {
    // agent can be multi-threaded
    std::lock_guard<std::mutex> lock(mutex_);
    events_.emplace_back(std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count(),
                         category,
                         event_name,
                         value);
}

// The next 4 methods might be removed, as addXferTime covers them.
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
nixlTelemetry::addXferTime(std::chrono::microseconds xfer_time, bool is_write, uint64_t bytes) {
    std::string bytes_name;
    std::string requests_name;

    if (is_write) {
        bytes_name = "agent_tx_bytes";
        requests_name = "agent_tx_requests_num";
    } else {
        bytes_name = "agent_rx_bytes";
        requests_name = "agent_rx_requests_num";
    }
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
    std::lock_guard<std::mutex> lock(mutex_);
    events_.emplace_back(time,
                         nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
                         "agent_xfer_time",
                         xfer_time.count());
    events_.emplace_back(
        time, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, bytes_name.c_str(), bytes);
    events_.emplace_back(
        time, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, requests_name.c_str(), 1);
}

void
nixlTelemetry::addPostTime(std::chrono::microseconds post_time) {
    updateData("agent_xfer_post_time",
               nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
               post_time.count());
}

std::string
nixlEnumStrings::telemetryCategoryStr(const nixl_telemetry_category_t &category) {
    static std::array<std::string, 9> nixl_telemetry_category_str = {"NIXL_TELEMETRY_MEMORY",
                                                                     "NIXL_TELEMETRY_TRANSFER",
                                                                     "NIXL_TELEMETRY_CONNECTION",
                                                                     "NIXL_TELEMETRY_BACKEND",
                                                                     "NIXL_TELEMETRY_ERROR",
                                                                     "NIXL_TELEMETRY_PERFORMANCE",
                                                                     "NIXL_TELEMETRY_SYSTEM",
                                                                     "NIXL_TELEMETRY_CUSTOM",
                                                                     "NIXL_TELEMETRY_MAX"};
    size_t category_int = static_cast<size_t>(category);
    if (category_int >= nixl_telemetry_category_str.size()) return "BAD_CATEGORY";
    return nixl_telemetry_category_str[category_int];
}
