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
#include "telemetry_plugin_manager.h"
#include "util.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

constexpr std::chrono::milliseconds DEFAULT_TELEMETRY_RUN_INTERVAL = 100ms;
constexpr size_t DEFAULT_TELEMETRY_BUFFER_SIZE = 4096;

nixlTelemetry::nixlTelemetry(const std::string &agent_name, backend_map_t &backend_map)
    : pool_(1),
      writeTask_(pool_.get_executor(), DEFAULT_TELEMETRY_RUN_INTERVAL, false),
      backendMap_(backend_map) {

    initializeTelemetry(agent_name);
}

nixlTelemetry::~nixlTelemetry() {
    writeTask_.enabled_ = false;
    try {
        writeTask_.timer_.cancel();
        pool_.stop();
        pool_.join();
    }
    catch (const asio::system_error &e) {
        NIXL_DEBUG << "Failed to cancel telemetry write timer: " << e.what();
        // continue anyway since it's not critical
    }
}

void
nixlTelemetry::initializeTelemetry(const std::string &agent_name) {
    auto buffer_size = std::getenv(TELEMETRY_BUFFER_SIZE_VAR) ?
        std::stoul(std::getenv(TELEMETRY_BUFFER_SIZE_VAR)) :
        DEFAULT_TELEMETRY_BUFFER_SIZE;

    if (buffer_size == 0) {
        throw std::invalid_argument("Telemetry buffer size cannot be 0");
    }

    if (agent_name.empty()) {
        throw std::invalid_argument("Agent name cannot be empty");
    }

    // Check if exporter is enabled and which type to use
    const char *exporter_type_env = std::getenv(telemetryExporterVar);
    if (!exporter_type_env) {
        return;
    }

    std::string exporter_type = exporter_type_env;

    NIXL_INFO << "Telemetry exporter enabled, type: " << exporter_type;

    // Prepare initialization parameters for the exporter
    const nixlTelemetryExporterInitParams init_params = {
        .outputPath = std::getenv(telemetryExporterOutputPathVar) ?
            std::getenv(telemetryExporterOutputPathVar) :
            "",
        .agentName = agent_name,
        .maxEventsBuffered = buffer_size};

    // Create exporter through plugin manager
    auto &exporter_manager = nixlTelemetryPluginManager::getInstance();
    exporter_ = exporter_manager.createExporter(exporter_type, init_params);

    if (!exporter_) {
        NIXL_WARN << "Failed to create telemetry exporter '" << exporter_type
                  << "', telemetry will not be exported";
        throw std::runtime_error("Failed to create telemetry exporter");
    }

    auto run_interval = std::getenv(TELEMETRY_RUN_INTERVAL_VAR) ?
        std::chrono::milliseconds(std::stoul(std::getenv(TELEMETRY_RUN_INTERVAL_VAR))) :
        DEFAULT_TELEMETRY_RUN_INTERVAL;

    // Update write task interval and start it
    writeTask_.callback_ = [this]() { return telemetryExporterHelper(); };
    writeTask_.interval_ = run_interval;
    writeTask_.enabled_ = true;
    registerPeriodicTask(writeTask_);
}

bool
nixlTelemetry::telemetryExporterHelper() {
    std::vector<nixlTelemetryEvent> next_queue;
    next_queue.reserve(exporter_->getMaxEventsBuffered());
    {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.swap(next_queue);
    }
    for (const auto &event : next_queue) {
        // if full, ignore
        exporter_->exportEvent(event);
    }
    // collect backend events and sort them by timestamp
    next_queue.clear();
    // std::vector<nixlTelemetryEvent> backend_events;
    for (const auto &backend : backendMap_) {
        auto events = backend.second->getTelemetryEvents();
        for (auto &event : events) {
            // don't trust enum value coming from backend,
            // as it might be different from the one in agent
            event.category_ = nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND;
            next_queue.push_back(event);
        }
    }
    std::sort(next_queue.begin(),
              next_queue.end(),
              [](const nixlTelemetryEvent &a, const nixlTelemetryEvent &b) {
                  return a.timestampUs_ < b.timestampUs_;
              });
    for (const auto &event : next_queue) {
        exporter_->exportEvent(event);
    }
    return true;
}

void
nixlTelemetry::registerPeriodicTask(periodicTask &task) {
    task.timer_.expires_after(task.interval_);
    task.timer_.async_wait([this, &task](const asio::error_code &ec) {
        if (ec != asio::error::operation_aborted) {

            task.callback_();

            if (!task.enabled_) {
                return;
            }

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
