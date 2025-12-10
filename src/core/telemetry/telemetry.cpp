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
#include "plugin_manager.h"
#include "buffer_exporter.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

constexpr std::chrono::milliseconds DEFAULT_TELEMETRY_RUN_INTERVAL = 100ms;
constexpr size_t DEFAULT_TELEMETRY_BUFFER_SIZE = 4096;
constexpr const char *defaultTelemetryPlugin = "BUFFER";

nixlTelemetry::nixlTelemetry(const std::string &agent_name, backend_map_t &backend_map)
    : pool_(1),
      writeTask_(pool_.get_executor(), DEFAULT_TELEMETRY_RUN_INTERVAL, false),
      agentName_(agent_name),
      backendMap_(backend_map) {
    if (agent_name.empty()) {
        throw std::invalid_argument("Expected non-empty agent name in nixl telemetry create");
    }
    initializeTelemetry();
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

    if (buffer_size == 0) {
        throw std::invalid_argument("Telemetry buffer size cannot be 0");
    }

    const char *exporter_name = std::getenv(telemetryExporterVar);

    if (!exporter_name) {
        NIXL_INFO << "No telemetry exporter was specified, using default: "
                  << defaultTelemetryPlugin;
        exporter_name = defaultTelemetryPlugin;
        if (!std::getenv(telemetryDirVar)) {
            NIXL_DEBUG << telemetryDirVar
                       << " is not set, NIXL telemetry is enabled without any exporter";
            return;
        }
    }
    auto &plugin_manager = nixlPluginManager::getInstance();
    std::shared_ptr<const nixlTelemetryPluginHandle> plugin_handle =
        plugin_manager.loadTelemetryPlugin(exporter_name);

    if (plugin_handle == nullptr) {
        throw std::runtime_error("Failed to load telemetry plugin: " + std::string(exporter_name));
    }

    const nixlTelemetryExporterInitParams init_params{agentName_, buffer_size};
    exporter_ = plugin_handle->createExporter(init_params);
    if (!exporter_) {
        NIXL_ERROR << "Failed to create telemetry exporter: " << exporter_name;
        return;
    }

    NIXL_DEBUG << "NIXL telemetry is enabled with " << exporter_name << "exporter";

    auto run_interval = std::getenv(TELEMETRY_RUN_INTERVAL_VAR) ?
        std::chrono::milliseconds(std::stoul(std::getenv(TELEMETRY_RUN_INTERVAL_VAR))) :
        DEFAULT_TELEMETRY_RUN_INTERVAL;

    // Update write task interval and start it
    writeTask_.callback_ = [this]() { return writeEventHelper(); };
    writeTask_.interval_ = run_interval;
    writeTask_.enabled_ = true;
    registerPeriodicTask(writeTask_);
}

bool
nixlTelemetry::writeEventHelper() {
    std::vector<nixlTelemetryEvent> next_queue;
    // assume next buffer will be the same size as the current one
    next_queue.reserve(exporter_->getMaxEventsBuffered());
    {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.swap(next_queue);
    }
    for (auto &event : next_queue) {
        // if full, ignore
        exporter_->exportEvent(event);
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
