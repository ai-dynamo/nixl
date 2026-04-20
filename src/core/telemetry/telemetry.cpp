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
#include <chrono>
#include <sstream>
#include <thread>
#include <filesystem>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>

#include "common/configuration.h"
#include "common/nixl_log.h"
#include "telemetry.h"
#include "telemetry_event.h"
#include "util.h"
#include "plugin_manager.h"
#include "buffer_exporter.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

std::optional<nixl_telemetry_event_type_t>
nixlTelemetryEventTypeForStatus(nixl_status_t s) {
    switch (s) {
    case NIXL_SUCCESS:
    case NIXL_IN_PROG:
        return std::nullopt;
    case NIXL_ERR_NOT_POSTED:
        return nixl_telemetry_event_type_t::AGENT_ERR_NOT_POSTED;
    case NIXL_ERR_INVALID_PARAM:
        return nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM;
    case NIXL_ERR_BACKEND:
        return nixl_telemetry_event_type_t::AGENT_ERR_BACKEND;
    case NIXL_ERR_NOT_FOUND:
        return nixl_telemetry_event_type_t::AGENT_ERR_NOT_FOUND;
    case NIXL_ERR_MISMATCH:
        return nixl_telemetry_event_type_t::AGENT_ERR_MISMATCH;
    case NIXL_ERR_NOT_ALLOWED:
        return nixl_telemetry_event_type_t::AGENT_ERR_NOT_ALLOWED;
    case NIXL_ERR_REPOST_ACTIVE:
        return nixl_telemetry_event_type_t::AGENT_ERR_REPOST_ACTIVE;
    case NIXL_ERR_UNKNOWN:
        return nixl_telemetry_event_type_t::AGENT_ERR_UNKNOWN;
    case NIXL_ERR_NOT_SUPPORTED:
        return nixl_telemetry_event_type_t::AGENT_ERR_NOT_SUPPORTED;
    case NIXL_ERR_REMOTE_DISCONNECT:
        return nixl_telemetry_event_type_t::AGENT_ERR_REMOTE_DISCONNECT;
    case NIXL_ERR_CANCELED:
        return nixl_telemetry_event_type_t::AGENT_ERR_CANCELED;
    case NIXL_ERR_NO_TELEMETRY:
        return nixl_telemetry_event_type_t::AGENT_ERR_NO_TELEMETRY;
    }
    NIXL_ASSERT_ALWAYS(false) << "nixlTelemetryEventTypeForStatus: unhandled nixl_status_t "
                              << static_cast<int>(s) << "; add a case when extending nixl_status_t";
}

constexpr std::chrono::milliseconds DEFAULT_TELEMETRY_RUN_INTERVAL = 100ms;
constexpr size_t DEFAULT_TELEMETRY_BUFFER_SIZE = 4096;
constexpr const char *defaultTelemetryPlugin = "BUFFER";

nixlTelemetry::nixlTelemetry(const std::string &agent_name)
    : pool_(1),
      writeTask_(pool_.get_executor(), DEFAULT_TELEMETRY_RUN_INTERVAL, false),
      agentName_(agent_name) {
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

namespace {
[[nodiscard]] std::optional<std::string>
getExporterName() {
    if (const auto name = nixl::config::getValueOptional<std::string>(telemetryExporterVar)) {
        if (!name->empty()) {
            return name;
        }
        NIXL_DEBUG << "Ignoring empty " << telemetryExporterVar << " environment variable";
    }

    if (!nixl::config::checkExistence(telemetryDirVar)) {
        NIXL_DEBUG << telemetryDirVar
                   << " is not set, NIXL telemetry is enabled without any exporter";
        return std::nullopt;
    }

    NIXL_INFO << "No telemetry exporter was specified, using default: " << defaultTelemetryPlugin;

    return defaultTelemetryPlugin;
}

} // namespace

void
nixlTelemetry::initializeTelemetry() {
    maxBufferedEvents_ = nixl::config::getValueDefaulted<size_t>(TELEMETRY_BUFFER_SIZE_VAR,
                                                                 DEFAULT_TELEMETRY_BUFFER_SIZE);

    if (maxBufferedEvents_ == 0) {
        throw std::invalid_argument("Telemetry buffer size cannot be 0");
    }

    const std::optional<std::string> exporter_name = getExporterName();

    if (!exporter_name) {
        return;
    }

    auto &plugin_manager = nixlPluginManager::getInstance();
    std::shared_ptr<const nixlTelemetryPluginHandle> plugin_handle =
        plugin_manager.loadTelemetryPlugin(*exporter_name);

    if (plugin_handle == nullptr) {
        throw std::runtime_error("Failed to load telemetry plugin: " + *exporter_name);
    }

    const nixlTelemetryExporterInitParams init_params{agentName_, maxBufferedEvents_};
    exporter_ = plugin_handle->createExporter(init_params);
    if (!exporter_) {
        NIXL_ERROR << "Failed to create telemetry exporter: " << *exporter_name;
        return;
    }

    NIXL_DEBUG << "NIXL telemetry is enabled with exporter: " << *exporter_name;

    const auto run_interval =
        nixl::config::getValueDefaulted(TELEMETRY_RUN_INTERVAL_VAR, DEFAULT_TELEMETRY_RUN_INTERVAL);

    // Update write task interval and start it
    writeTask_.callback_ = [this]() { return writeEventHelper(); };
    writeTask_.interval_ = run_interval;
    writeTask_.enabled_ = true;
    registerPeriodicTask(writeTask_);
}

bool
nixlTelemetry::writeEventHelper() {
    std::vector<nixlTelemetryEvent> next_queue;
    next_queue.reserve(maxBufferedEvents_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.swap(next_queue);
    }

    for (auto &event : next_queue) {
        // if full, ignore
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
nixlTelemetry::updateData(nixl_telemetry_event_type_t event_type,
                          nixl_telemetry_category_t category,
                          uint64_t value) {
    // agent can be multi-threaded
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.size() >= maxBufferedEvents_) {
        return;
    }
    events_.emplace_back(category, event_type, value);
}

// The next 4 methods might be removed, as addXferTime covers them.
void
nixlTelemetry::updateTxBytes(uint64_t tx_bytes) {
    updateData(nixl_telemetry_event_type_t::AGENT_TX_BYTES,
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
               tx_bytes);
}

void
nixlTelemetry::updateRxBytes(uint64_t rx_bytes) {
    updateData(nixl_telemetry_event_type_t::AGENT_RX_BYTES,
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
               rx_bytes);
}

void
nixlTelemetry::updateTxRequestsNum(uint32_t tx_requests_num) {
    updateData(nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM,
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
               tx_requests_num);
}

void
nixlTelemetry::updateRxRequestsNum(uint32_t rx_requests_num) {
    updateData(nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM,
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
               rx_requests_num);
}

void
nixlTelemetry::updateErrorCount(nixl_status_t error_type) {
    NIXL_ASSERT_ALWAYS(static_cast<int>(error_type) < 0)
        << "nixlTelemetry::updateErrorCount expects a negative nixl_status_t error code";
    const auto event_type = nixlTelemetryEventTypeForStatus(error_type);
    NIXL_ASSERT_ALWAYS(event_type.has_value())
        << "nixlTelemetryEventTypeForStatus missing mapping for nixl_status_t "
        << static_cast<int>(error_type);
    updateData(*event_type, nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR, 1);
}

void
nixlTelemetry::updateMemoryRegistered(uint64_t memory_registered) {
    updateData(nixl_telemetry_event_type_t::AGENT_MEMORY_REGISTERED,
               nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
               memory_registered);
}

void
nixlTelemetry::updateMemoryDeregistered(uint64_t memory_deregistered) {
    updateData(nixl_telemetry_event_type_t::AGENT_MEMORY_DEREGISTERED,
               nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
               memory_deregistered);
}

void
nixlTelemetry::addXferTime(std::chrono::microseconds xfer_time, bool is_write, uint64_t bytes) {
    const auto bytes_type = is_write ? nixl_telemetry_event_type_t::AGENT_TX_BYTES :
                                       nixl_telemetry_event_type_t::AGENT_RX_BYTES;
    const auto requests_type = is_write ? nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM :
                                          nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM;

    const std::lock_guard lock(mutex_);
    if (events_.size() + 3 > maxBufferedEvents_) {
        return;
    }
    events_.emplace_back(nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
                         nixl_telemetry_event_type_t::AGENT_XFER_TIME,
                         static_cast<uint64_t>(xfer_time.count()));
    events_.emplace_back(nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, bytes_type, bytes);
    events_.emplace_back(nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, requests_type, 1);
}

void
nixlTelemetry::addPostTime(std::chrono::microseconds post_time) {
    updateData(nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME,
               nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
               static_cast<uint64_t>(post_time.count()));
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
