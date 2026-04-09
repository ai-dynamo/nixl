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
#include <cstring>
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

void
fillTelemetryEventSlot(nixlTelemetryEvent &slot,
                       nixl_telemetry_category_t category,
                       const char *event_name,
                       uint64_t value) noexcept {
    slot.category_ = category;
    strncpy(slot.eventName_, event_name, MAX_EVENT_NAME_LEN - 1);
    slot.eventName_[MAX_EVENT_NAME_LEN - 1] = '\0';
    slot.value_ = value;
}

} // namespace

void
nixlTelemetry::initializeTelemetry() {
    const auto buffer_size = nixl::config::getValueDefaulted<size_t>(TELEMETRY_BUFFER_SIZE_VAR,
                                                                     DEFAULT_TELEMETRY_BUFFER_SIZE);

    if (buffer_size == 0) {
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

    const nixlTelemetryExporterInitParams init_params{agentName_, buffer_size};
    exporter_ = plugin_handle->createExporter(init_params);
    if (!exporter_) {
        NIXL_ERROR << "Failed to create telemetry exporter: " << *exporter_name;
        return;
    }

    NIXL_DEBUG << "NIXL telemetry is enabled with exporter: " << *exporter_name;

    maxEventsBuffered_ = buffer_size;
    eventBuffers_[0].resize(maxEventsBuffered_);
    eventBuffers_[1].resize(maxEventsBuffered_);
    writeIdx_.store(0);
    writeBufIndex_.store(0);

    const auto run_interval =
        nixl::config::getValueDefaulted(TELEMETRY_RUN_INTERVAL_VAR, DEFAULT_TELEMETRY_RUN_INTERVAL);

    // Update write task interval and start it
    writeTask_.callback_ = [this]() { return writeEventHelper(); };
    writeTask_.interval_ = run_interval;
    writeTask_.enabled_ = true;
    registerPeriodicTask(writeTask_);
    recording = true;
}

bool
nixlTelemetry::writeEventHelper() {

    // Block producers by setting writeIdx_=max so they get id>=max and skip.
    const int w = writeBufIndex_.load();
    const size_t num_events_raw = writeIdx_.exchange(maxEventsBuffered_);
    writeBufIndex_.store(1 - w);
    writeIdx_.store(0);
    const size_t num_events = std::min(num_events_raw, maxEventsBuffered_);

    for (size_t i = 0; i < num_events; ++i) {
        exporter_->exportEvent(eventBuffers_[w][i]);
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

    const size_t id = writeIdx_.fetch_add(1);
    if (id >= maxEventsBuffered_) {
        return;
    }

    fillTelemetryEventSlot(eventBuffers_[writeBufIndex_.load()][id], category, event_name.c_str(),
                           value);
}

// The next 4 methods might be removed, as addTransferComplete covers them.
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
    updateData("agent_tx_requests_num", nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
               tx_requests_num);
}

void
nixlTelemetry::updateRxRequestsNum(uint32_t rx_requests_num) {
    updateData("agent_rx_requests_num", nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
               rx_requests_num);
}

void
nixlTelemetry::updateErrorCount(nixl_status_t error_type) {
    updateData("agent_error", nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR,
               static_cast<uint64_t>(error_type));
}

void
nixlTelemetry::updateMemoryRegistered(uint64_t memory_registered) {
    updateData("agent_memory_registered", nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
               memory_registered);
}

void
nixlTelemetry::updateMemoryDeregistered(uint64_t memory_deregistered) {
    updateData("agent_memory_deregistered", nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
               memory_deregistered);
}

// addXferTime and addPostTime might be removed, as addTransferComplete covers them.
void
nixlTelemetry::addXferTime(std::chrono::microseconds xfer_time, bool is_write, uint64_t bytes) {
    const char *bytes_name = is_write ? "agent_tx_bytes" : "agent_rx_bytes";
    const char *requests_name = is_write ? "agent_tx_requests_num" : "agent_rx_requests_num";

    const size_t id_base = writeIdx_.fetch_add(3);
    if (id_base + 2 >= maxEventsBuffered_) {
        return;
    }
    auto &arr = eventBuffers_[writeBufIndex_.load()];
    fillTelemetryEventSlot(arr[id_base], nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
                           "agent_xfer_time", static_cast<uint64_t>(xfer_time.count()));
    fillTelemetryEventSlot(arr[id_base + 1], nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
                           bytes_name, bytes);
    fillTelemetryEventSlot(arr[id_base + 2], nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
                           requests_name, 1);
}

void
nixlTelemetry::addPostTime(std::chrono::microseconds post_time) {
    updateData("agent_xfer_post_time", nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
               post_time.count());
}

void
nixlTelemetry::addTransferComplete(std::chrono::microseconds post_time,
                                   std::chrono::microseconds xfer_time,
                                   bool is_write,
                                   uint64_t bytes) {
    const size_t id = writeIdx_.fetch_add(4);
    if (id + 4 > maxEventsBuffered_) {
        return;
    }

    const int buf = writeBufIndex_.load(std::memory_order_acquire);
    auto &arr = eventBuffers_[buf];

    fillTelemetryEventSlot(arr[id], nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
                           "agent_xfer_post_time", static_cast<uint64_t>(post_time.count()));
    fillTelemetryEventSlot(arr[id + 1], nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
                           "agent_xfer_time", static_cast<uint64_t>(xfer_time.count()));
    fillTelemetryEventSlot(arr[id + 2], nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
                           is_write ? "agent_tx_bytes" : "agent_rx_bytes", bytes);
    fillTelemetryEventSlot(arr[id + 3], nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
                           is_write ? "agent_tx_requests_num" : "agent_rx_requests_num", 1);
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
