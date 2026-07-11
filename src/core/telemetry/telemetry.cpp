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
#include <array>
#include <chrono>
#include <sstream>
#include <thread>
#include <filesystem>
#include <fnmatch.h>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>
#include <set>
#include <vector>

#include "common/configuration.h"
#include "common/nixl_duration.h"
#include "common/nixl_log.h"
#include "common/str_util.h"
#include "telemetry.h"
#include "telemetry_event.h"
#include "util.h"
#include "plugin_manager.h"
#include "buffer_exporter.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

[[nodiscard]] nixl_telemetry_event_type_t
nixlTelemetryEventTypeForStatus(nixl_status_t s) {
    switch (s) {
    case NIXL_SUCCESS:
    case NIXL_IN_PROG:
        NIXL_ASSERT_ALWAYS(false)
            << "nixlTelemetryEventTypeForStatus expects a negative nixl_status_t error code";
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
// Collect-only sink (registered as "NOP"): events are still recorded in process
// (so getXferTelemetry() returns data) but nothing is written out. Used when
// telemetry is explicitly requested but no output sink is configured.
constexpr const char *collectOnlyTelemetryPlugin = "NOP";

namespace {
// Defined below; declared here for use by create() and the constructor's
// member-initializer list.
[[nodiscard]] std::string
getExporterName();
[[nodiscard]] std::string
validateAgentName(const std::string &agent_name);
[[nodiscard]] size_t
resolveMaxBufferedEvents();
[[nodiscard]] nixl_telemetry_metric_mask_t
resolveEnabledMetrics();
} // namespace

[[nodiscard]] std::unique_ptr<nixlTelemetry>
nixlTelemetry::create(const std::string &agent_name) {
    try {
        return std::make_unique<nixlTelemetry>(agent_name, getExporterName());
    }
    catch (const nixlTelemetryBindFailed &e) {
        NIXL_WARN << e.what() << "; this process will run without a telemetry sink";
        return nullptr;
    }
}

nixlTelemetry::nixlTelemetry(const std::string &agent_name, const std::string &exporter_name)
    : agentName_(validateAgentName(agent_name)),
      maxBufferedEvents_(resolveMaxBufferedEvents()),
      exporter_(makeExporter(exporter_name)),
      metricEnabled_(resolveEnabledMetrics()),
      stagingQueue_(maxBufferedEvents_),
      pool_(1),
      writeTask_(pool_.get_executor(), DEFAULT_TELEMETRY_RUN_INTERVAL, false) {
    // A constructed nixlTelemetry always has an exporter (makeExporter throws
    // otherwise), so start the periodic export task.
    startExportTask();

    if (!nixlTime::fastClockUsesHwCounter()) {
        NIXL_WARN << "telemetry: no invariant CPU counter available; per-transfer timing falls "
                     "back to steady_clock (higher overhead)";
    }
}

nixlTelemetry::~nixlTelemetry() {
    writeTask_.enabled_ = false;
    try {
        // The pool thread re-arms writeTask_.timer_ from its own callback
        // (registerPeriodicTask -> async_wait), so cancelling the timer here on
        // the main thread races that re-arm. Stop and join the pool first; once
        // join() returns no other thread touches the timer, so cancel() is safe.
        pool_.stop();
        pool_.join();
        writeTask_.timer_.cancel();
    }
    catch (const asio::system_error &e) {
        NIXL_DEBUG << "Failed to cancel telemetry write timer: " << e.what();
        // continue anyway since it's not critical
    }
}

namespace {
[[nodiscard]] std::string
getExporterName() {
    if (const auto name = nixl::config::getValueOptional<std::string>(telemetryExporterVar)) {
        if (!name->empty()) {
            return *name;
        }
        NIXL_DEBUG << "Ignoring empty " << telemetryExporterVar << " environment variable";
    }

    if (nixl::config::checkExistence(telemetryDirVar)) {
        NIXL_INFO << "No telemetry exporter was specified, using default: "
                  << defaultTelemetryPlugin;
        return defaultTelemetryPlugin;
    }

    // No output sink configured; collect in-process via the NOP sink.
    NIXL_INFO << "No telemetry sink configured; using in-process telemetry collection via "
              << collectOnlyTelemetryPlugin;
    return collectOnlyTelemetryPlugin;
}

[[nodiscard]] std::string
validateAgentName(const std::string &agent_name) {
    if (agent_name.empty()) {
        throw std::invalid_argument("Expected non-empty agent name in nixl telemetry create");
    }
    return agent_name;
}

[[nodiscard]] size_t
resolveMaxBufferedEvents() {
    const size_t max_events = nixl::config::getValueDefaulted<size_t>(
        TELEMETRY_BUFFER_SIZE_VAR, DEFAULT_TELEMETRY_BUFFER_SIZE);
    if (max_events == 0) {
        throw std::invalid_argument("Telemetry buffer size cannot be 0");
    }
    return max_events;
}

[[nodiscard]] std::string
joinMetricNames(const nixl_telemetry_metric_mask_t &mask, bool active) {
    std::string result;
    for (size_t i = 0; i < nixl_telemetry_event_type_count; ++i) {
        if (mask[i] != active) {
            continue;
        }
        const auto name =
            nixlEnumStrings::telemetryEventTypeStr(static_cast<nixl_telemetry_event_type_t>(i));
        if (!result.empty()) {
            result += ',';
        }
        result.append(name.data(), name.size());
    }
    return result;
}

[[nodiscard]] nixl_telemetry_metric_mask_t
resolveEnabledMetrics() {
    nixl_telemetry_metric_mask_t enabled{};

    const auto spec = nixl::config::getValueOptional<std::string>(TELEMETRY_ENABLED_METRICS_VAR);
    const std::set<std::string> tokens =
        spec ? nixl::str::splitStrippedSet(*spec) : std::set<std::string>{};

    // Unset / empty / all-whitespace: export everything (backward compatible).
    if (tokens.empty()) {
        enabled.fill(true);
        return enabled;
    }

    std::set<std::string> unmatched(tokens);
    for (size_t i = 0; i < nixl_telemetry_event_type_count; ++i) {
        const std::string name{
            nixlEnumStrings::telemetryEventTypeStr(static_cast<nixl_telemetry_event_type_t>(i))};
        for (const auto &token : tokens) {
            if (fnmatch(token.c_str(), name.c_str(), 0) == 0) {
                enabled[i] = true;
                unmatched.erase(token);
            }
        }
    }

    for (const auto &token : unmatched) {
        NIXL_WARN << "Ignoring " << TELEMETRY_ENABLED_METRICS_VAR << " entry '" << token
                  << "': no telemetry metric matches";
    }

    NIXL_DEBUG << "Telemetry metric activation (" << TELEMETRY_ENABLED_METRICS_VAR << "): active=["
               << joinMetricNames(enabled, true) << "] inactive=["
               << joinMetricNames(enabled, false) << "]";

    return enabled;
}

} // namespace

[[nodiscard]] std::unique_ptr<nixlTelemetryExporter>
nixlTelemetry::makeExporter(const std::string &exporter_name) const {
    auto &plugin_manager = nixlPluginManager::getInstance();
    std::shared_ptr<const nixlTelemetryPluginHandle> plugin_handle =
        plugin_manager.loadTelemetryPlugin(exporter_name);

    if (plugin_handle == nullptr) {
        throw std::runtime_error("Failed to load telemetry plugin: " + exporter_name);
    }

    const nixlTelemetryExporterInitParams init_params{agentName_, maxBufferedEvents_};
    auto exporter = plugin_handle->createExporter(init_params);
    if (!exporter) {
        throw std::runtime_error("Failed to create telemetry exporter: " + exporter_name);
    }

    NIXL_DEBUG << "NIXL telemetry is enabled with exporter: " << exporter_name;
    return exporter;
}

void
nixlTelemetry::startExportTask() {
    const auto run_interval =
        nixl::config::getValueDefaulted(TELEMETRY_RUN_INTERVAL_VAR, DEFAULT_TELEMETRY_RUN_INTERVAL);

    writeTask_.callback_ = [this]() { return flushPendingEvents(); };
    writeTask_.interval_ = run_interval;
    writeTask_.enabled_ = true;
    registerPeriodicTask(writeTask_);
}

bool
nixlTelemetry::flushPendingEvents() {
    auto pending = stagingQueue_.takePending();

    for (auto &event : pending) {
        // Deactivated metrics never enter the queue (filtered at the source), so
        // everything here is already exportable.
        exporter_->exportEvent(event);
    }

    exportDroppedEvents();

    return true;
}

void
nixlTelemetry::exportDroppedEvents() {
    // Atomically take and reset the drops accumulated since the last flush and
    // emit them as a synthetic AGENT_TELEMETRY_EVENTS_DROPPED event. Bypassing the
    // staging queue, it can never itself be staging-dropped; emitting only a
    // positive count keeps the no-overflow path event-free (preserving
    // exact-count contracts).
    const uint64_t dropped = stagingQueue_.exchangeDropped();
    if (dropped > 0 &&
        isMetricEnabled(nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED)) {
        exporter_->exportEvent(
            {nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED, dropped});
    }
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
nixlTelemetry::updateData(nixl_telemetry_event_type_t event_type, uint64_t value) {
    // Skip deactivated metrics before taking the lock: no staging cost, and a
    // deactivated metric is not a drop.
    if (!isMetricEnabled(event_type)) {
        return;
    }
    // agent can be multi-threaded; the queue performs its own drop accounting.
    stagingQueue_.tryPush({event_type, value});
}

// TODO: the next 4 update* methods may be removable -- addXferStats covers them.
void
nixlTelemetry::updateTxBytes(uint64_t tx_bytes) {
    updateData(nixl_telemetry_event_type_t::AGENT_TX_BYTES, tx_bytes);
}

void
nixlTelemetry::updateRxBytes(uint64_t rx_bytes) {
    updateData(nixl_telemetry_event_type_t::AGENT_RX_BYTES, rx_bytes);
}

void
nixlTelemetry::updateTxRequestsNum(uint32_t tx_requests_num) {
    updateData(nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM, tx_requests_num);
}

void
nixlTelemetry::updateRxRequestsNum(uint32_t rx_requests_num) {
    updateData(nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM, rx_requests_num);
}

void
nixlTelemetry::updateErrorCount(nixl_status_t error_type) {
    NIXL_ASSERT_ALWAYS(static_cast<int>(error_type) < 0)
        << "nixlTelemetry::updateErrorCount expects a negative nixl_status_t error code";
    const auto event_type = nixlTelemetryEventTypeForStatus(error_type);
    updateData(event_type, 1);
}

void
nixlTelemetry::updateMemoryRegistered(uint64_t memory_registered) {
    updateData(nixl_telemetry_event_type_t::AGENT_MEMORY_REGISTERED, memory_registered);
}

void
nixlTelemetry::updateMemoryDeregistered(uint64_t memory_deregistered) {
    updateData(nixl_telemetry_event_type_t::AGENT_MEMORY_DEREGISTERED, memory_deregistered);
}

void
nixlTelemetry::addXferStats(std::chrono::microseconds xfer_time,
                            bool is_write,
                            uint64_t bytes,
                            std::chrono::microseconds post_time) {
    const auto bytes_type = is_write ? nixl_telemetry_event_type_t::AGENT_TX_BYTES :
                                       nixl_telemetry_event_type_t::AGENT_RX_BYTES;
    const auto requests_type = is_write ? nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM :
                                          nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM;

    // Resolve the activation of each candidate once (lock-free), then reuse it for
    // both the capacity check and the appends -- no intermediate buffer/copies.
    const bool post_on = isMetricEnabled(nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME);
    const bool time_on = isMetricEnabled(nixl_telemetry_event_type_t::AGENT_XFER_TIME);
    const bool bytes_on = isMetricEnabled(bytes_type);
    const bool requests_on = isMetricEnabled(requests_type);
    const size_t batch_size = static_cast<size_t>(post_on) + time_on + bytes_on + requests_on;
    if (batch_size == 0) {
        return;
    }

    // Build the activated subset in a fixed-size stack array (post time, transfer
    // time, bytes, request count) and submit it as one all-or-none batch; the
    // queue accepts every event or drops the whole batch.
    std::array<nixlTelemetryEvent, 4> batch;
    size_t count = 0;
    if (post_on) {
        batch[count++] = {nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME,
                          static_cast<uint64_t>(post_time.count())};
    }
    if (time_on) {
        batch[count++] = {nixl_telemetry_event_type_t::AGENT_XFER_TIME,
                          static_cast<uint64_t>(xfer_time.count())};
    }
    if (bytes_on) {
        batch[count++] = {bytes_type, bytes};
    }
    if (requests_on) {
        batch[count++] = {requests_type, 1};
    }

    stagingQueue_.tryPushBatch(batch.data(), count);
}
