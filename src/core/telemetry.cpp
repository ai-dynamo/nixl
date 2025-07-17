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
#include <unistd.h>
#include <cstdlib>

#include "common/nixl_log.h"
#include "telemetry.h"
#include "nixl_types.h"
#include "common/thread_executor.h"

using namespace std::chrono_literals;

#ifdef NIXL_ENABLE_TELEMETRY

std::string
getTelemetryBufferName() {
    std::stringstream ss;
    std::string telemetry_dir = std::getenv("NIXL_TELEMETRY_DIR") ?
        std::getenv("NIXL_TELEMETRY_DIR") : "/tmp";
    ss << telemetry_dir << "/" << TELEMETRY_PREFIX << "." << getpid();
    return ss.str();
}

nixlTelemetry::nixlTelemetry()
    : buffer_(getTelemetryBufferName().c_str(), true, TELEMETRY_VERSION),
      last_requests_index_(0),
      updated_(false),
      enabled_(true) {
    // Check environment variable for runtime disable
    enabled_ = std::getenv("NIXL_ENABLE_TELEMETRY") != nullptr;
    NIXL_WARN << "Telemetry enabled: " << enabled_;
    if (!enabled_) {
        NIXL_INFO << "Telemetry disabled via NIXL_ENABLE_TELEMETRY environment variable";
        return;
    }

    event_ = {};
    // write once at startup
    updated_ = true;

    // Only register periodic task if telemetry is enabled
    if (enabled_) {
        nixl::ThreadExecutor::getInstance().registerPeriodicTask(
            "telemetry", [this]() { writeEvent(); }, 100ms);
    }
}

nixlTelemetry::~nixlTelemetry() {
    if (enabled_) {
        nixl::ThreadExecutor::getInstance().unregisterTask("telemetry");
    }
}

void
nixlTelemetry::writeEvent() {
    if (!enabled_) {
        return;
    }
    if (updated_) {
        event_.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
        if (!buffer_.push(event_)) {
            NIXL_WARN << "Telemetry buffer full, dropping event";
        }
        // make it easy to track last requests times by clearing the array after each write
        memset(&event_.last_requests_time_us, 0, sizeof(event_.last_requests_time_us));
        updated_ = false;
    }
}

void
nixlTelemetry::updateTxBytes(uint64_t tx_bytes) {
    event_.tx_bytes += tx_bytes;
    updated_ = true;
}

void
nixlTelemetry::updateRxBytes(uint64_t rx_bytes) {
    event_.rx_bytes += rx_bytes;
    updated_ = true;
}

void
nixlTelemetry::updateTxRequestsNum(uint32_t tx_requests_num) {
    event_.tx_requests_num += tx_requests_num;
    updated_ = true;
}

void
nixlTelemetry::updateRxRequestsNum(uint32_t rx_requests_num) {
    event_.rx_requests_num += rx_requests_num;
    updated_ = true;
}

void
nixlTelemetry::updateErrorCount(nixl_status_t error_type) {
    int index = std::abs(static_cast<int>(error_type) - 1);
    event_.error_counts[index]++;
    updated_ = true;
}

void
nixlTelemetry::updateMemoryRegistered(int64_t memory_registered) {
    event_.memory_registered += memory_registered;
    updated_ = true;
}

void
nixlTelemetry::addRequestTime(std::chrono::microseconds transaction_time) {
    event_.last_requests_time_us[last_requests_index_] = static_cast<uint32_t>(transaction_time.count());
    last_requests_index_ = (last_requests_index_ + 1) % LAST_REQUESTS_ARRAY_SIZE;
    updated_ = true;
}

// Transaction time tracking methods
void
nixlTelemetry::updateTransactionTime(std::chrono::microseconds transaction_time) {
    event_.total_transaction_time_us += transaction_time.count();
    updated_ = true;
}

void
nixlTelemetry::resetTransactionTimeStats() {
    event_.total_transaction_time_us = 0;
    updated_ = true;
}

// Plugin telemetry methods
void
nixlTelemetry::updatePluginTelemetry(const std::string &plugin_name,
                                     const std::vector<nixlPluginTelemetryMetric> &metrics) {
    if (!enabled_) {
        return;
    }

    if (metrics.empty()) {
        return;
    }

    size_t backend_index = 0;
    std::string plugin_name_str = plugin_name.substr(0, MAX_PLUGIN_NAME_LEN - 1);

    for (backend_index = 0; backend_index < MAX_BACKENDS_PER_EVENT; backend_index++) {
        // find the first empty backend or existing backend
        if (event_.backend_telemetry[backend_index].plugin_name[0] == '\0') {
            strncpy(event_.backend_telemetry[backend_index].plugin_name,
                    plugin_name_str.c_str(),
                    MAX_PLUGIN_NAME_LEN - 1);
            event_.backend_telemetry[backend_index].plugin_name[MAX_PLUGIN_NAME_LEN - 1] = '\0';
            event_.num_backends++;
            break;
        } else if (strncmp(event_.backend_telemetry[backend_index].plugin_name,
                           plugin_name_str.c_str(),
                           MAX_PLUGIN_NAME_LEN - 1) == 0) {
            // update the existing backend
            break;
        }
    }
    if (backend_index == MAX_BACKENDS_PER_EVENT) {
        NIXL_WARN << "Too many backends, skipping plugin telemetry update for plugin: "
                  << plugin_name;
        return;
    }
    auto &plugin_entry = event_.backend_telemetry[backend_index];
    for (auto &metric : metrics) {
        size_t metric_index = 0;
        for (metric_index = 0; metric_index < MAX_METRICS_PER_EVENT; metric_index++) {
            if (strncmp(plugin_entry.metrics[metric_index].name, metric.name, MAX_METRIC_NAME_LEN - 1) == 0) {
                plugin_entry.metrics[metric_index].value = metric.value;
                break;
            }
            if (plugin_entry.metrics[metric_index].name[0] == '\0') {
                strncpy(plugin_entry.metrics[metric_index].name, metric.name, MAX_METRIC_NAME_LEN - 1);
                plugin_entry.metrics[metric_index].name[MAX_METRIC_NAME_LEN - 1] = '\0';
                plugin_entry.metrics[metric_index].value = metric.value;
                plugin_entry.num_metrics++;
                break;
            }
        }
        if (metric_index == MAX_METRICS_PER_EVENT) {
            NIXL_WARN << "Too many metrics, skipping plugin telemetry update for plugin: "
                      << plugin_name;
        }
    }
    updated_ = true;
}

#endif // NIXL_ENABLE_TELEMETRY
