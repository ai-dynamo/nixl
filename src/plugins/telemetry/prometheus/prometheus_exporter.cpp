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
#include "prometheus_exporter.h"
#include "common/nixl_log.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>

namespace {
const uint16_t prometheusExporterDefaultPort = 9090;

const char prometheusPortVar[] = "NIXL_TELEMETRY_PROMETHEUS_PORT";
const char prometheusLocalVar[] = "NIXL_TELEMETRY_PROMETHEUS_LOCAL";

const std::string prometheusExporterTransferCategory = "NIXL_TELEMETRY_TRANSFER";
const std::string prometheusExporterPerformanceCategory = "NIXL_TELEMETRY_PERFORMANCE";
const std::string prometheusExporterMemoryCategory = "NIXL_TELEMETRY_MEMORY";
const std::string prometheusExporterBackendCategory = "NIXL_TELEMETRY_BACKEND";
const std::string prometheusExporterLocalAddress = "127.0.0.1";
const std::string prometheusExporterPublicAddress = "0.0.0.0";

uint16_t
getPort() {
    auto port_str = std::getenv(prometheusPortVar);
    if (!port_str) {
        return prometheusExporterDefaultPort;
    }

    try {
        int port = std::stoi(port_str);
        if (port < 1 || port > 65535) {
            throw std::out_of_range("Port must be between 1-65535");
        }
        return port;
    }
    catch (const std::exception &e) {
        NIXL_WARN << "Invalid port '" << port_str
                  << "', expected numeric port between 1-65535. Using default: "
                  << prometheusExporterDefaultPort;
        return prometheusExporterDefaultPort;
    }
}

bool
getLocal() {
    auto local_str = std::getenv(prometheusLocalVar);
    return local_str &&
        (!strcasecmp(local_str, "y") || !strcasecmp(local_str, "1") ||
         !strcasecmp(local_str, "yes"));
}

std::string
getHostname() {
    char hostname[HOST_NAME_MAX + 1];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[HOST_NAME_MAX] = '\0'; // Ensure null-termination
        return std::string(hostname);
    }
    return "unknown";
}
} // namespace

nixlTelemetryPrometheusExporter::nixlTelemetryPrometheusExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params),
      local_(getLocal()),
      port_(getPort()),
      agent_name_(init_params.agentName),
      hostname_(getHostname()),
      registry_(std::make_shared<prometheus::Registry>()) {
    if (local_) {
        bind_address_ = prometheusExporterLocalAddress + ":" + std::to_string(port_);
    } else {
        bind_address_ = prometheusExporterPublicAddress + ":" + std::to_string(port_);
    }

    exposer_ = std::make_unique<prometheus::Exposer>(bind_address_);
    exposer_->RegisterCollectable(registry_);

    initializeMetrics();
    NIXL_INFO << "Prometheus exporter initialized on " << bind_address_;
}

// To make access cheaper we are creating static metrics with the labels already set
// Events are defined in the telemetry.cpp file
void
nixlTelemetryPrometheusExporter::initializeMetrics() {
    registerCounter(
        "agent_tx_bytes", "Number of bytes sent by the agent", prometheusExporterTransferCategory);
    registerCounter("agent_rx_bytes",
                    "Number of bytes received by the agent",
                    prometheusExporterTransferCategory);
    registerCounter("agent_tx_requests_num",
                    "Number of requests sent by the agent",
                    prometheusExporterTransferCategory);
    registerCounter("agent_rx_requests_num",
                    "Number of requests received by the agent",
                    prometheusExporterTransferCategory);

    registerGauge("agent_xfer_time",
                  "Start to Complete (per request)",
                  prometheusExporterPerformanceCategory);
    registerGauge("agent_xfer_post_time",
                  "Start to posting to Back-End (per request)",
                  prometheusExporterPerformanceCategory);
    registerGauge("agent_memory_registered", "Memory registered", prometheusExporterMemoryCategory);
    registerGauge(
        "agent_memory_deregistered", "Memory deregistered", prometheusExporterMemoryCategory);
}

void
nixlTelemetryPrometheusExporter::registerCounter(const std::string &name,
                                                 const std::string &help,
                                                 const std::string &category) {
    auto &counter = prometheus::BuildCounter().Name(name).Help(help).Register(*registry_);
    counters_[name] = &counter.Add(
        {{"category", category}, {"hostname", hostname_}, {"agent_name", agent_name_}});
}

void
nixlTelemetryPrometheusExporter::registerGauge(const std::string &name,
                                               const std::string &help,
                                               const std::string &category) {
    auto &gauge = prometheus::BuildGauge().Name(name).Help(help).Register(*registry_);
    gauges_[name] =
        &gauge.Add({{"category", category}, {"hostname", hostname_}, {"agent_name", agent_name_}});
}

void
nixlTelemetryPrometheusExporter::createOrUpdateBackendEvent(const std::string &event_name,
                                                            uint64_t value) {
    auto it = counters_.find(event_name);
    if (it != counters_.end()) {
        it->second->Increment(value);
        return;
    }

    registerCounter(event_name, "Backend event", prometheusExporterBackendCategory);
    counters_[event_name]->Increment(value);
}

nixl_status_t
nixlTelemetryPrometheusExporter::exportEvent(const nixlTelemetryEvent &event) {
    try {
        const std::string event_name(event.eventName_);

        switch (event.category_) {
        case nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER: {
            const auto it = counters_.find(event_name);
            if (it != counters_.end()) {
                it->second->Increment(event.value_);
            }
            break;
        }
        case nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE:
        case nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY: {
            auto it = gauges_.find(event_name);
            if (it != gauges_.end()) {
                it->second->Set(static_cast<double>(event.value_));
            }
            break;
        }
        case nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND:
            createOrUpdateBackendEvent(event_name, event.value_);
            break;
        case nixl_telemetry_category_t::NIXL_TELEMETRY_CONNECTION:
        case nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR:
        case nixl_telemetry_category_t::NIXL_TELEMETRY_SYSTEM:
        case nixl_telemetry_category_t::NIXL_TELEMETRY_CUSTOM:
        default:
            break;
        }

        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to export telemetry event: " << e.what();
        return NIXL_ERR_UNKNOWN;
    }
}
