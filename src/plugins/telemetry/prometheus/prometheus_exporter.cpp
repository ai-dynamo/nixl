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

inline const uint16_t prometheusExporterDefaultPort = 9090;
inline const std::string prometheusExporterTransferCategory = "NIXL_TELEMETRY_TRANSFER";
inline const std::string prometheusExporterPerformanceCategory = "NIXL_TELEMETRY_PERFORMANCE";
inline const std::string prometheusExporterMemoryCategory = "NIXL_TELEMETRY_MEMORY";
inline const std::string prometheusExporterBackendCategory = "NIXL_TELEMETRY_BACKEND";
inline const std::string prometheusExporterLocalAddress = "127.0.0.1";
inline const std::string prometheusExporterPublicAddress = "0.0.0.0";

inline constexpr char prometheusPortVar[] = "NIXL_TELEMETRY_PROMETHEUS_PORT";
inline constexpr char prometheusLocalVar[] = "NIXL_TELEMETRY_PROMETHEUS_LOCAL";

nixlTelemetryPrometheusExporter::nixlTelemetryPrometheusExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params),
      registry_(std::make_shared<prometheus::Registry>()) {
    auto port_str = std::getenv(prometheusPortVar);
    if (!port_str) {
        port_ = prometheusExporterDefaultPort;
    } else {
        try {
            const int port = std::stoi(port_str);
            if (port < 1 || port > 65535) {
                NIXL_WARN << "Invalid port number " << port
                          << ", must be between 1-65535. Using default: "
                          << prometheusExporterDefaultPort;
            } else {
                port_ = port;
            }
        }
        catch (const std::exception &e) {
            NIXL_WARN << "Invalid port " << port_str << "', expected numeric port. Using default: "
                      << prometheusExporterDefaultPort;
            port_ = prometheusExporterDefaultPort;
        }
    }

    auto local_str = std::getenv(prometheusLocalVar);
    if (local_str &&
        (!strcasecmp(local_str, "y") || !strcasecmp(local_str, "1") ||
         !strcasecmp(local_str, "yes"))) {
        local_ = true;
    } else {
        local_ = false;
    }

    try {
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
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to initialize Prometheus exporter: " << e.what();
    }
}

// To make access cheaper we are creating static metrics with the labels already set
// Events are defined in the telemetry.cpp file
void
nixlTelemetryPrometheusExporter::initializeMetrics() {
    auto &tx_bytes_counter = prometheus::BuildCounter()
                                 .Name("agent_tx_bytes")
                                 .Help("Number of bytes sent by the agent")
                                 .Register(*registry_);

    auto &rx_bytes_counter = prometheus::BuildCounter()
                                 .Name("agent_rx_bytes")
                                 .Help("Number of bytes received by the agent")
                                 .Register(*registry_);

    auto &tx_requests_counter = prometheus::BuildCounter()
                                    .Name("agent_tx_requests_num")
                                    .Help("Number of requests sent by the agent")
                                    .Register(*registry_);

    auto &rx_requests_counter = prometheus::BuildCounter()
                                    .Name("agent_rx_requests_num")
                                    .Help("Number of requests received by the agent")
                                    .Register(*registry_);

    registerCounter(
        "agent_tx_bytes", tx_bytes_counter, {{"category", prometheusExporterTransferCategory}});
    registerCounter(
        "agent_rx_bytes", rx_bytes_counter, {{"category", prometheusExporterTransferCategory}});
    registerCounter("agent_tx_requests_num",
                    tx_requests_counter,
                    {{"category", prometheusExporterTransferCategory}});
    registerCounter("agent_rx_requests_num",
                    rx_requests_counter,
                    {{"category", prometheusExporterTransferCategory}});

    auto &xfer_time_gauge = prometheus::BuildGauge()
                                .Name("agent_xfer_time")
                                .Help("Start to Complete (per request)")
                                .Register(*registry_);

    auto &xfer_post_time_gauge = prometheus::BuildGauge()
                                     .Name("agent_xfer_post_time")
                                     .Help("Start to posting to Back-End (per request)")
                                     .Register(*registry_);

    auto &memory_registered_gauge = prometheus::BuildGauge()
                                        .Name("agent_memory_registered")
                                        .Help("Memory registered")
                                        .Register(*registry_);

    auto &memory_deregistered_gauge = prometheus::BuildGauge()
                                          .Name("agent_memory_deregistered")
                                          .Help("Memory deregistered")
                                          .Register(*registry_);

    registerGauge(
        "agent_xfer_time", xfer_time_gauge, {{"category", prometheusExporterPerformanceCategory}});
    registerGauge("agent_xfer_post_time",
                  xfer_post_time_gauge,
                  {{"category", prometheusExporterPerformanceCategory}});
    registerGauge("agent_memory_registered",
                  memory_registered_gauge,
                  {{"category", prometheusExporterMemoryCategory}});
    registerGauge("agent_memory_deregistered",
                  memory_deregistered_gauge,
                  {{"category", prometheusExporterMemoryCategory}});
}

void
nixlTelemetryPrometheusExporter::createOrUpdateBackendEvent(const std::string &event_name,
                                                            uint64_t value) {
    auto it = counters_.find(event_name);
    if (it != counters_.end()) {
        it->second->Increment(value);
        return;
    }

    auto &backend_counter =
        prometheus::BuildCounter().Name(event_name).Help("Backend event").Register(*registry_);
    counters_[event_name] = &backend_counter.Add({{"category", prometheusExporterBackendCategory}});
    counters_[event_name]->Increment(value);
}

nixl_status_t
nixlTelemetryPrometheusExporter::exportEvent(const nixlTelemetryEvent &event) {
    try {
        const std::string event_name(event.eventName_);

        switch (event.category_) {
        case nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER: {
            auto it = counters_.find(event_name);
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
