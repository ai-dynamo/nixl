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

const std::string PROMETHEUS_EXPORTER_DEFAULT_BIND_ADDRESS = "0.0.0.0:9090";

nixlTelemetryPrometheusExporter::nixlTelemetryPrometheusExporter(
    const nixlTelemetryExporterInitParams *init_params)
    : nixlTelemetryExporter(init_params),
      registry_(std::make_shared<prometheus::Registry>()),
      bind_address_(PROMETHEUS_EXPORTER_DEFAULT_BIND_ADDRESS) {

    if (init_params && !init_params->outputPath.empty()) {
        // Validate format: ip_addr:port_num
        std::string path = init_params->outputPath;
        size_t colon_pos = path.find(':');

        if (colon_pos == std::string::npos || colon_pos == 0 || colon_pos == path.length() - 1) {
            NIXL_WARN << "Invalid bind address format '" << path
                      << "', expected 'ip_addr:port_num'. Using default: "
                      << PROMETHEUS_EXPORTER_DEFAULT_BIND_ADDRESS;
        } else {
            std::string ip_addr = path.substr(0, colon_pos);
            std::string port_str = path.substr(colon_pos + 1);

            // Validate port is numeric
            try {
                int port = std::stoi(port_str);
                if (port < 1 || port > 65535) {
                    NIXL_WARN << "Invalid port number " << port
                              << ", must be between 1-65535. Using default: "
                              << PROMETHEUS_EXPORTER_DEFAULT_BIND_ADDRESS;
                } else {
                    bind_address_ = path;
                }
            }
            catch (const std::exception &e) {
                NIXL_WARN << "Invalid port in bind address '" << path
                          << "', expected numeric port. Using default: "
                          << PROMETHEUS_EXPORTER_DEFAULT_BIND_ADDRESS;
            }
        }
    }

    try {
        exposer_ = std::make_unique<prometheus::Exposer>(bind_address_);
        exposer_->RegisterCollectable(registry_);

        initializeMetrics();
        NIXL_INFO << "Prometheus exporter initialized on " << bind_address_;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to initialize Prometheus exporter: " << e.what();
    }
}

nixlTelemetryPrometheusExporter::~nixlTelemetryPrometheusExporter() {
    NIXL_INFO << "Prometheus exporter shutting down";
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

    counters_["agent_tx_bytes"] = &tx_bytes_counter.Add({{"category", "NIXL_TELEMETRY_TRANSFER"}});
    counters_["agent_rx_bytes"] = &rx_bytes_counter.Add({{"category", "NIXL_TELEMETRY_TRANSFER"}});
    counters_["agent_tx_requests_num"] =
        &tx_requests_counter.Add({{"category", "NIXL_TELEMETRY_TRANSFER"}});
    counters_["agent_rx_requests_num"] =
        &rx_requests_counter.Add({{"category", "NIXL_TELEMETRY_TRANSFER"}});

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

    gauges_["agent_xfer_time"] = &xfer_time_gauge.Add({{"category", "NIXL_TELEMETRY_PERFORMANCE"}});
    gauges_["agent_xfer_post_time"] =
        &xfer_post_time_gauge.Add({{"category", "NIXL_TELEMETRY_PERFORMANCE"}});
    gauges_["agent_memory_registered"] =
        &memory_registered_gauge.Add({{"category", "NIXL_TELEMETRY_MEMORY"}});
    gauges_["agent_memory_deregistered"] =
        &memory_deregistered_gauge.Add({{"category", "NIXL_TELEMETRY_MEMORY"}});
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
    counters_[event_name] = &backend_counter.Add({{"category", "NIXL_TELEMETRY_BACKEND"}});
    counters_[event_name]->Increment(value);
}

nixl_status_t
nixlTelemetryPrometheusExporter::exportEvent(const nixlTelemetryEvent &event) {
    try {
        std::string event_name(event.eventName_);

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
