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
#ifndef _TELEMETRY_PROMETHEUS_EXPORTER_H
#define _TELEMETRY_PROMETHEUS_EXPORTER_H

#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "nixl_types.h"

#include <string>
#include <memory>
#include <unordered_map>

#include <prometheus/registry.h>
#include <prometheus/exposer.h>
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>

/**
 * @class nixlTelemetryPrometheusExporter
 * @brief Prometheus-based telemetry exporter implementation
 *
 * This class implements the telemetry exporter interface to export
 * telemetry events to a Prometheus-compatible format using prometheus-cpp.
 * It exposes metrics via an HTTP endpoint that can be scraped by Prometheus.
 */
class nixlTelemetryPrometheusExporter : public nixlTelemetryExporter {
public:
    /**
     * @brief Constructor using init params (plugin-compatible)
     * @param init_params Initialization parameters
     */
    explicit nixlTelemetryPrometheusExporter(const nixlTelemetryExporterInitParams &init_params);

    nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

private:
    // Prometheus components
    const bool local_ = false;
    const uint16_t port_;
    const std::string agent_name_;
    const std::string hostname_;
    std::shared_ptr<prometheus::Registry> registry_;
    std::unique_ptr<prometheus::Exposer> exposer_;
    std::string bind_address_;


    // Maps to track created metrics by event name
    std::unordered_map<std::string, prometheus::Counter *> counters_;
    std::unordered_map<std::string, prometheus::Gauge *> gauges_;

    // Helper methods
    void
    initializeMetrics();
    void
    createOrUpdateBackendEvent(const std::string &event_name, uint64_t value);

    template<typename Family>
    void
    registerGauge(const std::string &name,
                  Family &family,
                  const std::map<std::string, std::string> &labels = {}) {
        gauges_[name] = &family.Add(labels);
    }

    void
    registerCounter(const std::string &name, const std::string &help, const std::string &category);

    void
    registerGauge(const std::string &name, const std::string &help, const std::string &category);
};

#endif // _TELEMETRY_PROMETHEUS_EXPORTER_H
