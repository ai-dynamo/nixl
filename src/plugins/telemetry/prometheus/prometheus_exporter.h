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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_EXPORTER_H
#define NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_EXPORTER_H

#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "nixl_types.h"

#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

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
 *
 * All agents in the same process share a single Exposer (HTTP server) and
 * Registry. Each agent adds its own metric instances distinguished by the
 * agent_name label. The shared resources are created by the first agent
 * and destroyed when the last agent's exporter is released.
 */
class nixlTelemetryPrometheusExporter : public nixlTelemetryExporter {
public:
    explicit nixlTelemetryPrometheusExporter(const nixlTelemetryExporterInitParams &init_params);
    ~nixlTelemetryPrometheusExporter() override;

    nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

private:
    static std::mutex s_mutex_;
    static std::weak_ptr<prometheus::Exposer> s_exposer_weak_;
    static std::weak_ptr<prometheus::Registry> s_registry_weak_;
    static std::string s_bind_address_;
    static std::unordered_set<std::string> s_agent_names_;

    const std::string agent_name_;
    const std::string hostname_;
    std::shared_ptr<prometheus::Exposer> exposer_;
    std::shared_ptr<prometheus::Registry> registry_;
    std::string bind_address_;

    struct CounterEntry {
        prometheus::Family<prometheus::Counter> *family;
        prometheus::Counter *metric;
    };

    struct GaugeEntry {
        prometheus::Family<prometheus::Gauge> *family;
        prometheus::Gauge *metric;
    };

    std::unordered_map<std::string, CounterEntry> counters_;
    std::unordered_map<std::string, GaugeEntry> gauges_;

    void
    initializeMetrics();

    void
    registerCounter(const std::string &name, const std::string &help, const std::string &category);

    void
    registerGauge(const std::string &name, const std::string &help, const std::string &category);
};

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_EXPORTER_H
