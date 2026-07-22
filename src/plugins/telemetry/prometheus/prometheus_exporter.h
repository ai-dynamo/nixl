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

#include <cassert>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

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
    struct CounterEntry {
        CounterEntry(prometheus::Family<prometheus::Counter> *family, prometheus::Counter *metric)
            : family(family),
              metric(metric) {}

        CounterEntry(const CounterEntry &) = delete;
        CounterEntry &
        operator=(const CounterEntry &) = delete;
        CounterEntry(CounterEntry &&) = delete;
        CounterEntry &
        operator=(CounterEntry &&) = delete;

        ~CounterEntry() noexcept {
            assert(family && metric);
            family->Remove(metric);
        }

        prometheus::Family<prometheus::Counter> *family = nullptr;
        prometheus::Counter *metric = nullptr;
    };

    struct GaugeEntry {
        GaugeEntry(prometheus::Family<prometheus::Gauge> *family, prometheus::Gauge *metric)
            : family(family),
              metric(metric) {}

        GaugeEntry(const GaugeEntry &) = delete;
        GaugeEntry &
        operator=(const GaugeEntry &) = delete;
        GaugeEntry(GaugeEntry &&) = delete;
        GaugeEntry &
        operator=(GaugeEntry &&) = delete;

        ~GaugeEntry() noexcept {
            assert(family && metric);
            family->Remove(metric);
        }

        prometheus::Family<prometheus::Gauge> *family = nullptr;
        prometheus::Gauge *metric = nullptr;
    };

    struct HistogramEntry {
        HistogramEntry(prometheus::Family<prometheus::Histogram> *family,
                       prometheus::Histogram *metric)
            : family(family),
              metric(metric) {}

        HistogramEntry(const HistogramEntry &) = delete;
        HistogramEntry &
        operator=(const HistogramEntry &) = delete;
        HistogramEntry(HistogramEntry &&) = delete;
        HistogramEntry &
        operator=(HistogramEntry &&) = delete;

        ~HistogramEntry() noexcept {
            assert(family && metric);
            family->Remove(metric);
        }

        prometheus::Family<prometheus::Histogram> *family = nullptr;
        prometheus::Histogram *metric = nullptr;
    };

    const std::string agent_name_;
    const std::string hostname_;
    std::shared_ptr<prometheus::Exposer> exposer_;
    std::shared_ptr<prometheus::Registry> registry_;

    std::unordered_map<nixl_telemetry_event_type_t, CounterEntry> counters_;
    std::unordered_map<nixl_telemetry_event_type_t, GaugeEntry> gauges_;
    std::unordered_map<nixl_telemetry_event_type_t, HistogramEntry> histograms_;

    void
    initializeMetrics();

    void
    registerCounter(nixl_telemetry_event_type_t event_type,
                    const std::string &metric_name,
                    const std::string &help);

    void
    registerErrorCounters();

    // event_type is the lookup key exportEvent() uses; metric_name is the
    // exposed Prometheus series name. They differ for last-operation gauges,
    // e.g. the AGENT_TX_BYTES event drives a gauge published as
    // "agent_tx_last_bytes".
    void
    registerGauge(nixl_telemetry_event_type_t event_type,
                  const std::string &metric_name,
                  const std::string &help);

    void
    registerHistogram(nixl_telemetry_event_type_t event_type,
                      const std::string &metric_name,
                      const std::string &help,
                      const std::vector<double> &buckets);
};

#endif // NIXL_SRC_PLUGINS_TELEMETRY_PROMETHEUS_EXPORTER_H
