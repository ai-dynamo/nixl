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
#include "prometheus_exporter.h"
#include "common/configuration.h"
#include "common/nixl_log.h"
#include "histogram_buckets.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <unordered_set>

namespace {
const uint16_t prometheusExporterDefaultPort = 9090;

const char prometheusPortVar[] = "NIXL_TELEMETRY_PROMETHEUS_PORT";
const char prometheusLocalVar[] = "NIXL_TELEMETRY_PROMETHEUS_LOCAL";

const std::string prometheusExporterLocalAddress = "127.0.0.1";
const std::string prometheusExporterPublicAddress = "0.0.0.0";

std::string
getHostname() {
    char hostname[HOST_NAME_MAX + 1];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[HOST_NAME_MAX] = '\0'; // Ensure null-termination
        return std::string(hostname);
    }
    return "unknown";
}

std::mutex s_mutex;
std::weak_ptr<prometheus::Exposer> s_exposer_weak;
std::weak_ptr<prometheus::Registry> s_registry_weak;
std::unordered_set<std::string> s_agent_names;
} // namespace

nixlTelemetryPrometheusExporter::nixlTelemetryPrometheusExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params),
      agent_name_(init_params.agentName),
      hostname_(getHostname()) {
    const bool local = nixl::config::getValueDefaulted(prometheusLocalVar, false);
    const uint16_t port =
        nixl::config::getValueDefaulted(prometheusPortVar, prometheusExporterDefaultPort);

    std::string bind_address;
    if (local) {
        bind_address = prometheusExporterLocalAddress + ":" + std::to_string(port);
    } else {
        bind_address = prometheusExporterPublicAddress + ":" + std::to_string(port);
    }

    const std::lock_guard lock(s_mutex);

    if (!s_agent_names.insert(agent_name_).second) {
        throw std::runtime_error("Prometheus exporter: duplicate agent name '" + agent_name_ +
                                 "'; each agent must have a unique name");
    }

    try {
        exposer_ = s_exposer_weak.lock();
        registry_ = s_registry_weak.lock();

        if (!exposer_ || !registry_) {
            registry_.reset();
            exposer_.reset();
            registry_ = std::make_shared<prometheus::Registry>();
            exposer_ = std::make_shared<prometheus::Exposer>(bind_address);
            exposer_->RegisterCollectable(registry_);
            s_exposer_weak = exposer_;
            s_registry_weak = registry_;
            NIXL_INFO << "Prometheus exporter initialized on " << bind_address;
        } else {
            NIXL_INFO << "Prometheus exporter for agent '" << agent_name_
                      << "' sharing existing server";
        }

        initializeMetrics();
    }
    catch (...) {
        counters_.clear();
        gauges_.clear();
        histograms_.clear();
        s_agent_names.erase(agent_name_);
        throw;
    }
}

nixlTelemetryPrometheusExporter::~nixlTelemetryPrometheusExporter() {
    const std::lock_guard lock(s_mutex);
    counters_.clear();
    gauges_.clear();
    histograms_.clear();
    s_agent_names.erase(agent_name_);
    exposer_.reset();
    registry_.reset();
}

// To make access cheaper we are creating static metrics with the labels already set
// Events are defined in the telemetry.cpp file
void
nixlTelemetryPrometheusExporter::initializeMetrics() {
    const std::vector<double> histogram_buckets = nixl::telemetry::resolveHistogramBucketsUs();
    for (const auto event_type : telemetry_metric_event_types) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(event_type);
        if (descriptor.counterName != nullptr) {
            registerCounter(event_type, descriptor.counterName, descriptor.counterHelp);
        }
        if (descriptor.gaugeName != nullptr) {
            registerGauge(event_type, descriptor.gaugeName, descriptor.gaugeHelp);
        }
        if (descriptor.histogramName != nullptr) {
            registerHistogram(
                event_type, descriptor.histogramName, descriptor.histogramHelp, histogram_buckets);
        }
    }
    registerErrorCounters();
}

void
nixlTelemetryPrometheusExporter::registerCounter(const nixl_telemetry_event_type_t event_type,
                                                 const std::string &metric_name,
                                                 const std::string &help) {
    auto &family = prometheus::BuildCounter().Name(metric_name).Help(help).Register(*registry_);
    auto &metric = family.Add({{"hostname", hostname_}, {"agent_name", agent_name_}});
    const auto inserted = counters_.try_emplace(event_type, &family, &metric).second;
    if (!inserted) {
        family.Remove(&metric);
    }
    NIXL_ASSERT(inserted);
}

void
nixlTelemetryPrometheusExporter::registerErrorCounters() {
    auto &family = prometheus::BuildCounter()
                       .Name("agent_errors_total")
                       .Help("Cumulative error count by status")
                       .Register(*registry_);

    for (const auto event_type : telemetry_error_event_types) {
        const char *const status = nixlEnumStrings::telemetryErrorStatusLabel(event_type);
        auto &metric =
            family.Add({{"hostname", hostname_}, {"agent_name", agent_name_}, {"status", status}});
        const auto inserted = counters_.try_emplace(event_type, &family, &metric).second;
        if (!inserted) {
            family.Remove(&metric);
        }
        NIXL_ASSERT(inserted);
    }
}

void
nixlTelemetryPrometheusExporter::registerGauge(const nixl_telemetry_event_type_t event_type,
                                               const std::string &metric_name,
                                               const std::string &help) {
    auto &family = prometheus::BuildGauge().Name(metric_name).Help(help).Register(*registry_);
    auto &metric = family.Add({{"hostname", hostname_}, {"agent_name", agent_name_}});
    const auto inserted = gauges_.try_emplace(event_type, &family, &metric).second;
    if (!inserted) {
        family.Remove(&metric);
    }
    NIXL_ASSERT(inserted);
}

void
nixlTelemetryPrometheusExporter::registerHistogram(const nixl_telemetry_event_type_t event_type,
                                                   const std::string &metric_name,
                                                   const std::string &help,
                                                   const std::vector<double> &buckets) {
    auto &family = prometheus::BuildHistogram().Name(metric_name).Help(help).Register(*registry_);
    auto &metric = family.Add({{"hostname", hostname_}, {"agent_name", agent_name_}},
                              prometheus::Histogram::BucketBoundaries(buckets));
    const auto inserted = histograms_.try_emplace(event_type, &family, &metric).second;
    if (!inserted) {
        family.Remove(&metric);
    }
    NIXL_ASSERT(inserted);
}

nixl_status_t
nixlTelemetryPrometheusExporter::exportEvent(const nixlTelemetryEvent &event) {
    try {
        const auto counter = counters_.find(event.eventType_);
        if (counter != counters_.end()) {
            counter->second.metric->Increment(event.value_);
        }

        const auto gauge = gauges_.find(event.eventType_);
        if (gauge != gauges_.end()) {
            gauge->second.metric->Set(static_cast<double>(event.value_));
        }

        const auto histogram = histograms_.find(event.eventType_);
        if (histogram != histograms_.end()) {
            histogram->second.metric->Observe(static_cast<double>(event.value_));
        }

        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to export telemetry event: " << e.what();
        return NIXL_ERR_UNKNOWN;
    }
}
