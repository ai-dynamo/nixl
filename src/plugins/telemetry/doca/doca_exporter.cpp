/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "doca_exporter.h"
#include "common/configuration.h"
#include "common/nixl_log.h"
#include "histogram_buckets.h"

#include <doca_telemetry_exporter.h>
#include <doca_error.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
const uint16_t docaPrometheusExporterDefaultPort = 9091;

const char docaPrometheusPortVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_PORT";
const char docaPrometheusLocalVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_LOCAL";

const std::string docaExporterLocalAddress = "http://127.0.0.1";
const std::string docaExporterPublicAddress = "http://0.0.0.0";

[[nodiscard]] std::string
getBindAddress() {
    const bool local = nixl::config::getValueDefaulted(docaPrometheusLocalVar, false);
    const uint16_t port =
        nixl::config::getValueDefaulted(docaPrometheusPortVar, docaPrometheusExporterDefaultPort);
    return (local ? docaExporterLocalAddress : docaExporterPublicAddress) + ":" +
        std::to_string(port);
}

[[nodiscard]] std::string
getHostname() {
    std::array<char, HOST_NAME_MAX + 1> hostname{};
    if (gethostname(hostname.data(), hostname.size()) == 0) {
        hostname.back() = '\0';
        return std::string(hostname.data());
    }
    return "unknown";
}

[[nodiscard]] uint64_t
docaTimestamp() noexcept {
    uint64_t ts = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    doca_telemetry_exporter_get_timestamp(&ts);
#pragma GCC diagnostic pop
    return ts;
}

std::mutex g_ctx_mutex;
std::weak_ptr<DocaSharedContext> g_ctx_weak;
std::mutex g_metrics_mutex;
} // namespace

/**
 * @brief Process-wide shared DOCA context
 *
 * DOCA only supports one metrics context per process, so all agents share
 * this context. The underlying CLX Metrics API is not thread-safe, so all
 * metric recording calls (metrics_add_counter / metrics_add_gauge) are
 * serialised by a dedicated mutex (g_metrics_mutex).
 */
struct DocaSharedContext {
    doca_telemetry_exporter_schema *schema = nullptr;
    doca_telemetry_exporter_source *source = nullptr;
    doca_telemetry_exporter_label_set_id_t label_set_id = 0;
    doca_telemetry_exporter_label_set_id_t error_label_set_id = 0;
    bool source_started = false;
    bool metrics_context_created = false;

    // Base histograms are templates created once on the shared source; observing
    // against one with concrete label values yields a concrete histogram id. The
    // general metrics flush does not push histograms, so track those concrete ids
    // to flush them explicitly.
    std::unordered_map<nixl_telemetry_event_type_t, doca_telemetry_exporter_base_histogram_id_t>
        base_histograms;
    std::unordered_set<doca_telemetry_exporter_histogram_id_t> histogram_ids;

    explicit DocaSharedContext(const std::string &bind_address);
    ~DocaSharedContext();

    DocaSharedContext(const DocaSharedContext &) = delete;
    DocaSharedContext &
    operator=(const DocaSharedContext &) = delete;

private:
    void
    cleanup();
};

DocaSharedContext::DocaSharedContext(const std::string &bind_address) {
    const std::string hostname = getHostname();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    try {
        // DOCA reads its HTTP bind address from this env var. setenv is not
        // thread-safe per POSIX, but the caller holds g_ctx_mutex and this runs
        // only once during first-agent init (before heavy threading).
        setenv("PROMETHEUS_ENDPOINT", bind_address.c_str(), 1);

        doca_error_t result = doca_telemetry_exporter_schema_init("nixl_telemetry", &schema);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Failed to initialize DOCA schema");
        }

        result = doca_telemetry_exporter_schema_start(schema);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Failed to start DOCA schema");
        }

        result = doca_telemetry_exporter_source_create(schema, &source);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Failed to create DOCA source");
        }

        doca_telemetry_exporter_source_set_id(source, "nixl");
        doca_telemetry_exporter_source_set_tag(source, "nixl");

        result = doca_telemetry_exporter_source_start(source);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Failed to start DOCA source");
        }
        source_started = true;

        result = doca_telemetry_exporter_metrics_create_context(source);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Failed to create DOCA metrics context");
        }
        metrics_context_created = true;

        result = doca_telemetry_exporter_metrics_add_constant_label(
            source, "hostname", hostname.c_str());
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Failed to add DOCA constant label");
        }

        const char *label_names[] = {"agent_name"};
        result =
            doca_telemetry_exporter_metrics_add_label_names(source, label_names, 1, &label_set_id);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Failed to create DOCA label set");
        }

        const char *error_label_names[] = {"agent_name", "status"};
        result = doca_telemetry_exporter_metrics_add_label_names(
            source, error_label_names, 2, &error_label_set_id);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Failed to create DOCA error label set");
        }

        doca_telemetry_exporter_metrics_set_flush_interval_ms(source, 1000);

        const std::vector<double> histogram_buckets = nixl::telemetry::resolveHistogramBucketsUs();
        for (const auto event_type : telemetry_metric_event_types) {
            const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(event_type);
            if (descriptor.histogramName == nullptr) {
                continue;
            }
            doca_telemetry_exporter_base_histogram_id_t base_id = 0;
            result = doca_telemetry_exporter_metrics_add_base_histogram(source,
                                                                        descriptor.histogramName,
                                                                        histogram_buckets.data(),
                                                                        histogram_buckets.size(),
                                                                        label_set_id,
                                                                        1000,
                                                                        &base_id);
            if (result != DOCA_SUCCESS) {
                throw std::runtime_error("Failed to create DOCA base histogram");
            }
            base_histograms.emplace(event_type, base_id);
        }
    }
    catch (...) {
        cleanup();
        throw;
    }

#pragma GCC diagnostic pop
}

DocaSharedContext::~DocaSharedContext() {
    cleanup();
}

void
DocaSharedContext::cleanup() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (source) {
        if (source_started) {
            doca_telemetry_exporter_source_flush(source);
        }
        if (metrics_context_created) {
            doca_telemetry_exporter_metrics_destroy_context(source);
        }
        doca_telemetry_exporter_source_destroy(source);
    }
    if (schema) {
        doca_telemetry_exporter_schema_destroy(schema);
    }
#pragma GCC diagnostic pop
}

nixlTelemetryDocaExporter::nixlTelemetryDocaExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params),
      agent_name_(init_params.agentName) {
    const std::string bind_address = getBindAddress();

    const std::lock_guard lock(g_ctx_mutex);
    ctx_ = g_ctx_weak.lock();
    if (!ctx_) {
        ctx_ = std::make_shared<DocaSharedContext>(bind_address);
        g_ctx_weak = ctx_;
        NIXL_INFO << "DOCA Telemetry exporter initialized on " << bind_address;
    } else {
        NIXL_INFO << "DOCA Telemetry exporter for agent '" << agent_name_
                  << "' sharing existing server on " << bind_address;
    }
}

nixlTelemetryDocaExporter::~nixlTelemetryDocaExporter() {
    const std::lock_guard lock(g_ctx_mutex);
    ctx_.reset();
}

doca_error_t
nixlTelemetryDocaExporter::appendCounterSample(const nixlTelemetryEvent &event,
                                               const char *counter_name,
                                               const char *label_values[]) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    // Counter events carry a per-operation delta; increment so the exported
    // counter is a monotonic cumulative total (add_counter would instead push
    // each delta as an absolute value, yielding a non-monotonic series).
    return doca_telemetry_exporter_metrics_add_counter_increment(ctx_->source,
                                                                 docaTimestamp(),
                                                                 counter_name,
                                                                 event.value_,
                                                                 ctx_->label_set_id,
                                                                 label_values);
#pragma GCC diagnostic pop
}

doca_error_t
nixlTelemetryDocaExporter::appendErrorCounterSample(const nixlTelemetryEvent &event,
                                                    const char *status) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    const char *label_values[] = {agent_name_.c_str(), status};
    return doca_telemetry_exporter_metrics_add_counter_increment(ctx_->source,
                                                                 docaTimestamp(),
                                                                 "agent_errors_total",
                                                                 event.value_,
                                                                 ctx_->error_label_set_id,
                                                                 label_values);
#pragma GCC diagnostic pop
}

doca_error_t
nixlTelemetryDocaExporter::appendGaugeSample(const nixlTelemetryEvent &event,
                                             const char *metric_name,
                                             const char *label_values[]) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return doca_telemetry_exporter_metrics_add_gauge(
        ctx_->source, docaTimestamp(), metric_name, event.value_, ctx_->label_set_id, label_values);
#pragma GCC diagnostic pop
}

doca_error_t
nixlTelemetryDocaExporter::appendHistogramSample(const nixlTelemetryEvent &event,
                                                 const char *label_values[]) {
    const auto base = ctx_->base_histograms.find(event.eventType_);
    if (base == ctx_->base_histograms.end()) {
        return DOCA_SUCCESS;
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    doca_telemetry_exporter_histogram_id_t histogram_id = 0;
    const doca_error_t result = doca_telemetry_exporter_metrics_base_histogram_observe(
        ctx_->source, base->second, label_values, static_cast<double>(event.value_), &histogram_id);
    if (result == DOCA_SUCCESS) {
        ctx_->histogram_ids.insert(histogram_id);
    }
    return result;
#pragma GCC diagnostic pop
}

nixl_status_t
nixlTelemetryDocaExporter::exportEvent(const nixlTelemetryEvent &event) {
    try {
        const std::lock_guard lock(g_metrics_mutex);
        const char *label_values[] = {agent_name_.c_str()};

        if (const char *const status =
                nixlEnumStrings::telemetryErrorStatusLabel(event.eventType_)) {
            const auto result = appendErrorCounterSample(event, status);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add error counter: " << result;
                return NIXL_ERR_UNKNOWN;
            }
            return NIXL_SUCCESS;
        }

        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(event.eventType_);

        // Idempotent gauge first, non-idempotent counter increment last.
        doca_error_t gauge_result = DOCA_SUCCESS;
        if (descriptor.gaugeName != nullptr) {
            gauge_result = appendGaugeSample(event, descriptor.gaugeName, label_values);
            if (gauge_result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add gauge: " << gauge_result;
            }
        }

        doca_error_t histogram_result = DOCA_SUCCESS;
        if (descriptor.histogramName != nullptr) {
            histogram_result = appendHistogramSample(event, label_values);
            if (histogram_result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to observe histogram: " << histogram_result;
            }
        }

        doca_error_t counter_result = DOCA_SUCCESS;
        if (descriptor.counterName != nullptr) {
            counter_result = appendCounterSample(event, descriptor.counterName, label_values);
            if (counter_result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add counter: " << counter_result;
            }
        }

        if (gauge_result != DOCA_SUCCESS || counter_result != DOCA_SUCCESS ||
            histogram_result != DOCA_SUCCESS) {
            return NIXL_ERR_UNKNOWN;
        }
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to export telemetry event: " << e.what();
        return NIXL_ERR_UNKNOWN;
    }
}

nixl_status_t
nixlTelemetryDocaExporter::flush() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    const std::lock_guard lock(g_metrics_mutex);

    // Observing only accumulates a histogram in the metrics-API cache; it must be
    // snapshotted into the export buffer via histogram_flush before the general
    // metrics flush transmits the buffer. Order matters: snapshot histograms
    // first, then flush everything (counters/gauges + histogram snapshots) out.
    // A single histogram failure must not skip the remaining snapshots or the
    // general flush, so aggregate the outcome and report it after both phases.
    bool histogram_flush_failed = false;
    for (const auto histogram_id : ctx_->histogram_ids) {
        const auto histogram_result = doca_telemetry_exporter_metrics_histogram_flush(
            ctx_->source, histogram_id, DOCA_TELEMETRY_EXPORTER_HISTOGRAM_TIMESTAMP_UPDATE, false);
        if (histogram_result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to flush DOCA histogram: " << histogram_result;
            histogram_flush_failed = true;
        }
    }

    const auto result = doca_telemetry_exporter_metrics_flush(ctx_->source);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to flush DOCA metrics: " << result;
        return NIXL_ERR_UNKNOWN;
    }
    return histogram_flush_failed ? NIXL_ERR_UNKNOWN : NIXL_SUCCESS;
#pragma GCC diagnostic pop
}
