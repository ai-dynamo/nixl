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

#include <doca_telemetry_exporter.h>
#include <doca_error.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

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

[[nodiscard]] constexpr bool
isCounterEvent(nixl_telemetry_event_type_t event_type) noexcept {
    switch (event_type) {
    case nixl_telemetry_event_type_t::AGENT_TX_BYTES:
    case nixl_telemetry_event_type_t::AGENT_RX_BYTES:
    case nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM:
    case nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM:
        return true;
    case nixl_telemetry_event_type_t::AGENT_MEMORY_REGISTERED:
    case nixl_telemetry_event_type_t::AGENT_MEMORY_DEREGISTERED:
    case nixl_telemetry_event_type_t::AGENT_XFER_TIME:
    case nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME:
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_POSTED:
    case nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM:
    case nixl_telemetry_event_type_t::AGENT_ERR_BACKEND:
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_FOUND:
    case nixl_telemetry_event_type_t::AGENT_ERR_MISMATCH:
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_ALLOWED:
    case nixl_telemetry_event_type_t::AGENT_ERR_REPOST_ACTIVE:
    case nixl_telemetry_event_type_t::AGENT_ERR_UNKNOWN:
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_SUPPORTED:
    case nixl_telemetry_event_type_t::AGENT_ERR_REMOTE_DISCONNECT:
    case nixl_telemetry_event_type_t::AGENT_ERR_CANCELED:
    case nixl_telemetry_event_type_t::AGENT_ERR_NO_TELEMETRY:
        return false;
    }
    return false;
}

// Gauge series name for an event, or nullptr if the event has no gauge.
[[nodiscard]] constexpr const char *
gaugeMetricName(nixl_telemetry_event_type_t event_type) noexcept {
    switch (event_type) {
    case nixl_telemetry_event_type_t::AGENT_TX_BYTES:
        return "agent_tx_last_bytes";
    case nixl_telemetry_event_type_t::AGENT_RX_BYTES:
        return "agent_rx_last_bytes";
    case nixl_telemetry_event_type_t::AGENT_MEMORY_REGISTERED:
        return "agent_memory_registered_last_bytes";
    case nixl_telemetry_event_type_t::AGENT_MEMORY_DEREGISTERED:
        return "agent_memory_deregistered_last_bytes";
    case nixl_telemetry_event_type_t::AGENT_XFER_TIME:
        return "agent_xfer_time";
    case nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME:
        return "agent_xfer_post_time";
    case nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM:
    case nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM:
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_POSTED:
    case nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM:
    case nixl_telemetry_event_type_t::AGENT_ERR_BACKEND:
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_FOUND:
    case nixl_telemetry_event_type_t::AGENT_ERR_MISMATCH:
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_ALLOWED:
    case nixl_telemetry_event_type_t::AGENT_ERR_REPOST_ACTIVE:
    case nixl_telemetry_event_type_t::AGENT_ERR_UNKNOWN:
    case nixl_telemetry_event_type_t::AGENT_ERR_NOT_SUPPORTED:
    case nixl_telemetry_event_type_t::AGENT_ERR_REMOTE_DISCONNECT:
    case nixl_telemetry_event_type_t::AGENT_ERR_CANCELED:
    case nixl_telemetry_event_type_t::AGENT_ERR_NO_TELEMETRY:
        return nullptr;
    }
    return nullptr;
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
                                               const char *label_values[]) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    const std::string event_name(nixlEnumStrings::telemetryEventTypeStr(event.eventType_));
    // Counter events carry a per-operation delta; increment so the exported
    // counter is a monotonic cumulative total (add_counter would instead push
    // each delta as an absolute value, yielding a non-monotonic series).
    return doca_telemetry_exporter_metrics_add_counter_increment(ctx_->source,
                                                                 docaTimestamp(),
                                                                 event_name.c_str(),
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

        // Idempotent gauge first, non-idempotent counter increment last.
        if (const char *const gauge_name = gaugeMetricName(event.eventType_)) {
            const auto result = appendGaugeSample(event, gauge_name, label_values);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add gauge: " << result;
                return NIXL_ERR_UNKNOWN;
            }
        }

        if (isCounterEvent(event.eventType_)) {
            const auto result = appendCounterSample(event, label_values);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add counter: " << result;
                return NIXL_ERR_UNKNOWN;
            }
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
    const auto result = doca_telemetry_exporter_metrics_flush(ctx_->source);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to flush DOCA metrics: " << result;
        return NIXL_ERR_UNKNOWN;
    }
    return NIXL_SUCCESS;
#pragma GCC diagnostic pop
}
