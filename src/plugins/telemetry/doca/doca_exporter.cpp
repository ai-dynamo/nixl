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

#include <doca_error.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

namespace {
const uint16_t docaPrometheusExporterDefaultPort = 9091;

const char docaPrometheusPortVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_PORT";
const char docaPrometheusLocalVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_LOCAL";

const char docaExporterTransferCategory[] = "NIXL_TELEMETRY_TRANSFER";
const char docaExporterPerformanceCategory[] = "NIXL_TELEMETRY_PERFORMANCE";
const char docaExporterMemoryCategory[] = "NIXL_TELEMETRY_MEMORY";
const char docaExporterBackendCategory[] = "NIXL_TELEMETRY_BACKEND";
const std::string docaExporterLocalAddress = "http://127.0.0.1";
const std::string docaExporterPublicAddress = "http://0.0.0.0";

std::string
getHostname() {
    char hostname[HOST_NAME_MAX + 1];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[HOST_NAME_MAX] = '\0';
        return std::string(hostname);
    }
    return "unknown";
}

std::mutex g_ctx_mutex;
std::weak_ptr<DocaSharedContext> g_ctx_weak;
} // namespace

DocaSharedContext::~DocaSharedContext() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (source) {
        if (source_started) doca_telemetry_exporter_source_flush(source);
        if (metrics_context_created) doca_telemetry_exporter_metrics_destroy_context(source);
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
      local_(nixl::config::getValueDefaulted(docaPrometheusLocalVar, false)),
      port_(nixl::config::getValueDefaulted(docaPrometheusPortVar,
                                            docaPrometheusExporterDefaultPort)),
      agent_name_(init_params.agentName),
      hostname_(getHostname()) {
    std::string bind_address = (local_ ? docaExporterLocalAddress : docaExporterPublicAddress) +
        ":" + std::to_string(port_);

    nixl_status_t status = initializeDoca(bind_address);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to initialize DOCA Telemetry exporter");
    }

    initialized_ = true;
}

nixlTelemetryDocaExporter::~nixlTelemetryDocaExporter() {
    const std::lock_guard lock(g_ctx_mutex);
    ctx_.reset();
}

nixl_status_t
nixlTelemetryDocaExporter::initializeDoca(const std::string &bind_address) {
    doca_error_t result;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    const std::lock_guard lock(g_ctx_mutex);
    ctx_ = g_ctx_weak.lock();
    if (!ctx_) {
        auto new_ctx = std::make_shared<DocaSharedContext>();

        // DOCA reads its HTTP bind address from this env var. setenv is not
        // thread-safe per POSIX, but g_ctx_mutex serialises all callers and
        // this runs only once during first-agent init (before heavy threading).
        setenv("PROMETHEUS_ENDPOINT", bind_address.c_str(), 1);

        result = doca_telemetry_exporter_schema_init("nixl_telemetry", &new_ctx->schema);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to initialize DOCA schema: " << result;
            return NIXL_ERR_UNKNOWN;
        }

        result = doca_telemetry_exporter_schema_start(new_ctx->schema);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to start DOCA schema: " << result;
            return NIXL_ERR_UNKNOWN;
        }

        result = doca_telemetry_exporter_source_create(new_ctx->schema, &new_ctx->source);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to create DOCA source: " << result;
            return NIXL_ERR_UNKNOWN;
        }

        doca_telemetry_exporter_source_set_id(new_ctx->source, "nixl");
        doca_telemetry_exporter_source_set_tag(new_ctx->source, "nixl");

        result = doca_telemetry_exporter_source_start(new_ctx->source);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to start DOCA source: " << result;
            return NIXL_ERR_UNKNOWN;
        }
        new_ctx->source_started = true;

        result = doca_telemetry_exporter_metrics_create_context(new_ctx->source);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to create DOCA metrics context: " << result;
            return NIXL_ERR_UNKNOWN;
        }
        new_ctx->metrics_context_created = true;

        result = doca_telemetry_exporter_metrics_add_constant_label(
            new_ctx->source, "hostname", hostname_.c_str());
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to add constant label: " << result;
            return NIXL_ERR_UNKNOWN;
        }

        const char *label_names[] = {"category", "agent_name"};
        result = doca_telemetry_exporter_metrics_add_label_names(
            new_ctx->source, label_names, 2, &new_ctx->label_set_id);
        if (result != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to create label set: " << result;
            return NIXL_ERR_UNKNOWN;
        }

        doca_telemetry_exporter_metrics_set_flush_interval_ms(new_ctx->source, 1000);

        ctx_ = new_ctx;
        g_ctx_weak = ctx_;
        NIXL_INFO << "DOCA Telemetry exporter initialized on " << bind_address;
    } else {
        NIXL_INFO << "DOCA Telemetry exporter for agent '" << agent_name_
                  << "' sharing existing server on " << bind_address;
    }

#pragma GCC diagnostic pop

    return NIXL_SUCCESS;
}

doca_error_t
nixlTelemetryDocaExporter::registerCounter(const nixlTelemetryEvent &event,
                                           const char *label_values[]) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return doca_telemetry_exporter_metrics_add_counter(ctx_->source,
                                                       event.timestampUs_,
                                                       event.eventName_,
                                                       event.value_,
                                                       ctx_->label_set_id,
                                                       label_values);
#pragma GCC diagnostic pop
}

doca_error_t
nixlTelemetryDocaExporter::registerGauge(const nixlTelemetryEvent &event,
                                         const char *label_values[]) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return doca_telemetry_exporter_metrics_add_gauge(ctx_->source,
                                                     event.timestampUs_,
                                                     event.eventName_,
                                                     event.value_,
                                                     ctx_->label_set_id,
                                                     label_values);
#pragma GCC diagnostic pop
}

nixl_status_t
nixlTelemetryDocaExporter::exportEvent(const nixlTelemetryEvent &event) {
    doca_error_t result;
    if (!initialized_) {
        NIXL_ERROR << "DOCA exporter not initialized";
        return NIXL_ERR_UNKNOWN;
    }

    try {
        switch (event.category_) {
        case nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER: {
            const char *label_values[] = {docaExporterTransferCategory, agent_name_.c_str()};
            result = registerCounter(event, label_values);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add counter: " << result;
                return NIXL_ERR_UNKNOWN;
            }
            break;
        }
        case nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND: {
            const char *label_values[] = {docaExporterBackendCategory, agent_name_.c_str()};
            result = registerCounter(event, label_values);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add counter: " << result;
                return NIXL_ERR_UNKNOWN;
            }
            break;
        }
        case nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE: {
            const char *label_values[] = {docaExporterPerformanceCategory, agent_name_.c_str()};
            result = registerGauge(event, label_values);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add gauge: " << result;
                return NIXL_ERR_UNKNOWN;
            }
            break;
        }
        case nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY: {
            const char *label_values[] = {docaExporterMemoryCategory, agent_name_.c_str()};
            result = registerGauge(event, label_values);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add gauge: " << result;
                return NIXL_ERR_UNKNOWN;
            }
            break;
        }
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
