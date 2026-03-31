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
#include "doca_exporter.h"
#include "common/configuration.h"
#include "common/nixl_log.h"

#include <doca_error.h>
#include <cstring>

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
} // namespace

nixlTelemetryDocaExporter::nixlTelemetryDocaExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params),
      local_(nixl::config::getValueDefaulted(docaPrometheusLocalVar, false)),
      port_(nixl::config::getValueDefaulted(docaPrometheusPortVar, docaPrometheusExporterDefaultPort)),
      agent_name_(init_params.agentName),
      hostname_(getHostname()) {
    if (local_) {
        bind_address_ = docaExporterLocalAddress + ":" + std::to_string(port_);
    } else {
        bind_address_ = docaExporterPublicAddress + ":" + std::to_string(port_);
    }

    nixl_status_t status = initializeDoca(init_params);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to initialize DOCA Telemetry exporter";
        return;
    }

    initialized_ = true;
    NIXL_INFO << "DOCA Telemetry exporter initialized on " << bind_address_;
}

nixlTelemetryDocaExporter::~nixlTelemetryDocaExporter() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (source_) {
        doca_telemetry_exporter_source_flush(source_);
        doca_telemetry_exporter_metrics_destroy_context(source_);
        doca_telemetry_exporter_source_destroy(source_);
        source_ = nullptr;
    }

    if (schema_) {
        doca_telemetry_exporter_schema_destroy(schema_);
        schema_ = nullptr;
    }
#pragma GCC diagnostic pop
}

nixl_status_t
nixlTelemetryDocaExporter::initializeDoca(const nixlTelemetryExporterInitParams &params) {
    doca_error_t result;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    setenv("PROMETHEUS_ENDPOINT", bind_address_.c_str(), 1);
    result = doca_telemetry_exporter_schema_init("nixl_telemetry", &schema_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to initialize DOCA schema: " << result;
        return NIXL_ERR_UNKNOWN;
    }

    result = doca_telemetry_exporter_schema_start(schema_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to start DOCA schema: " << result;
        doca_telemetry_exporter_schema_destroy(schema_);
        return NIXL_ERR_UNKNOWN;
    }

    result = doca_telemetry_exporter_source_create(schema_, &source_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create DOCA source: " << result;
        doca_telemetry_exporter_source_destroy(source_);
        doca_telemetry_exporter_schema_destroy(schema_);
        return NIXL_ERR_UNKNOWN;
    }

    doca_telemetry_exporter_source_set_id(source_, params.agentName.c_str());
    doca_telemetry_exporter_source_set_tag(source_, "nixl");

    result = doca_telemetry_exporter_source_start(source_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to start DOCA source: " << result;
        doca_telemetry_exporter_source_destroy(source_);
        doca_telemetry_exporter_schema_destroy(schema_);
        return NIXL_ERR_UNKNOWN;
    }

    result = doca_telemetry_exporter_metrics_create_context(source_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create DOCA metrics context: " << result;
        return NIXL_ERR_UNKNOWN;
    }

    result =
        doca_telemetry_exporter_metrics_add_constant_label(source_, "hostname", hostname_.c_str());
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to add constant label: " << result;
        doca_telemetry_exporter_metrics_destroy_context(source_);
        doca_telemetry_exporter_source_destroy(source_);
        doca_telemetry_exporter_schema_destroy(schema_);
        return NIXL_ERR_UNKNOWN;
    }

    result = doca_telemetry_exporter_metrics_add_constant_label(
        source_, "agent_name", agent_name_.c_str());
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to add constant label: " << result;
        doca_telemetry_exporter_metrics_destroy_context(source_);
        doca_telemetry_exporter_source_destroy(source_);
        doca_telemetry_exporter_schema_destroy(schema_);
        return NIXL_ERR_UNKNOWN;
    }

    const char *label_names[] = {"category"};
    result =
        doca_telemetry_exporter_metrics_add_label_names(source_, label_names, 1, &label_set_id_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create label set: " << result;
        doca_telemetry_exporter_metrics_destroy_context(source_);
        doca_telemetry_exporter_source_destroy(source_);
        doca_telemetry_exporter_schema_destroy(schema_);
        return NIXL_ERR_UNKNOWN;
    }

    doca_telemetry_exporter_metrics_set_flush_interval_ms(source_, 1000);

#pragma GCC diagnostic pop

    return NIXL_SUCCESS;
}

doca_error_t
nixlTelemetryDocaExporter::registerCounter(const nixlTelemetryEvent &event,
                                           const char *label_values[]) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return doca_telemetry_exporter_metrics_add_counter(
        source_, event.timestampUs_, event.eventName_, event.value_, label_set_id_, label_values);
#pragma GCC diagnostic pop
}

doca_error_t
nixlTelemetryDocaExporter::registerGauge(const nixlTelemetryEvent &event,
                                         const char *label_values[]) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return doca_telemetry_exporter_metrics_add_gauge(
        source_, event.timestampUs_, event.eventName_, event.value_, label_set_id_, label_values);
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
            const char *label_values[] = {docaExporterTransferCategory};
            result = registerCounter(event, label_values);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add counter: " << result;
                return NIXL_ERR_UNKNOWN;
            }
            break;
        }
        case nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND: {
            const char *label_values[] = {docaExporterBackendCategory};
            result = registerCounter(event, label_values);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add counter: " << result;
                return NIXL_ERR_UNKNOWN;
            }
            break;
        }
        case nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE: {
            const char *label_values[] = {docaExporterPerformanceCategory};
            result = registerGauge(event, label_values);
            if (result != DOCA_SUCCESS) {
                NIXL_ERROR << "Failed to add gauge: " << result;
                return NIXL_ERR_UNKNOWN;
            }
            break;
        }
        case nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY: {
            const char *label_values[] = {docaExporterMemoryCategory};
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
