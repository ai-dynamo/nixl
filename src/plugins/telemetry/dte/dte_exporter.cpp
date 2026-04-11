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

#include "dte_exporter.h"
#include "common/nixl_log.h"
#include "common/util.h"
#include "common/configuration.h"
#include <absl/strings/str_format.h>

#include <limits.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <string_view>

// DOCA Telemetry Exporter API Documentation:
// https://docs.nvidia.com/doca/sdk/doca-telemetry-exporter/index.html

namespace {

constexpr uint32_t dte_flush_interval_ms = 1000U;

const char *dte_dynamic_labels[] = {"category"};

constexpr const char *const dte_grpc_bug_workaround_var =
    "NIXL_TELEMETRY_DTE_ENABLE_GRPC_BUG_WORKAROUND";
constexpr const char *const dte_data_root_var = "NIXL_TELEMETRY_DTE_DATA_ROOT";
constexpr const char *const dte_prometheus_ep_enabled_var =
    "NIXL_TELEMETRY_DTE_PROMETHEUS_EP_ENABLED";
constexpr const char *const dte_prometheus_ep_address_var =
    "NIXL_TELEMETRY_DTE_PROMETHEUS_EP_ADDRESS";
constexpr const char *const dte_prometheus_ep_port_var = "NIXL_TELEMETRY_DTE_PROMETHEUS_EP_PORT";
constexpr const char *const dte_ipc_enabled_var = "NIXL_TELEMETRY_DTE_IPC_ENABLED";
constexpr const char *const dte_ipc_sockets_dir_var = "NIXL_TELEMETRY_DTE_IPC_SOCKETS_DIR";
constexpr const char *const dte_ipc_reconnect_time_var = "NIXL_TELEMETRY_DTE_IPC_RECONNECT_TIME";
constexpr const char *const dte_ipc_reconnect_tries_var = "NIXL_TELEMETRY_DTE_IPC_RECONNECT_TRIES";
constexpr const char *const dte_ipc_socket_timeout_var = "NIXL_TELEMETRY_DTE_IPC_SOCKET_TIMEOUT";
constexpr const char *const dte_file_enabled_var = "NIXL_TELEMETRY_DTE_FILE_ENABLED";
constexpr const char *const dte_file_max_size_var = "NIXL_TELEMETRY_DTE_FILE_MAX_SIZE";
constexpr const char *const dte_file_max_age_var = "NIXL_TELEMETRY_DTE_FILE_MAX_AGE";
constexpr const char *const dte_otlp_enabled_var = "NIXL_TELEMETRY_DTE_OTLP_ENABLED";
constexpr const char *const dte_otlp_address_var = "NIXL_TELEMETRY_DTE_OTLP_ADDRESS";
constexpr const char *const dte_otlp_port_var = "NIXL_TELEMETRY_DTE_OTLP_PORT";

constexpr uint16_t dte_default_otlp_port = 9502;
constexpr std::string_view dte_otlp_write_endpoint = "/v1/metrics";
constexpr std::string_view dte_default_prometheus_ep_endpoint_address = "0.0.0.0";
constexpr uint16_t dte_default_prometheus_ep_endpoint_port = 9101;

// TODO: solve backward linkage issue with nixlEnumStrings::telemetryCategoryStr
std::string
telemetryCategoryStr(const nixl_telemetry_category_t &category) {
    static std::array<std::string, 9> nixl_telemetry_category_str = {"NIXL_TELEMETRY_MEMORY",
                                                                     "NIXL_TELEMETRY_TRANSFER",
                                                                     "NIXL_TELEMETRY_CONNECTION",
                                                                     "NIXL_TELEMETRY_BACKEND",
                                                                     "NIXL_TELEMETRY_ERROR",
                                                                     "NIXL_TELEMETRY_PERFORMANCE",
                                                                     "NIXL_TELEMETRY_SYSTEM",
                                                                     "NIXL_TELEMETRY_CUSTOM",
                                                                     "NIXL_TELEMETRY_MAX"};
    size_t category_int = static_cast<size_t>(category);
    if (category_int >= nixl_telemetry_category_str.size()) return "BAD_CATEGORY";
    return nixl_telemetry_category_str[category_int];
}

void
getApplicationName(std::string &name) {
    std::ifstream comm("/proc/self/comm");
    getline(comm, name);
}

} // namespace

nixlTelemetryDteExporter::nixlTelemetryDteExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params),
      agent_name_(init_params.agentName),
      label_set_id_(0) {

    getApplicationName(application_name_);

    nixl_status_t status = prepareDOCATelemetryExporter();
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot prepare DOCA Telemetry Exporter: %d", status);
        throw std::runtime_error(
            absl::StrFormat("Cannot prepare DOCA Telemetry Exporter: %d", status));
    }
}

nixl_status_t
nixlTelemetryDteExporter::exportEvent(const nixlTelemetryEvent &event) {
    const std::string event_name(event.eventName_);
    doca_error_t res;
    uint64_t timestamp;
    const std::string category_name = telemetryCategoryStr(event.category_);
    const char *dynamic_label_values[] = {category_name.c_str()};

    static_assert(ARRAY_SIZE(dynamic_label_values) == ARRAY_SIZE(dte_dynamic_labels),
                  "Dynamic label values array size mismatch");

    doca_telemetry_exporter_get_timestamp(&timestamp);

    switch (event.category_) {
    case nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER: {
        res = doca_telemetry_exporter_metrics_add_counter_increment(doca_source_.get(),
                                                                    timestamp,
                                                                    event_name.c_str(),
                                                                    event.value_,
                                                                    label_set_id_,
                                                                    dynamic_label_values);
        if (res != DOCA_SUCCESS) {
            NIXL_ERROR << absl::StrFormat("Cannot add counter increment: %s",
                                          doca_error_get_name(res));
            return NIXL_ERR_BACKEND;
        }
        break;
    }
    case nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE:
    case nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY: {
        res = doca_telemetry_exporter_metrics_add_gauge_uint64(doca_source_.get(),
                                                               timestamp,
                                                               event_name.c_str(),
                                                               event.value_,
                                                               label_set_id_,
                                                               dynamic_label_values);
        if (res != DOCA_SUCCESS) {
            NIXL_ERROR << absl::StrFormat("Cannot add gauge: %s", doca_error_get_name(res));
            return NIXL_ERR_BACKEND;
        }
        break;
    }
    case nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND:
        res = doca_telemetry_exporter_metrics_add_counter(doca_source_.get(),
                                                          timestamp,
                                                          event_name.c_str(),
                                                          event.value_,
                                                          label_set_id_,
                                                          dynamic_label_values);
        if (res != DOCA_SUCCESS) {
            NIXL_ERROR << absl::StrFormat("Cannot add counter: %s", doca_error_get_name(res));
            return NIXL_ERR_BACKEND;
        }
        break;
    case nixl_telemetry_category_t::NIXL_TELEMETRY_CONNECTION:
    case nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR:
    case nixl_telemetry_category_t::NIXL_TELEMETRY_SYSTEM:
    case nixl_telemetry_category_t::NIXL_TELEMETRY_CUSTOM:
        NIXL_INFO << absl::StrFormat("Unsupported event category: %s", category_name.c_str());
        break;
    default:
        NIXL_ERROR << absl::StrFormat("Unknown event category: %s", category_name.c_str());
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

void
nixlTelemetryDteExporter::resetDOCAEnv() {
    unsetenv("CLX_API_ENABLE_EXPORT_MANAGER");
    unsetenv("CLX_OPEN_TELEMETRY_RECEIVER");
    unsetenv("PROMETHEUS_ENDPOINT");
}

nixl_status_t
nixlTelemetryDteExporter::setDOCAEnv(const std::string &application_name) {
    resetDOCAEnv();

    if (nixl::config::getValueDefaulted<bool>(dte_grpc_bug_workaround_var, false)) {
        // Workaround for DOCA Telemetry Exporter API issue in older versions:
        // The export manager (enabled by default) caused the exporter to crash.
        // Disabling the export manager eliminates the crash but reduces DOCA Telemetry Exporter API
        // functionality and effectively disables the OpenTelemetry (OTLP) and Prometheus Remote
        // Write exporters.
        setenv("CLX_API_ENABLE_EXPORT_MANAGER", "false", 1);
    }

    // Configure OTLP destination
    if (nixl::config::getValueDefaulted<bool>(dte_otlp_enabled_var, false)) {
        const auto otpl_address = nixl::config::getValueOptional<std::string>(dte_otlp_address_var);
        if (!otpl_address.has_value()) {
            NIXL_ERROR << absl::StrFormat("'%s' must be set to enable OTLP destination",
                                          dte_otlp_address_var);
            return NIXL_ERR_BACKEND;
        }

        const auto otpl_port =
            nixl::config::getValueDefaulted<uint16_t>(dte_otlp_port_var, dte_default_otlp_port);

        std::string receiver_url = absl::StrFormat("http://%s:%d%s",
                                                   otpl_address.value().c_str(),
                                                   otpl_port,
                                                   dte_otlp_write_endpoint.data());

        setenv("CLX_OPEN_TELEMETRY_RECEIVER", receiver_url.c_str(), 1);
        setenv("CLX_OPEN_TELEMETRY_WITH_DATA_POINT_ATTRIBUTES", "true", 1);
        setenv("CLX_OPEN_TELEMETRY_TYPE_AS_LABEL", "true", 1);
        setenv("CLX_OPEN_TELEMETRY_SERVICE_NAME", application_name.c_str(), 1);
        setenv("CLX_OPEN_TELEMETRY_TAG_AS_LABEL", "true", 1);
    }

    // Configure Prometheus endpoint
    if (nixl::config::getValueDefaulted<bool>(dte_prometheus_ep_enabled_var, false)) {
        const auto prometheus_ep_address = nixl::config::getValueDefaulted<std::string>(
            dte_prometheus_ep_address_var, std::string(dte_default_prometheus_ep_endpoint_address));
        const auto prometheus_ep_port = nixl::config::getValueDefaulted<uint16_t>(
            dte_prometheus_ep_port_var, dte_default_prometheus_ep_endpoint_port);
        std::string receiver_url =
            absl::StrFormat("http://%s:%d", prometheus_ep_address, prometheus_ep_port);

        setenv("PROMETHEUS_ENDPOINT", receiver_url.c_str(), 1);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTelemetryDteExporter::createDOCATelemetrySchema() {
    doca_telemetry_exporter_schema *doca_schema_ptr;
    doca_error_t res =
        doca_telemetry_exporter_schema_init("nixl_telemetry_schema", &doca_schema_ptr);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot init doca schema: %s", doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }

    doca_schema_.reset(doca_schema_ptr);

    // Configure DOCA Telemetry Exporter buffer data root
    const auto telemetry_data_root = nixl::config::getValueOptional<std::string>(dte_data_root_var);
    if (telemetry_data_root.has_value()) {
        doca_telemetry_exporter_schema_set_buf_data_root(doca_schema_.get(),
                                                         telemetry_data_root.value().c_str());
    }

    // Configure IPC destination
    if (nixl::config::getValueDefaulted<bool>(dte_ipc_enabled_var, false)) {
        doca_telemetry_exporter_schema_set_ipc_enabled(doca_schema_.get());

        const auto sockets_dir =
            nixl::config::getValueOptional<std::string>(dte_ipc_sockets_dir_var);
        if (sockets_dir.has_value()) {
            doca_telemetry_exporter_schema_set_ipc_sockets_dir(doca_schema_.get(),
                                                               sockets_dir.value().c_str());
        }

        const auto reconnect_time =
            nixl::config::getValueDefaulted<uint32_t>(dte_ipc_reconnect_time_var, 0);
        if (reconnect_time > 0) {
            doca_telemetry_exporter_schema_set_ipc_reconnect_time(doca_schema_.get(),
                                                                  reconnect_time);
        }

        const auto reconnect_tries =
            nixl::config::getValueDefaulted<uint8_t>(dte_ipc_reconnect_tries_var, 0);
        if (reconnect_tries) {
            doca_telemetry_exporter_schema_set_ipc_reconnect_tries(doca_schema_.get(),
                                                                   reconnect_tries);
        }

        const auto socket_timeout =
            nixl::config::getValueDefaulted<uint32_t>(dte_ipc_socket_timeout_var, 0);
        if (socket_timeout > 0) {
            doca_telemetry_exporter_schema_set_ipc_socket_timeout(doca_schema_.get(),
                                                                  socket_timeout);
        }
    }

    // Configure file destination
    if (nixl::config::getValueDefaulted<bool>(dte_file_enabled_var, false)) {
        doca_telemetry_exporter_schema_set_file_write_enabled(doca_schema_.get());

        const auto max_size = nixl::config::getValueDefaulted<size_t>(dte_file_max_size_var, 0);
        if (max_size > 0) {
            doca_telemetry_exporter_schema_set_file_write_max_size(doca_schema_.get(), max_size);
        }

        const auto max_age = nixl::config::getValueDefaulted<doca_telemetry_exporter_timestamp_t>(
            dte_file_max_age_var, 0);
        if (max_age > 0) {
            doca_telemetry_exporter_schema_set_file_write_max_age(doca_schema_.get(), max_age);
        }
    }

    res = doca_telemetry_exporter_schema_start(doca_schema_.get());
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot start doca schema: %s", doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTelemetryDteExporter::createDOCATelemetrySource() {
    char hostname[HOST_NAME_MAX + 1];
    if (gethostname(hostname, sizeof(hostname)) < 0) {
        NIXL_ERROR << absl::StrFormat("Cannot get hostname: %s", strerror(errno));
        return NIXL_ERR_BACKEND;
    }

    doca_telemetry_exporter_source *doca_source_ptr;
    doca_error_t res = doca_telemetry_exporter_source_create(doca_schema_.get(), &doca_source_ptr);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot create doca source: %s", doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }

    doca_source_.reset(doca_source_ptr);

    doca_telemetry_exporter_source_set_id(doca_source_.get(), hostname);
    doca_telemetry_exporter_source_set_tag(doca_source_.get(), application_name_.c_str());

    // Start DOCA Telemetry source
    res = doca_telemetry_exporter_source_start(doca_source_.get());
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot start doca source: %s", doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }

    // Create metrics context - must be done AFTER starting the source
    res = doca_telemetry_exporter_metrics_create_context(doca_source_.get());
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot create metrics context: %s",
                                      doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << "Metrics context created successfully";

    // Set automatic flush interval - metrics will be flushed automatically every N milliseconds
    res = doca_telemetry_exporter_metrics_set_flush_interval_ms(doca_source_.get(),
                                                                dte_flush_interval_ms);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Failed to set metrics flush interval: %s",
                                      doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }
    NIXL_INFO << absl::StrFormat("Automatic flush interval set to %d ms", dte_flush_interval_ms);

    // Add constant labels
    res = doca_telemetry_exporter_metrics_add_constant_label(
        doca_source_.get(), "app_name", application_name_.c_str());
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot add constant label 'app_name': %s",
                                      doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }

    res = doca_telemetry_exporter_metrics_add_constant_label(
        doca_source_.get(), "hostname", hostname);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot add constant label 'hostname': %s",
                                      doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }

    res = doca_telemetry_exporter_metrics_add_constant_label(
        doca_source_.get(), "agent_name", agent_name_.c_str());
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot add constant label 'agent_name': %s",
                                      doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }

    // Register dynamic labels
    res = doca_telemetry_exporter_metrics_add_label_names(
        doca_source_.get(), dte_dynamic_labels, ARRAY_SIZE(dte_dynamic_labels), &label_set_id_);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot add dynamic label names: %s",
                                      doca_error_get_name(res));
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTelemetryDteExporter::prepareDOCATelemetryExporter() {
    nixl_status_t status = setDOCAEnv(application_name_);
    if (status != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }

    status = createDOCATelemetrySchema();
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot create doca schema: %d", status);
        doca_schema_.reset();
        resetDOCAEnv();
        return NIXL_ERR_BACKEND;
    }

    status = createDOCATelemetrySource();
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot create doca source: %d", status);
        doca_source_.reset();
        doca_schema_.reset();
        resetDOCAEnv();
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << "DOCA Telemetry Exporter prepared successfully";

    return NIXL_SUCCESS;
}
