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
#include <absl/strings/str_format.h>

#include <limits.h>

#include <fstream>

// DOCA Telemetry Exporter API Documentation:
// https://docs.nvidia.com/doca/sdk/doca-telemetry-exporter/index.html

#define FLUSH_INTERVAL_MS 1000

static const char *dynamic_labels[] = {"category"};

const char dteGRPCBugWorkaroundVar[] = "NIXL_TELEMETRY_DTE_ENABLE_GRPC_BUG_WORKAROUND";
const char dteDataRootVar[] = "NIXL_TELEMETRY_DTE_DATA_ROOT";
const char dtePrometheusEPEnabledVar[] = "NIXL_TELEMETRY_DTE_PROMETHEUS_EP_ENABLED";
const char dtePrometheusEPAddressVar[] = "NIXL_TELEMETRY_DTE_PROMETHEUS_EP_ADDRESS";
const char dtePrometheusEPPortVar[] = "NIXL_TELEMETRY_DTE_PROMETHEUS_EP_PORT";
const char dteIPCEnabledVar[] = "NIXL_TELEMETRY_DTE_IPC_ENABLED";
const char dteIPCSocketsDirVar[] = "NIXL_TELEMETRY_DTE_IPC_SOCKETS_DIR";
const char dteIPCReconnectTimeVar[] = "NIXL_TELEMETRY_DTE_IPC_RECONNECT_TIME";
const char dteIPCReconnectTriesVar[] = "NIXL_TELEMETRY_DTE_IPC_RECONNECT_TRIES";
const char dteIPCSocketTimeoutVar[] = "NIXL_TELEMETRY_DTE_IPC_SOCKET_TIMEOUT";
const char dteFileEnabledVar[] = "NIXL_TELEMETRY_DTE_FILE_ENABLED";
const char dteFileMaxSizeVar[] = "NIXL_TELEMETRY_DTE_FILE_MAX_SIZE";
const char dteFileMaxAgeVar[] = "NIXL_TELEMETRY_DTE_FILE_MAX_AGE";
const char dteOTLPEnabledVar[] = "NIXL_TELEMETRY_DTE_OTLP_ENABLED";
const char dteOTLPAddressVar[] = "NIXL_TELEMETRY_DTE_OTLP_ADDRESS";
const char dteOTLPPortVar[] = "NIXL_TELEMETRY_DTE_OTLP_PORT";

static const uint16_t DOCA_TELEMETRY_DEFAULT_OTLP_PORT = 9502;
static const std::string_view DOCA_TELEMETRY_OTLP_WRITE_ENDPOINT = "/v1/metrics";
static const char *DOCA_TELEMETRY_DEFAULT_PROMETHEUS_EP_ENDPOINT_ADDRESS = "0.0.0.0";
static const uint16_t DOCA_TELEMETRY_DEFAULT_PROMETHEUS_EP_ENDPOINT_PORT = 9101;

static bool
getenvBool(const char *var) {
    auto value_str = std::getenv(var);
    if (!value_str) {
        return false;
    }

    return strcasecmp(value_str, "y") == 0 || strcasecmp(value_str, "1") == 0 ||
        strcasecmp(value_str, "yes") == 0 || strcasecmp(value_str, "true") == 0;
}

template<typename T>
T
getenvInt(const char *var, T default_value = 0) {
    auto value_str = std::getenv(var);
    if (!value_str) {
        return default_value;
    }

    std::stringstream ss(value_str);
    T num;

    if (ss >> num) {
        return num;
    }

    return default_value;
}

// TODO: solve backward linkage issue with nixlEnumStrings::telemetryCategoryStr
static std::string
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

static void
getApplicationName(std::string &name) {
    std::ifstream comm("/proc/self/comm");
    getline(comm, name);
}

nixlTelemetryDteExporter::nixlTelemetryDteExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params),
      agent_name_(init_params.agentName),
      doca_schema_(NULL),
      doca_source_(NULL),
      label_set_id_(0) {}

nixlTelemetryDteExporter::~nixlTelemetryDteExporter() {
    if (doca_source_) {
        doca_telemetry_exporter_source_destroy(doca_source_);
    }
    if (doca_schema_) {
        doca_telemetry_exporter_schema_destroy(doca_schema_);
    }
}

nixl_status_t
nixlTelemetryDteExporter::exportEvent(const nixlTelemetryEvent &event) {
    const std::string event_name(event.eventName_);
    doca_error_t res;
    uint64_t timestamp;
    std::string category_name = telemetryCategoryStr(event.category_);
    const char *dynamic_label_values[] = {category_name.c_str()};

    static_assert(ARRAY_SIZE(dynamic_label_values) == ARRAY_SIZE(dynamic_labels),
                  "Dynamic label values array size mismatch");

    if (!doca_schema_) {
        res = prepareDOCATelemetryExporter();
        if (res != DOCA_SUCCESS) {
            NIXL_ERROR << absl::StrFormat("Cannot prepare DOCA Telemetry Exporter: %s",
                                          doca_error_get_name(res));
            return NIXL_ERR_BACKEND;
        }
    }

    doca_telemetry_exporter_get_timestamp(&timestamp);

    switch (event.category_) {
    case nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER: {
        res = doca_telemetry_exporter_metrics_add_counter_increment(doca_source_,
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
        res = doca_telemetry_exporter_metrics_add_gauge_uint64(doca_source_,
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
        res = doca_telemetry_exporter_metrics_add_counter(doca_source_,
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
    default:
        NIXL_WARN << absl::StrFormat("Unknown event category: %s", category_name.c_str());
        break;
    }

    return NIXL_SUCCESS;
}

doca_error_t
nixlTelemetryDteExporter::prepareDOCATelemetryExporter(void) {
    doca_error_t res;
    char hostname[HOST_NAME_MAX + 1];
    std::string application_name;

    getApplicationName(application_name);

    if (getenvBool(dteGRPCBugWorkaroundVar)) {
        // Workaround for DOCA Telemetry Exporter API issue in older versions:
        // The export manager (enabled by default) caused the exporter to crash.
        // Disabling the export manager eliminates the crash but reduces DOCA Telemetry Exporter API
        // functionality and effectively disables the OpenTelemetry (OTLP) and Prometheus Remote
        // Write exporters.
        setenv("CLX_API_ENABLE_EXPORT_MANAGER", " false", 1);
    }

    // Initialize DOCA Telemetry schema
    res = doca_telemetry_exporter_schema_init("nixl_telemetry_schema", &doca_schema_);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot init doca schema: %s", doca_error_get_name(res));
        return res;
    }

    // Configure DOCA Telemetry Exporter buffer data root
    char *telemetry_data_root = getenv(dteDataRootVar);
    if (telemetry_data_root && telemetry_data_root[0] != '\0') {
        doca_telemetry_exporter_schema_set_buf_data_root(doca_schema_, telemetry_data_root);
    }

    // Configure IPC destination
    if (getenvBool(dteIPCEnabledVar)) {
        doca_telemetry_exporter_schema_set_ipc_enabled(doca_schema_);

        char *sockets_dir = getenv(dteIPCSocketsDirVar);
        if (sockets_dir && sockets_dir[0] != '\0') {
            doca_telemetry_exporter_schema_set_ipc_sockets_dir(doca_schema_, sockets_dir);
        }

        uint32_t reconnect_time = getenvInt(dteIPCReconnectTimeVar, 0);
        if (reconnect_time > 0) {
            doca_telemetry_exporter_schema_set_ipc_reconnect_time(doca_schema_, reconnect_time);
        }

        uint8_t reconnect_tries = getenvInt(dteIPCReconnectTriesVar, 0);
        if (reconnect_tries) {
            doca_telemetry_exporter_schema_set_ipc_reconnect_tries(doca_schema_, reconnect_tries);
        }

        uint32_t socket_timeout = getenvInt(dteIPCSocketTimeoutVar, 0);
        if (socket_timeout > 0) {
            doca_telemetry_exporter_schema_set_ipc_socket_timeout(doca_schema_, socket_timeout);
        }
    }

    // Configure file destination
    if (getenvBool(dteFileEnabledVar)) {
        doca_telemetry_exporter_schema_set_file_write_enabled(doca_schema_);

        size_t max_size = getenvInt(dteFileMaxSizeVar, 0);
        if (max_size > 0) {
            doca_telemetry_exporter_schema_set_file_write_max_size(doca_schema_, max_size);
        }

        doca_telemetry_exporter_timestamp_t max_age = getenvInt(dteFileMaxAgeVar, 0);
        if (max_age > 0) {
            doca_telemetry_exporter_schema_set_file_write_max_age(doca_schema_, max_age);
        }
    }

    // Configure OTLP destination
    if (getenvBool(dteOTLPEnabledVar)) {
        char *otpl_address = getenv(dteOTLPAddressVar);
        if (!otpl_address || otpl_address[0] == '\0') {
            NIXL_ERROR << absl::StrFormat("'%s' must be set to enable OTLP destination",
                                          dteOTLPAddressVar);
            goto err_return;
        }

        uint16_t otpl_port = getenvInt(dteOTLPPortVar, 0);
        if (!otpl_port) {
            otpl_port = DOCA_TELEMETRY_DEFAULT_OTLP_PORT;
        }

        std::string receiver_url = absl::StrFormat(
            "http://%s:%d%s", otpl_address, otpl_port, DOCA_TELEMETRY_OTLP_WRITE_ENDPOINT);

        setenv("CLX_OPEN_TELEMETRY_RECEIVER", receiver_url.c_str(), 1);
        setenv("CLX_OPEN_TELEMETRY_WITH_DATA_POINT_ATTRIBUTES", "true", 1);
        setenv("CLX_OPEN_TELEMETRY_TYPE_AS_LABEL", "true", 1);
        setenv("CLX_OPEN_TELEMETRY_SERVICE_NAME", application_name.c_str(), 1);
        setenv("CLX_OPEN_TELEMETRY_TAG_AS_LABEL", "true", 1);
    } else {
        unsetenv("CLX_OPEN_TELEMETRY_RECEIVER");
    }

    // Configure Prometheus endpoint
    if (getenvBool(dtePrometheusEPEnabledVar)) {
        const char *prometheus_ep_address = getenv(dtePrometheusEPAddressVar);
        if (!prometheus_ep_address || prometheus_ep_address[0] == '\0') {
            prometheus_ep_address = DOCA_TELEMETRY_DEFAULT_PROMETHEUS_EP_ENDPOINT_ADDRESS;
        }
        uint16_t prometheus_ep_port =
            getenvInt(dtePrometheusEPPortVar, DOCA_TELEMETRY_DEFAULT_PROMETHEUS_EP_ENDPOINT_PORT);
        std::string receiver_url =
            absl::StrFormat("http://%s:%d", prometheus_ep_address, prometheus_ep_port);

        setenv("PROMETHEUS_ENDPOINT", receiver_url.c_str(), 1);
    }

    // Start DOCA Telemetry schema
    res = doca_telemetry_exporter_schema_start(doca_schema_);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot start doca schema: %s", doca_error_get_name(res));
        goto err_return;
    }

    // Create DOCA Telemetry source
    res = doca_telemetry_exporter_source_create(doca_schema_, &doca_source_);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot create doca source: %s", doca_error_get_name(res));
        goto err_return;
    }

    if (gethostname(hostname, sizeof(hostname)) < 0) {
        NIXL_ERROR << absl::StrFormat("Cannot get hostname: %s", strerror(errno));
        goto err_return;
    }

    doca_telemetry_exporter_source_set_id(doca_source_, hostname);
    doca_telemetry_exporter_source_set_tag(doca_source_, application_name.c_str());

    // Start DOCA Telemetry source
    res = doca_telemetry_exporter_source_start(doca_source_);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot start doca source: %s", doca_error_get_name(res));
        goto err_return;
    }

    // Create metrics context - must be done AFTER starting the source
    res = doca_telemetry_exporter_metrics_create_context(doca_source_);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot create metrics context: %s",
                                      doca_error_get_name(res));
        goto err_return;
    }

    NIXL_INFO << "Metrics context created successfully";

    // Set automatic flush interval - metrics will be flushed automatically every N milliseconds
    res = doca_telemetry_exporter_metrics_set_flush_interval_ms(doca_source_, FLUSH_INTERVAL_MS);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Failed to set metrics flush interval: %s",
                                      doca_error_get_name(res));
        goto err_return;
    }
    NIXL_INFO << absl::StrFormat("Automatic flush interval set to %d ms", FLUSH_INTERVAL_MS);

    // Add constant labels
    res = doca_telemetry_exporter_metrics_add_constant_label(
        doca_source_, "app_name", application_name.c_str());
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot add constant label 'app_name': %s",
                                      doca_error_get_name(res));
        goto err_return;
    }

    res = doca_telemetry_exporter_metrics_add_constant_label(doca_source_, "hostname", hostname);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot add constant label 'hostname': %s",
                                      doca_error_get_name(res));
        goto err_return;
    }

    res = doca_telemetry_exporter_metrics_add_constant_label(
        doca_source_, "agent_name", agent_name_.c_str());
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot add constant label 'agent_name': %s",
                                      doca_error_get_name(res));
        goto err_return;
    }

    // Register dynamic labels
    res = doca_telemetry_exporter_metrics_add_label_names(
        doca_source_, dynamic_labels, ARRAY_SIZE(dynamic_labels), &label_set_id_);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << absl::StrFormat("Cannot add dummy label names: %s", doca_error_get_name(res));
        goto err_return;
    }

    NIXL_INFO << "DOCA Telemetry Exporter prepared successfully";

    return DOCA_SUCCESS;

err_return:
    if (doca_source_) {
        doca_telemetry_exporter_source_destroy(doca_source_);
        doca_source_ = NULL;
    }
    doca_telemetry_exporter_schema_destroy(doca_schema_);
    doca_schema_ = NULL;
    return res;
}
