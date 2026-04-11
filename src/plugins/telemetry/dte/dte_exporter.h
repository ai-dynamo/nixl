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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_DTE_DTE_EXPORTER_H
#define NIXL_SRC_PLUGINS_TELEMETRY_DTE_DTE_EXPORTER_H

#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "nixl_types.h"
#include <memory>

#include <doca_telemetry_exporter.h>

struct docaTelemetryExporterSourceDeleter {
    void
    operator()(doca_telemetry_exporter_source *p) const {
        doca_telemetry_exporter_source_destroy(p);
    }
};

struct docaTelemetryExporterSchemaDeleter {
    void
    operator()(doca_telemetry_exporter_schema *p) const {
        doca_telemetry_exporter_schema_destroy(p);
    }
};

using docaTelemetryExporterSource =
    std::unique_ptr<doca_telemetry_exporter_source, docaTelemetryExporterSourceDeleter>;
using docaTelemetryExporterSchema =
    std::unique_ptr<doca_telemetry_exporter_schema, docaTelemetryExporterSchemaDeleter>;

class nixlTelemetryDteExporter : public nixlTelemetryExporter {
public:
    explicit nixlTelemetryDteExporter(const nixlTelemetryExporterInitParams &init_params);
    nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

protected:
    [[nodiscard]] static nixl_status_t
    setDOCAEnv(const std::string &application_name);
    static void
    resetDOCAEnv();
    [[nodiscard]] nixl_status_t
    createDOCATelemetrySchema();
    [[nodiscard]] nixl_status_t
    createDOCATelemetrySource();
    [[nodiscard]] nixl_status_t
    prepareDOCATelemetryExporter();

private:
    std::string application_name_;
    const std::string agent_name_;
    docaTelemetryExporterSchema doca_schema_;
    docaTelemetryExporterSource doca_source_;
    uint64_t label_set_id_;
};

#endif // NIXL_SRC_PLUGINS_TELEMETRY_DTE_DTE_EXPORTER_H
