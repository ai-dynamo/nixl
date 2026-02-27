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
#ifndef _TELEMETRY_DTE_EXPORTER_H
#define _TELEMETRY_DTE_EXPORTER_H

#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "nixl_types.h"

#include <doca_telemetry_exporter.h>

class nixlTelemetryDteExporter : public nixlTelemetryExporter {
public:
    explicit nixlTelemetryDteExporter(const nixlTelemetryExporterInitParams &init_params);
    ~nixlTelemetryDteExporter();
    nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

protected:
    doca_error_t
    prepareDOCATelemetryExporter(void);

private:
    std::string agent_name_;
    struct doca_telemetry_exporter_schema *doca_schema_;
    struct doca_telemetry_exporter_source *doca_source_;
    uint64_t label_set_id_;
};

#endif // _TELEMETRY_DTE_EXPORTER_H
