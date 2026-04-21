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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_DOCA_EXPORTER_H
#define NIXL_SRC_PLUGINS_TELEMETRY_DOCA_EXPORTER_H

#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "nixl_types.h"

#include <memory>
#include <string>

struct DocaSharedContext;

class nixlTelemetryDocaExporter : public nixlTelemetryExporter {
public:
    explicit nixlTelemetryDocaExporter(const nixlTelemetryExporterInitParams &init_params);
    ~nixlTelemetryDocaExporter() override;

    nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

private:
    const std::string agent_name_;
    std::shared_ptr<DocaSharedContext> ctx_;

    void
    initializeDoca(const std::string &bind_address);

    doca_error_t
    registerCounter(const nixlTelemetryEvent &event, const char *label_values[]);

    doca_error_t
    registerGauge(const nixlTelemetryEvent &event, const char *label_values[]);
};

#endif // NIXL_SRC_PLUGINS_TELEMETRY_DOCA_EXPORTER_H
