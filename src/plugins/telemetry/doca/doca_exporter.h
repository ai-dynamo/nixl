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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_DOCA_EXPORTER_H
#define NIXL_SRC_PLUGINS_TELEMETRY_DOCA_EXPORTER_H

#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "nixl_types.h"

#include <doca_telemetry_exporter.h>
#include <string>
#include <memory>
#include <mutex>

struct DocaSharedContext {
    doca_telemetry_exporter_schema *schema = nullptr;
    doca_telemetry_exporter_source *source = nullptr;
    doca_telemetry_exporter_label_set_id_t label_set_id = 0;
    ~DocaSharedContext();
};

class nixlTelemetryDocaExporter : public nixlTelemetryExporter {
public:
    explicit nixlTelemetryDocaExporter(const nixlTelemetryExporterInitParams &init_params);

    nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

private:
    static std::mutex s_ctx_mutex_;
    static std::weak_ptr<DocaSharedContext> s_ctx_weak_;

    const bool local_ = false;
    const uint16_t port_;
    bool initialized_ = false;
    const std::string agent_name_;
    const std::string hostname_;
    std::shared_ptr<DocaSharedContext> ctx_;
    std::string bind_address_;

    nixl_status_t
    initializeDoca(const nixlTelemetryExporterInitParams &params);

    doca_error_t
    registerCounter(const nixlTelemetryEvent &event, const char *label_values[]);

    doca_error_t
    registerGauge(const nixlTelemetryEvent &event, const char *label_values[]);
};

#endif // NIXL_SRC_PLUGINS_TELEMETRY_DOCA_EXPORTER_H
