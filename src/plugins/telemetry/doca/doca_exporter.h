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

#include <doca_error.h>
#include <memory>
#include <string>

struct DocaSharedContext;

class nixlTelemetryDocaExporter : public nixlTelemetryExporter {
public:
    explicit nixlTelemetryDocaExporter(const nixlTelemetryExporterInitParams &init_params);
    ~nixlTelemetryDocaExporter() override;

    [[nodiscard]] nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

    // Force-pushes accumulated metrics to the DOCA/CollectX Prometheus endpoint.
    // Production relies on CollectX auto-flush (buffer fill / flush interval);
    // this is exposed so tests can flush a few events before scraping, since a
    // handful of samples will not fill the buffer to trigger an auto-flush.
    [[nodiscard]] nixl_status_t
    flush();

private:
    const std::string agent_name_;
    std::shared_ptr<DocaSharedContext> ctx_;

    // Appends a counter sample to the time-series. DOCA accumulates the value
    // (add_counter_increment), so repeated per-operation deltas produce a
    // monotonic cumulative total, matching the Prometheus exporter.
    [[nodiscard]] doca_error_t
    appendCounterSample(const nixlTelemetryEvent &event, const char *label_values[]);

    // Appends a gauge sample to the time-series (absolute last-operation value).
    [[nodiscard]] doca_error_t
    appendGaugeSample(const nixlTelemetryEvent &event,
                      const char *metric_name,
                      const char *label_values[]);
};

#endif // NIXL_SRC_PLUGINS_TELEMETRY_DOCA_EXPORTER_H
