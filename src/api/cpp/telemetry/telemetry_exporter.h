/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _TELEMETRY_EXPORTER_H
#define _TELEMETRY_EXPORTER_H

#include "nixl_types.h"
#include "telemetry_event.h"
#include "common/cyclic_buffer.h"

#include <string>
#include <vector>
#include <fstream>

constexpr char TELEMETRY_EXPORTER_VAR[] = "NIXL_TELEMETRY_EXPORTER";
constexpr char TELEMETRY_EXPORTER_OUTPUT_PATH_VAR[] = "NIXL_TELEMETRY_EXPORTER_OUTPUT_PATH";

/**
 * @struct nixlTelemetryExporterInitParams
 * @brief Initialization parameters for telemetry exporters
 */
struct nixlTelemetryExporterInitParams {
    std::string outputPath; // Output path (file path, URL, etc.)
    uint32_t eventLimit; // Maximum number of events to buffer
};

/**
 * @class nixlTelemetryExporter
 * @brief Abstract base class for telemetry exporters
 *
 * This class defines the interface that all telemetry exporters must implement.
 * It provides the core functionality for reading telemetry events and exporting
 * them to various destinations.
 */
class nixlTelemetryExporter {
protected:
    uint32_t eventLimit_;

public:
    explicit nixlTelemetryExporter(const nixlTelemetryExporterInitParams *init_params)
        : eventLimit_(init_params->eventLimit) {};
    nixlTelemetryExporter(nixlTelemetryExporter &&) = delete;
    nixlTelemetryExporter(const nixlTelemetryExporter &) = delete;

    void
    operator=(nixlTelemetryExporter &&) = delete;
    void
    operator=(const nixlTelemetryExporter &) = delete;

    virtual ~nixlTelemetryExporter() = default;

    uint32_t
    getEventLimit() const noexcept {
        return eventLimit_;
    }

    virtual nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) = 0;
};

#endif // _TELEMETRY_EXPORTER_H
