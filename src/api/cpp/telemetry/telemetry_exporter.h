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
#ifndef NIXL_SRC_API_CPP_TELEMETRY_TELEMETRY_EXPORTER_H
#define NIXL_SRC_API_CPP_TELEMETRY_TELEMETRY_EXPORTER_H

#include "nixl_types.h"
#include "telemetry_event.h"

#include <string>

inline constexpr char telemetryExporterVar[] = "NIXL_TELEMETRY_EXPORTER";
inline constexpr char telemetryExporterOutputPathVar[] = "NIXL_TELEMETRY_EXPORTER_OUTPUT_PATH";

/**
 * @struct nixlTelemetryExporterInitParams
 * @brief Initialization parameters for telemetry exporters
 */
struct nixlTelemetryExporterInitParams {
    std::string outputPath; // Output path (file path, URL, etc.)
    std::string agentName;
    size_t maxEventsBuffered;
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
public:
    explicit nixlTelemetryExporter(const nixlTelemetryExporterInitParams &init_params) noexcept
        : maxEventsBuffered_(init_params.maxEventsBuffered) {};
    nixlTelemetryExporter(nixlTelemetryExporter &&) = delete;
    nixlTelemetryExporter(const nixlTelemetryExporter &) = delete;

    void
    operator=(nixlTelemetryExporter &&) = delete;
    void
    operator=(const nixlTelemetryExporter &) = delete;

    virtual ~nixlTelemetryExporter() = default;

    [[nodiscard]] size_t
    getMaxEventsBuffered() const noexcept {
        return maxEventsBuffered_;
    }

    virtual nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) = 0;

private:
    const size_t maxEventsBuffered_;
};

#endif // NIXL_SRC_API_CPP_TELEMETRY_TELEMETRY_EXPORTER_H
