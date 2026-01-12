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
#ifndef _TELEMETRY_BUFFER_EXPORTER_H
#define _TELEMETRY_BUFFER_EXPORTER_H

#include "common/cyclic_buffer.h"
#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"
#include "nixl_types.h"

#include <filesystem>

constexpr const char telemetryDirVar[] = "NIXL_TELEMETRY_DIR";

/**
 * @class nixlTelemetryBufferExporter
 * @brief Shared memory buffer based telemetry exporter implementation
 *
 * This class implements the telemetry exporter interface to export
 * telemetry events to a shared memory buffer.
 */
class nixlTelemetryBufferExporter : public nixlTelemetryExporter {
public:
    /**
     * @brief Constructor using init params (plugin-compatible)
     * @param init_params Initialization parameters
     */
    explicit nixlTelemetryBufferExporter(const nixlTelemetryExporterInitParams &init_params);

    nixl_status_t
    exportEvent(const nixlTelemetryEvent &event) override;

private:
    std::filesystem::path filePath_;
    sharedRingBuffer<nixlTelemetryEvent> buffer_;
};

#endif // _TELEMETRY_BUFFER_EXPORTER_H
