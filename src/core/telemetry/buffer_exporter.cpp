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
#include "buffer_exporter.h"
#include "common/nixl_log.h"

constexpr const char telemetryDirVar[] = "NIXL_TELEMETRY_DIR";

nixlTelemetryBufferExporter::nixlTelemetryBufferExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params) {
    auto telemetry_dir = std::getenv(telemetryDirVar);
    if (!telemetry_dir) {
        throw std::invalid_argument(std::string(telemetryDirVar) + " is not set");
    }

    auto full_file_path = std::string(telemetry_dir) + "/" + init_params.agentName.data();
    buffer_ = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        full_file_path, true, TELEMETRY_VERSION, getMaxEventsBuffered());

    NIXL_INFO << "Telemetry enabled, using buffer path: " << full_file_path
              << " with size: " << getMaxEventsBuffered();
}

nixl_status_t
nixlTelemetryBufferExporter::exportEvent(const nixlTelemetryEvent &event) {
    if (!buffer_->push(event)) {
        return NIXL_ERR_UNKNOWN;
    }

    return NIXL_SUCCESS;
}
