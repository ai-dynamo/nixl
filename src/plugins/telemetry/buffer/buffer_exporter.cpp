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

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>

nixlTelemetryBufferExporter::nixlTelemetryBufferExporter(
    const nixlTelemetryExporterInitParams &init_params)
    : nixlTelemetryExporter(init_params),
      buffer_(std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
          init_params.outputPath + "/" + init_params.agentName,
          true,
          TELEMETRY_VERSION,
          init_params.maxEventsBuffered)) {}

nixl_status_t
nixlTelemetryBufferExporter::exportEvent(const nixlTelemetryEvent &event) {
    if (!buffer_->push(event)) {
        return NIXL_ERR_UNKNOWN;
    }

    return NIXL_SUCCESS;
}
