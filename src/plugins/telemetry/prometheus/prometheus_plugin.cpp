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

#include "prometheus_exporter.h"
#include "telemetry/telemetry_plugin.h"
#include "telemetry/telemetry_exporter.h"

// Plugin type alias for convenience
using prometheus_exporter_plugin_t = nixlTelemetryPluginCreator<nixlTelemetryPrometheusExporter>;

// Plugin initialization function - must be extern "C" for dynamic loading
extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT nixlTelemetryPlugin *
nixl_telemetry_plugin_init() {
    return prometheus_exporter_plugin_t::create(
        nixlTelemetryPluginApiVersionV1, "prometheus", "1.0.0");
}

// Plugin cleanup function
extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT void
nixl_telemetry_plugin_fini() {
    // Nothing to clean up for prometheus exporter
}
