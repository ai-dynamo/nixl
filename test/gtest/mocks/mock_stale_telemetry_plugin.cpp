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
#include "telemetry/telemetry_plugin.h"

extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT nixlTelemetryPlugin *
nixl_telemetry_plugin_init() {
    static nixlTelemetryPlugin plugin(
        nixl_telemetry_plugin_api_version::V2, "mock_stale", "0.0.1", nullptr);
    return &plugin;
}

extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT void
nixl_telemetry_plugin_fini() {}
