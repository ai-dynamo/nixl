<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NIXL Custom Telemetry Plugin Development Guide

This guide explains how to create custom telemetry exporter plugins for NIXL. Telemetry plugins allow you to export NIXL telemetry data to different monitoring systems, databases, or file formats.

## Overview

NIXL telemetry plugins are dynamically loaded shared libraries that export telemetry events from the NIXL agent

### Built-in Event Types

NIXL generates the following telemetry events:

| Category | Event Name | Type | Description |
|----------|-----------|------|-------------|
| MEMORY | `agent_memory_registered` | Gauge | Total bytes of memory registered |
| MEMORY | `agent_memory_deregistered` | Gauge | Total bytes of memory deregistered |
| TRANSFER | `agent_tx_bytes` | Counter | Total bytes transmitted |
| TRANSFER | `agent_rx_bytes` | Counter | Total bytes received |
| TRANSFER | `agent_tx_requests_num` | Counter | Number of transmit requests |
| TRANSFER | `agent_rx_requests_num` | Counter | Number of receive requests |
| PERFORMANCE | `agent_xfer_time` | Gauge | Transfer time in microseconds |
| PERFORMANCE | `agent_xfer_post_time` | Gauge | Post time in microseconds |
| BACKEND | Backend-specific events | Counter | Dynamic events from backends |

## Quick Start

Here's a minimal example of a CSV file exporter plugin:

### 1. Create Your Exporter Class (`csv_exporter.h`)

```cpp
#ifndef _TELEMETRY_CSV_EXPORTER_H
#define _TELEMETRY_CSV_EXPORTER_H

#include "telemetry/telemetry_exporter.h"
#include <fstream>

class nixlTelemetryCsvExporter : public nixlTelemetryExporter {
public:
    explicit nixlTelemetryCsvExporter(const nixlTelemetryExporterInitParams *init_params);
    ~nixlTelemetryCsvExporter() override;

    nixl_status_t exportEvent(const nixlTelemetryEvent &event) override;

private:
    std::ofstream file_;
};

#endif // _TELEMETRY_CSV_EXPORTER_H
```

### 2. Implement Your Exporter (`csv_exporter.cpp`)

```cpp
#include "csv_exporter.h"
#include "common/nixl_log.h"

nixlTelemetryCsvExporter::nixlTelemetryCsvExporter(
    const nixlTelemetryExporterInitParams *init_params)
    : nixlTelemetryExporter(init_params) {

    auto file_path = std::get_end("NIXL_TELEMETRY_CSV_FILE");
    file_.open(file_path, std::ios::out | std::ios::app);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + file_path);
    }

    // Write CSV header
    file_ << "timestamp_us,category,event_name,value\n";
    NIXL_INFO << "CSV exporter initialized: " << file_path;
}

nixl_status_t
nixlTelemetryCsvExporter::exportEvent(const nixlTelemetryEvent &event) {
    if (!file_.is_open()) {
        return NIXL_ERR_UNKNOWN;
    }

    try {
        file_ << event.timestampUs_ << ","
              << static_cast<int>(event.category_) << ","
              << event.eventName_ << ","
              << event.value_ << "\n";
        file_.flush();
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to export event: " << e.what();
        return NIXL_ERR_UNKNOWN;
    }
}
```

### 3. Create Plugin Interface (`csv_plugin.cpp`)

```cpp
#include "csv_exporter.h"
#include "telemetry/telemetry_plugin.h"

// Use the plugin creator template for minimal boilerplate
using csv_exporter_plugin_t = nixlTelemetryPluginCreator<nixlTelemetryCsvExporter>;

// Plugin initialization function - must be extern "C"
extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT nixlTelemetryPlugin *
nixl_telemetry_plugin_init() {
    return csv_exporter_plugin_t::create(
        NIXL_TELEMETRY_PLUGIN_API_VERSION,
        "csv",      // Plugin name
        "1.0.0"     // Plugin version
    );
}

// Plugin cleanup function
extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT void
nixl_telemetry_plugin_fini() {
    // Add any global cleanup if needed
}
```

### 4. Build Configuration (`meson.build`)

```meson
# CSV Exporter Plugin
csv_exporter_plugin = shared_library(
    'libtelemetry_exporter_csv',
    'csv_plugin.cpp',
    'csv_exporter.cpp',
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [nixl_infra, absl_log_dep],
    install: true,
    install_dir: get_option('libdir') / 'nixl' / 'telemetry_exporters',
    name_prefix: '',
)
```
