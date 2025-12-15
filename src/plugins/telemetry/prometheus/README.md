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

# NIXL Prometheus Telemetry exporter plug-in

This telemetry exporter plug-in exports NIXL telemetry events in Prometheus format, by exposing an HTTP endpoint that can be scraped by Prometheus servers.
More detailed information on NIXL telemetry [docs/telemetry.md](../../../../docs/telemetry.md).

## Dependencies

The Prometheus exporter requires the prometheus-cpp library, which is included as a subproject.

libcurl is not downloaded automatically. To build, you need to install the libcurl package:

```bash
# Ubuntu/Debian
sudo apt-get install libcurl4-openssl-dev
# RHEL/CentOS/Fedora
sudo dnf install libcurl-devel
```

## Configuration

To enable the Prometheus plug-in, set the following environment variables:

```bash
export NIXL_TELEMETRY_ENABLE="y" # Enable NIXL telemetry
export NIXL_TELEMETRY_EXPORTER="prometheus" # Sets which plug-in to select in format libtelemetry_exporter_${NIXL_TELEMETRY_EXPORTER}.so
```

### Optional Configuration

You can configure the exposed prometheus port:

```bash
# Default port is 9090
export NIXL_TELEMETRY_PROMETHEUS_PORT="<port_num>"
```

Default addres is public, but you configure to expose prometheus endpoint only on localhost:

```bash
export NIXL_TELEMETRY_PROMETHEUS_LOCAL="y"
# May also use "yes" or "1"
```

You can alter where to look for plug-in .so files
NOTE: the same var is used for backend plug-ins search

```bash
export NIXL_PLUGIN_DIR="path/to/dir/with/.so/files"
```

### Metrics & Events

| Event Name | Category | Counter | Gauge | Histogram |
|------------|----------|---------|-------|-----------|
| `agent_memory_registered` | `NIXL_TELEMETRY_MEMORY` | Next version | Yes | No |
| `agent_memory_deregistered` | `NIXL_TELEMETRY_MEMORY` | Next version | Yes | No |
| `agent_tx_bytes` | `NIXL_TELEMETRY_TRANSFER` | Yes | Next version | No |
| `agent_rx_bytes` | `NIXL_TELEMETRY_TRANSFER` | Yes | Next version | No |
| `agent_tx_requests_num` | `NIXL_TELEMETRY_TRANSFER` | Yes | Next version | No |
| `agent_rx_requests_num` | `NIXL_TELEMETRY_TRANSFER` | Yes | Next version | No |
| `agent_xfer_time` | `NIXL_TELEMETRY_PERFORMANCE` | Yes | No | Next Version |
| `agent_xfer_post_time` | `NIXL_TELEMETRY_PERFORMANCE` | Yes | No | Next Version |
| Backend-specific events | `NIXL_TELEMETRY_BACKEND` | - | Yes | Next version | No |
| Error status strings | `NIXL_TELEMETRY_ERROR` | No | No | No |

**Counter, Gauge, Histogram** - as implemented by the Prometheus exporter
- **Counter**: Instance lifetime count of the related value. Summed over the separate events' values.
- **Gauge**: Shows the value per the last event (transaction). E.g agent_memory_registered represents the memory amount registered by the last operation (and not the total memory registered during instance lifetime). The value is updated per each event (request) and can grow or decrease.
- **Histogram**: Counts the number of observations per pre-defined bins. Please see [Prometheus histograms documentation](https://prometheus.io/docs/practices/histograms/) for more details.

### Metric labels

Each telemetry metrics is provided with the following labels:
- Telemetry Category
- Hostname where the agent runs
- Agent name (as custom provided during initialization, can be deprecated in the next versions)
