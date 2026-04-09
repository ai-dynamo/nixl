<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NIXL DOCA Telemetry Exporter Plug-in

This telemetry exporter plug-in exports NIXL telemetry events via DOCA Telemetry Exporter, by exposing an HTTP endpoint that can be scraped by Prometheus servers.
More detailed information on NIXL telemetry [docs/telemetry.md](../../../../docs/telemetry.md).

## Dependencies

DOCA exporter requires the DOCA Telemetry Exporter library to be present on the system.
If the DOCA headers are not found at build time, this plug-in is automatically skipped.

## Configuration

To enable the DOCA plug-in, set the following environment variables:

```bash
export NIXL_TELEMETRY_ENABLE="y" # Enable NIXL telemetry
export NIXL_TELEMETRY_EXPORTER="doca" # Sets which plug-in to select in format libtelemetry_exporter_${NIXL_TELEMETRY_EXPORTER}.so
```

### Optional Configuration

You can configure the exposed prometheus port:

```bash
# Default port is 9091
export NIXL_TELEMETRY_DOCA_PROMETHEUS_PORT="<port_num>"
```

Default address is public, but you can configure to expose prometheus endpoint only on localhost:

```bash
export NIXL_TELEMETRY_DOCA_PROMETHEUS_LOCAL="y"
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
| `agent_memory_registered` | `NIXL_TELEMETRY_MEMORY` | No | Yes | No |
| `agent_memory_deregistered` | `NIXL_TELEMETRY_MEMORY` | No | Yes | No |
| `agent_tx_bytes` | `NIXL_TELEMETRY_TRANSFER` | Yes | No | No |
| `agent_rx_bytes` | `NIXL_TELEMETRY_TRANSFER` | Yes | No | No |
| `agent_tx_requests_num` | `NIXL_TELEMETRY_TRANSFER` | Yes | No | No |
| `agent_rx_requests_num` | `NIXL_TELEMETRY_TRANSFER` | Yes | No | No |
| `agent_xfer_time` | `NIXL_TELEMETRY_PERFORMANCE` | No | Yes | No |
| `agent_xfer_post_time` | `NIXL_TELEMETRY_PERFORMANCE` | No | Yes | No |
| Backend-specific events | `NIXL_TELEMETRY_BACKEND` | Yes | No | No |
| Error status strings | `NIXL_TELEMETRY_ERROR` | No | No | No |

**Counter, Gauge, Histogram** - as implemented by the DOCA Telemetry Exporter
- **Counter**: Instance lifetime count of the related value. Summed over the separate events' values.
- **Gauge**: Shows the value per the last event (transaction). E.g agent_memory_registered represents the memory amount registered by the last operation (and not the total memory registered during instance lifetime). The value is updated per each event (request) and can grow or decrease.
- **Histogram**: Counts the number of observations per pre-defined bins. Please see [Prometheus histograms documentation](https://prometheus.io/docs/practices/histograms/) for more details.

### Metric labels

Each telemetry metric is provided with the following labels:
- Telemetry Category
- Hostname where the agent runs
- Agent name (as provided during initialization, may be deprecated in future versions)
