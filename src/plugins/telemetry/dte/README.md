<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NIXL DTE (DOCA Telemetry Exporter) Telemetry exporter plug-in

This telemetry exporter plug-in exports NIXL telemetry events via the [DOCA Telemetry Exporter API](https://docs.nvidia.com/doca/sdk/doca-telemetry-exporter/index.html). Metrics can be sent to one or more destinations: a Prometheus scrape endpoint, OpenTelemetry (OTLP), IPC sockets, or file output.

More detailed information on NIXL telemetry [docs/telemetry.md](../../../../docs/telemetry.md).

## Dependencies

The DTE exporter requires the DOCA SDK with the Telemetry Exporter API. The plug-in is built only when explicitly enabled.

Build with the DTE plug-in:

```bash
meson setup build -Denable_dte=true
# Optionally specify DOCA install path (default: /opt/mellanox/doca/)
meson setup build -Denable_dte=true -Ddoca_path=/path/to/doca
```

Required DOCA libraries: `doca_common`, `doca_telemetry_exporter`.

## Configuration

To enable the DTE plug-in, set the following environment variables:

```bash
export NIXL_TELEMETRY_ENABLE="y"   # Enable NIXL telemetry
export NIXL_TELEMETRY_EXPORTER="dte"  # Select plug-in: libtelemetry_exporter_${NIXL_TELEMETRY_EXPORTER}.so
```

### Optional configuration

#### Data root and workarounds

```bash
# Buffer data root for DOCA Telemetry Exporter
export NIXL_TELEMETRY_DTE_DATA_ROOT="<path>"

# Enable workaround for DOCA Telemetry Exporter gRPC bug (older DOCA versions).
# Disables export manager; OTLP and Prometheus Remote Write are then unavailable.
export NIXL_TELEMETRY_DTE_ENABLE_GRPC_BUG_WORKAROUND="y"  # or "yes", "1", "true"
```

#### Prometheus endpoint (scrape target)

Expose metrics for Prometheus scraping:

```bash
export NIXL_TELEMETRY_DTE_PROMETHEUS_EP_ENABLED="y"

# Optional: address (default 0.0.0.0) and port (default 9101)
export NIXL_TELEMETRY_DTE_PROMETHEUS_EP_ADDRESS="<address>"
export NIXL_TELEMETRY_DTE_PROMETHEUS_EP_PORT="<port_num>"
```

#### IPC destination

```bash
export NIXL_TELEMETRY_DTE_IPC_ENABLED="y"

# Optional
export NIXL_TELEMETRY_DTE_IPC_SOCKETS_DIR="<path>"
export NIXL_TELEMETRY_DTE_IPC_RECONNECT_TIME="<ms>"
export NIXL_TELEMETRY_DTE_IPC_RECONNECT_TRIES="<count>"
export NIXL_TELEMETRY_DTE_IPC_SOCKET_TIMEOUT="<ms>"
```

#### File destination

```bash
export NIXL_TELEMETRY_DTE_FILE_ENABLED="y"

# Optional
export NIXL_TELEMETRY_DTE_FILE_MAX_SIZE="<bytes>"
export NIXL_TELEMETRY_DTE_FILE_MAX_AGE="<timestamp>"
```

#### OpenTelemetry (OTLP) destination

```bash
export NIXL_TELEMETRY_DTE_OTLP_ENABLED="y"
export NIXL_TELEMETRY_DTE_OTLP_ADDRESS="<host>"   # Required when OTLP is enabled

# Optional: port (default 9502)
export NIXL_TELEMETRY_DTE_OTLP_PORT="<port_num>"
```

#### Plug-in search path

Same variable as for other telemetry plug-ins:

```bash
export NIXL_PLUGIN_DIR="path/to/dir/with/.so/files"
```

### Metrics & Events

| Event Name | Category | Counter | Gauge | Exported by DTE |
|------------|----------|---------|-------|-----------------|
| `agent_memory_registered` | `NIXL_TELEMETRY_MEMORY` | No | Yes | Yes |
| `agent_memory_deregistered` | `NIXL_TELEMETRY_MEMORY` | No | Yes | Yes |
| `agent_tx_bytes` | `NIXL_TELEMETRY_TRANSFER` | Yes | No | Yes |
| `agent_rx_bytes` | `NIXL_TELEMETRY_TRANSFER` | Yes | No | Yes |
| `agent_tx_requests_num` | `NIXL_TELEMETRY_TRANSFER` | Yes | No | Yes |
| `agent_rx_requests_num` | `NIXL_TELEMETRY_TRANSFER` | Yes | No | Yes |
| `agent_xfer_time` | `NIXL_TELEMETRY_PERFORMANCE` | No | Yes | Yes |
| `agent_xfer_post_time` | `NIXL_TELEMETRY_PERFORMANCE` | No | Yes | Yes |
| Backend-specific events | `NIXL_TELEMETRY_BACKEND` | Yes | - | Yes |
| Connection / Error / System / Custom events | `NIXL_TELEMETRY_CONNECTION`, `_ERROR`, `_SYSTEM`, `_CUSTOM` | - | - | No (logged only) |

- **Counter**: Instance lifetime count; increments summed from events (e.g. transfer bytes, request counts).
- **Gauge**: Value from the last event (e.g. last memory registered, last transfer time); updated per event.

Metrics are flushed automatically at a fixed interval (see plug-in implementation).
