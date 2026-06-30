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
# Can be set to `y`/`yes`/`on`/`true`/`enable`/`1` to enable local only, and `n`/`no`/`off`/`false`/`disable`/`0` (or not set) to disable. Matching is case insensitive.
```

You can alter where to look for plug-in .so files
NOTE: the same var is used for backend plug-ins search

```bash
export NIXL_PLUGIN_DIR="path/to/dir/with/.so/files"
```

### Metrics & Events

| Event Name | Counter | Gauge | Histogram |
| ---------- | ------- | ----- | --------- |
| `agent_memory_registered` | Yes | Yes | No |
| `agent_memory_deregistered` | Yes | Yes | No |
| `agent_tx_bytes` | Yes | Yes | No |
| `agent_rx_bytes` | Yes | Yes | No |
| `agent_tx_requests_num` | Yes | No | No |
| `agent_rx_requests_num` | Yes | No | No |
| `agent_xfer_time` | Yes | No | No |
| `agent_xfer_post_time` | Yes | No | No |
| Error event types (`agent_err_*`) | Yes | No | No |

**Counter, Gauge, Histogram** - as implemented by the Prometheus exporter

- **Counter**: Instance lifetime count of the related value. Summed over the separate events' values. Counter metrics have suffix '_total'
- Error events are exposed as one labeled counter: `agent_errors_total{status="..."}`. The `status` label is bounded by the fixed `AGENT_ERR_*` event set.
- **Gauge**: Shows the value per the last event (transaction) and can grow or decrease as each event updates it. The byte gauges follow the `agent_<subject>_last_<unit>` convention (the `_last` qualifier precedes the unit, keeping it distinct from the cumulative `_total` counter of the same base name): `agent_tx_last_bytes` / `agent_rx_last_bytes` carry the byte size of the latest TX/RX request, while `agent_tx_bytes_total` / `agent_rx_bytes_total` carry the running total. The memory gauges follow the same convention -- `agent_memory_registered_last_bytes` / `agent_memory_deregistered_last_bytes` -- and report the byte size of the last (de)registration, distinct from the cumulative `agent_memory_registered_total` / `agent_memory_deregistered_total` counters.
- **Histogram**: Counts the number of observations per pre-defined bins. Please see [Prometheus histograms documentation](https://prometheus.io/docs/practices/histograms/) for more details.

### Metric labels

Each telemetry metrics is provided with the following labels:

- Hostname where the agent runs
- Agent name (as custom provided during initialization, can be deprecated in the next versions)
