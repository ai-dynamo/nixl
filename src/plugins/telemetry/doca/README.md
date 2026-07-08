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

When multiple agents run in the same process, the first agent to initialize creates the DOCA server and its port/address settings take effect. Subsequent agents share that endpoint and are distinguished by the `agent_name` label.

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
| `agent_xfer_time` | Yes | Yes | Yes |
| `agent_xfer_post_time` | Yes | Yes | Yes |
| `agent_telemetry_events_dropped` | Yes | No | No |
| Error event types (`agent_err_*`) | Yes | No | No |

**Counter, Gauge, Histogram** - as implemented by the DOCA Telemetry Exporter

- **Counter**: Instance lifetime count of the related value. Summed over the separate events' values. Counter metrics have the `_total` suffix. The native Prometheus and DOCA exporters emit identical series (same names, types, labels) from one shared metric descriptor.
- Error events are exposed as one labeled counter: `agent_errors_total{status="..."}`. The `status` label is bounded by the fixed `AGENT_ERR_*` event set.
- `agent_telemetry_events_dropped_total` is the cumulative count of telemetry events dropped at the producer-side staging queue (when the queue is full and an event cannot be enqueued for export). It does not count BUFFER cyclic-ring loss. Emitted through the standard counter path (identical to `agent_tx_bytes`), not any DOCA-native "dropped metrics" feature.
- **Gauge**: Shows the value per the last event (transaction) and can grow or decrease as each event updates it. The byte events publish both a cumulative counter and a last-operation gauge: `agent_tx_bytes_total` / `agent_rx_bytes_total` carry the running total, while `agent_tx_last_bytes` / `agent_rx_last_bytes` (the `agent_<subject>_last_<unit>` convention) carry the byte size of the latest TX/RX request. The memory gauges follow the same convention -- `agent_memory_registered_last_bytes` / `agent_memory_deregistered_last_bytes` -- and report the byte size of the last (de)registration. The transfer-time events likewise publish both a cumulative `_total` counter and a last-operation gauge (`agent_xfer_time` / `agent_xfer_post_time`).
- **Histogram**: Counts the number of observations per pre-defined bins. Please see [Prometheus histograms documentation](https://prometheus.io/docs/practices/histograms/) for more details. The transfer-time events additionally publish latency-distribution histograms `agent_xfer_time_us` and `agent_xfer_post_time_us` (microseconds), exposed as the usual `_bucket{le="..."}` / `_sum` / `_count` series alongside the existing counter and gauge, at parity with the native Prometheus exporter. Bucket boundaries default to a microsecond range covering ~10us..~10s and can be overridden (see below).

### Histogram buckets

The default bucket boundaries for the transfer-time histograms can be overridden with a comma-separated list of strictly-increasing positive microsecond upper bounds (shared with the native Prometheus exporter):

```bash
export NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US="10,100,1000,10000,100000"
```

An absent, empty, or invalid value falls back to the built-in defaults.

### Metric labels

Each telemetry metric is provided with the following labels:

- Hostname where the agent runs
- Agent name (as provided during initialization, may be deprecated in future versions)
- `status` (only on `agent_errors_total`): the error kind, bounded by the fixed `AGENT_ERR_*` event set
