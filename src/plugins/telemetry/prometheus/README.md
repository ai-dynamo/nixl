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

This telemetry exporter plug-in exports NIXL telemetry events in Prometheus format, by_exposing an HTTP endpoint that can be scraped by Prometheus servers.

## Dependencies

The Prometheus exporter requires the prometheus-cpp library, which is included as a subproject:

libcurl is not handled by prometheus-cpp libaray. need to install the libcurl package:

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
export NIXL_PLUGIN_DIR="path/to/dir/with/.so/files" # Sets where to find plug-in libs (NOTE: the same var is used for backend plug-ins search)
```

### Optional Configuration

You can configure the exposed prometheus port:

```bash
# Default port is 9090
export NIXL_TELEMETRY_PROMETHEUS_PORT="<port_num>"
```

Default addres is public, but you configure to expose prometheus endpoint only locally:

```bash
export NIXL_TELEMETRY_PROMETHEUS_LOCAL="y"
# May also use "yes" or "1"
```

### Transfer Metrics (Counters)
- `agent_tx_bytes` - Total bytes transmitted
- `agent_rx_bytes` - Total bytes received
- `agent_tx_requests_num` - Number of transmit requests
- `agent_rx_requests_num` - Number of receive requests

### Performance Metrics (Gauges)
- `agent_xfer_time` - Transfer time in microseconds
- `agent_xfer_post_time` - Post time in microseconds

### Memory Metrics (Gauges)
- `agent_memory_registered` - Amount of memory registered
- `agent_memory_deregistered` - Amount of memory deregistered

### Backend Metrics (Dynamic Counters)
- Backend-specific events are dynamically created as counters with category label

