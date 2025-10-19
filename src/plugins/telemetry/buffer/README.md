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

# NIXL Buffer Telemetry exporter plug-in

This telemetry exporter plug-in exports NIXL telemetry events in shared memory buffers that can be read by telemetry reader applications.

## Configuration

To enable the Buffer plug-in, set the following environment variables:

```bash
export NIXL_TELEMETRY_ENABLE="y" # Enable NIXL telemetry
export NIXL_TELEMETRY_EXPORTER="buffer" # Sets which plug-in to select in format libtelemetry_exporter_${NIXL_TELEMETRY_EXPORTER}.so
export NIXL_TELEMETRY_EXPORTER_PLUGIN_DIR="path/to/dir/with/.so/files" # Sets where to find plug-in libs
export NIXL_TELEMETRY_EXPORTER_OUTPUT_PATH="path/to/store/telemetry/file/" # Sets where shared memory will be located
```

## Telemetry File Format

Telemetry data is stored in shared memory files with the agent name passed when creating the agent.

## Using Telemetry Readers

### C++ Telemetry Reader

The C++ telemetry reader (`telemetry_reader.cpp`) provides a robust way to read and display telemetry events.

#### Running the C++ Reader

```bash
# Read from a specific telemetry file
./builddir/examples/cpp/telemetry_reader /tmp/agent_name
```

### Python Telemetry Reader

The Python telemetry reader (`telemetry_reader.py`) provides similar functionality with additional features.

#### Running the Python Reader

```bash
# Read from a specific telemetry file
python3 examples/python/telemetry_reader.py --telemetry_path /tmp/agent_name
```

## Example Output

Both readers produce similar formatted output:

```
=== NIXL Telemetry Event ===
Timestamp: 2025-01-15 14:30:25.123456
Category: TRANSFER
Event: agent_tx_bytes
Value: 1048576
===========================

=== NIXL Telemetry Event ===
Timestamp: 2025-01-15 14:30:25.124567
Category: MEMORY
Event: agent_memory_registered
Value: 4096
===========================
```
