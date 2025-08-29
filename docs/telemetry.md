# NIXL Telemetry System

## Overview

The NIXL telemetry system provides real-time monitoring and performance tracking capabilities for NIXL applications. It collects various metrics and events during runtime and stores them in shared memory buffers that can be read by telemetry reader applications.

## Architecture

### Telemetry Components

1. **Telemetry Collection**: Built into the NIXL core library, collects events and metrics
2. **Shared Memory Buffer**: Cyclic buffer implementation for efficient event storage
3. **Telemetry Readers**: C++ and Python applications to read and display telemetry data

### Event Structure

Each telemetry event contains:
- **Timestamp**: Microsecond precision timestamp
- **Category**: Event category for filtering and aggregation
- **Event Name**: Descriptive name/identifier for the event
- **Value**: Numeric value associated with the event

### Event Categories

The telemetry system supports the following event categories:

| Category | Description | Example Events |
|----------|-------------|----------------|
| `NIXL_TELEMETRY_MEMORY` | Memory operations | Memory registration, deregistration, allocation |
| `NIXL_TELEMETRY_TRANSFER` | Data transfer operations | Bytes transmitted/received, request counts |
| `NIXL_TELEMETRY_CONNECTION` | Connection management | Connect, disconnect events |
| `NIXL_TELEMETRY_BACKEND` | Backend-specific operations | Backend initialization, configuration |
| `NIXL_TELEMETRY_ERROR` | Error events | Error counts by type |
| `NIXL_TELEMETRY_PERFORMANCE` | Performance metrics | Transaction times, latency measurements |
| `NIXL_TELEMETRY_SYSTEM` | System-level events | Process start/stop, resource usage |
| `NIXL_TELEMETRY_CUSTOM` | Custom/user-defined events | Application-specific metrics |

## Enabling Telemetry

### Runtime Configuration

Telemetry is controlled by environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NIXL_TELEMETRY_ENABLE` | Enable telemetry collection | Disabled |
| `NIXL_TELEMETRY_DIR` | Directory for telemetry files | - |
| `NIXL_TELEMETRY_BUFFER_SIZE` | Number of events in buffer | `4096` |
| `NIXL_TELEMETRY_RUN_INTERVAL` | Flush interval (ms) | `100` |

- NIXL_TELEMETRY_ENABLE can be set to y/yes/on/1 to be enabled, and n/no/off/0 (or not set) to be disabled,
- If NIXL_TELEMETRY_ENABLE is set to enabled but NIXL_TELEMETRY_DIR is not set, no telemetry file is generated and NIXL_TELEMETRY_RUN_INTERVAL is not used.

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
