# NIXL Telemetry System

## Overview

The NIXL telemetry system provides real-time monitoring and performance tracking capabilities for NIXL applications. It collects various metrics and events during runtime and exports them through customizable plug-ins

## Architecture

### Telemetry Components

1. **Telemetry Collection**: Built into the NIXL core library, collects events and metrics
2. **Shared Memory Buffer exporter**: Cyclic buffer exporter implementation for efficient event storage. [See Buffer Exporter README](../src/plugins/telemetry/buffer/README.md)
3. **Telemetry Readers**: C++ and Python applications to read and display telemetry data
4. **Prometheus exporter**: Prometheus compitable exporter. [See Prometheus Exporter README](../src/plugins/telemetry/prometheus/README.md)

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
| `NIXL_TELEMETRY_BUFFER_SIZE` | Number of events in buffer | `4096` |
| `NIXL_TELEMETRY_RUN_INTERVAL` | Flush interval (ms) | `100` |
| `NIXL_TELEMETRY_EXPORTER_PLUGIN_DIR` | Directory where to look for exporter plug-in | - |
| `NIXL_TELEMETRY_EXPORTER` | Plugin {name}. Format for plugin lib is libtelemetry_exporter_{name}.so | - |
| `NIXL_TELEMETRY_EXPORTER_OUTPUT_PATH` | Implementation specific configuration field. (e.g external DB address, path to file) | - |

- NIXL_TELEMETRY_ENABLE can be set to y/yes/on/1 to be enabled, and n/no/off/0 (or not set) to be disabled,
