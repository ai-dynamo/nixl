---
title: Environment Variables
description: Complete reference for all NIXL environment variables covering core configuration, etcd, telemetry, and backend-specific settings.
---

## Overview

NIXL uses environment variables for runtime configuration. This page documents all recognized variables, grouped by subsystem. Environment variables are read at agent initialization time and cannot be changed after an agent is created.

Variables are organized into the following categories:

- **Core** -- Logging and plug-in discovery that apply to all NIXL deployments
- **etcd** -- Distributed metadata exchange configuration
- **Telemetry** -- Performance monitoring, event collection, and export
- **Backend-Specific** -- Transport-level settings for individual backends (UCX, Libfabric, Mooncake, S3, Azure Blob, GDS)

<Note>
Boolean environment variables follow the convention: set to `y`/`yes`/`on`/`true`/`enable`/`1` (case-insensitive) to enable, or `n`/`no`/`off`/`false`/`disable`/`0` to disable. Presence-check variables are activated by setting them to any value.
</Note>

## Core Variables

These variables control fundamental NIXL behavior across all backends.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NIXL_LOG_LEVEL` | String | `WARN` | Controls log verbosity. Values: `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`. |
| `NIXL_PLUGIN_DIR` | String (path) | System default | Custom directory to search for backend plug-in shared libraries. |
| `NIXL_DISABLE_CUDA_ADDR_WA` | Boolean (presence) | Not set (workaround enabled) | Disables CUDA address workaround in the Libfabric backend. Set this variable to any value to disable the workaround. |

## etcd Variables

These variables configure NIXL's distributed metadata exchange via etcd.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NIXL_ETCD_ENDPOINTS` | String (URL) | None (etcd disabled) | etcd server endpoint(s). Setting this activates etcd metadata exchange mode. Example: `http://localhost:2379`. |
| `NIXL_ETCD_NAMESPACE` | String (path) | `/nixl/agents/` | Key prefix namespace for agent metadata stored in etcd. |

<Tip>
For detailed etcd setup and usage, see the [Metadata Exchange with etcd](../user-guide/etcd-metadata-exchange) guide.
</Tip>

## Telemetry Variables

These variables control NIXL's built-in telemetry collection and export system.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NIXL_TELEMETRY_ENABLE` | Boolean | `false` | Enable telemetry collection. Accepts `y`/`yes`/`on`/`true`/`enable`/`1` (case-insensitive) to enable. |
| `NIXL_TELEMETRY_BUFFER_SIZE` | Integer | `4096` | Number of events in the cyclic telemetry buffer. Must be a power of 2. |
| `NIXL_TELEMETRY_RUN_INTERVAL` | Integer (ms) | `100` | Flush interval in milliseconds for the telemetry exporter. |
| `NIXL_TELEMETRY_EXPORTER` | String | None | Name of the telemetry exporter plug-in to load (e.g., `prometheus`). |
| `NIXL_TELEMETRY_DIR` | String (path) | None | Directory for shared memory telemetry files used by the cyclic buffer exporter. If telemetry is enabled but this is not set, no telemetry file is generated. |
| `NIXL_TELEMETRY_PROMETHEUS_PORT` | Integer | `9090` | HTTP listen port for the Prometheus telemetry exporter. |
| `NIXL_TELEMETRY_PROMETHEUS_LOCAL` | Boolean | `false` | When `true`, binds the Prometheus exporter to localhost only (`127.0.0.1`) instead of all interfaces (`0.0.0.0`). |

<Tip>
For the full telemetry architecture and usage examples, see the [Telemetry Guide](../user-guide/telemetry).
</Tip>

## Backend-Specific Variables

### UCX

<Note>
These variables apply when using the UCX transport backend.
</Note>

<Markdown src="/snippets/env-vars-ucx.mdx" />

### Libfabric

<Note>
These variables apply when using the Libfabric transport backend.
</Note>

<Markdown src="/snippets/env-vars-libfabric.mdx" />

### Mooncake

<Note>
These variables apply when using the Mooncake transport backend.
</Note>

<Markdown src="/snippets/env-vars-mooncake.mdx" />

### S3 / Object Storage

<Note>
These variables apply when using the S3/Object Storage backend.
</Note>

<Markdown src="/snippets/env-vars-obj.mdx" />

### Azure Blob Storage

<Note>
These variables apply when using the Azure Blob Storage backend.
</Note>

<Markdown src="/snippets/env-vars-azure-blob.mdx" />

### GDS (GPUDirect Storage)

<Note>
These variables apply when using the GDS backend.
</Note>

<Markdown src="/snippets/env-vars-gds.mdx" />

## Quick Reference

Set environment variables before launching your NIXL application:

```bash
# Enable debug logging and telemetry with Prometheus export
export NIXL_LOG_LEVEL=DEBUG
export NIXL_TELEMETRY_ENABLE=true
export NIXL_TELEMETRY_EXPORTER=prometheus
export NIXL_TELEMETRY_PROMETHEUS_PORT=9090

# Configure etcd for distributed metadata exchange
export NIXL_ETCD_ENDPOINTS=http://localhost:2379

# Run your NIXL application
./my_nixl_app
```
