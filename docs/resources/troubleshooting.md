---
title: Troubleshooting
description: Common errors, debugging steps, and solutions for NIXL build, runtime, and transfer issues.
---

This guide helps you diagnose and resolve common issues when building, configuring, and using NIXL. Start with the [Debug Logging](#debug-logging) section to enable detailed output, then find your issue category below.

## Debug Logging

NIXL uses a configurable logging system that can help diagnose issues. Set the `NIXL_LOG_LEVEL` environment variable to increase verbosity:

```bash
export NIXL_LOG_LEVEL=DEBUG  # or TRACE for maximum detail
```

The available log levels are:

| Level | Description |
|-------|-------------|
| `ERROR` | Critical errors preventing normal operation |
| `WARN` | Warning conditions, recoverable errors (default) |
| `INFO` | Normal operation events |
| `DEBUG` | Detailed debugging information |
| `TRACE` | Very detailed trace for deep debugging |

<Tip>
See [Environment Variables](environment-variables#core-variables) for the full `NIXL_LOG_LEVEL` reference and all other NIXL configuration knobs.
</Tip>

## Error Codes Reference

NIXL uses integer status codes defined in `nixl_types.h`. Understanding these codes helps you quickly identify the root cause of failures:

| Code | Value | Meaning |
|------|-------|---------|
| `NIXL_IN_PROG` | 1 | Transfer is in progress (not an error -- check again later) |
| `NIXL_SUCCESS` | 0 | Operation completed successfully |
| `NIXL_ERR_NOT_POSTED` | -1 | Transfer was not posted; call `postXferReq` before checking status |
| `NIXL_ERR_INVALID_PARAM` | -2 | Invalid parameter passed to API; check argument types and ranges |
| `NIXL_ERR_BACKEND` | -3 | Backend-level failure; check backend logs and connectivity |
| `NIXL_ERR_NOT_FOUND` | -4 | Requested resource not found; verify agent name and memory registration |
| `NIXL_ERR_MISMATCH` | -5 | Type or configuration mismatch between source and destination |
| `NIXL_ERR_NOT_ALLOWED` | -6 | Operation not permitted in current state; check operation ordering |
| `NIXL_ERR_REPOST_ACTIVE` | -7 | Attempted to repost a transfer that is still active |
| `NIXL_ERR_UNKNOWN` | -8 | Unclassified error; enable DEBUG logging for details |
| `NIXL_ERR_NOT_SUPPORTED` | -9 | Operation not supported by this backend; check backend capabilities |
| `NIXL_ERR_REMOTE_DISCONNECT` | -10 | Remote agent disconnected; verify network connectivity |
| `NIXL_ERR_CANCELED` | -11 | Operation was canceled |
| `NIXL_ERR_NO_TELEMETRY` | -12 | Telemetry not enabled; set `NIXL_TELEMETRY_ENABLE=true` |

## Build and Installation Issues

### Meson Setup Fails with Missing Dependencies

Ensure all required build tools are installed:
- C++17 compatible compiler (GCC 7+ or Clang 5+)
- Meson build system (`pip install meson`)
- Ninja build tool (`pip install ninja`)
- pkg-config

```bash
# Install on Ubuntu/Debian
sudo apt-get install build-essential cmake pkg-config

# Install Meson and Ninja
pip install meson ninja
```

<Tip>
See [Building NIXL from Source](../user-guide/building-nixl) for detailed source build instructions including Docker containers, or [Quick Start](../getting-started/quick-start) for PyPI installation.
</Tip>

### UCX Not Found During Build

UCX is required for most network backends. If Meson cannot find UCX:

```bash
# Specify UCX path explicitly
meson setup build -Ducx_path=/path/to/ucx

# Or install UCX system-wide
sudo apt-get install libucx-dev
```

<Warning>
UCX version compatibility matters. NIXL requires UCX 1.14+ for full feature support. Check your UCX version with `ucx_info -v`.
</Warning>

### Python Bindings Import Error

If you get `ImportError: libnixl.so: cannot open shared object file`:

```bash
# Option 1: Set library path
export LD_LIBRARY_PATH=/path/to/nixl/build:$LD_LIBRARY_PATH

# Option 2: Install via pip (recommended)
pip install nixl
```

### Plug-in Shared Library Not Found at Runtime

If NIXL cannot find backend plug-ins at runtime, set the plug-in directory:

```bash
export NIXL_PLUGIN_DIR=/path/to/nixl/plugins
```

The default plug-in directory is `{libdir}/plugins` relative to the NIXL installation.

## Runtime Errors

### Agent Initialization Fails (NIXL_ERR_BACKEND)

The backend library is not found or not properly configured:

1. Verify `NIXL_PLUGIN_DIR` points to the correct plug-in directory
2. Check that the backend's shared library exists (e.g., `libplugin_UCX.so`)
3. Verify backend prerequisites are installed (e.g., UCX libraries for the UCX backend)
4. Enable `NIXL_LOG_LEVEL=DEBUG` to see the exact plug-in loading error

### Memory Registration Fails (NIXL_ERR_INVALID_PARAM)

Invalid memory address or size was passed to `registerMem`:

1. Verify the memory allocation succeeded before registering
2. Check that the memory type matches the backend's supported types (use `getSupportedMems()`)
3. Ensure the address and length are valid for the memory region

### Operation Not Allowed (NIXL_ERR_NOT_ALLOWED)

Operations must follow the correct sequence: initialize agent, register memory, exchange metadata, then transfer:

1. Verify you have registered memory before attempting transfers
2. Verify metadata has been exchanged (either side-channel, etcd, or programmatic) before transfer
3. Check that the transfer handle is in the correct state

## Transfer Failures

### Transfer Stays in NIXL_IN_PROG

Transfers are asynchronous. If a transfer appears stuck:

1. Continue polling with `getXferStatus()` -- some transfers take time depending on data size
2. Check network connectivity between the source and destination agents
3. Verify the remote agent is still running and healthy
4. Enable `NIXL_LOG_LEVEL=DEBUG` to see backend-level transfer progress

### Type Mismatch (NIXL_ERR_MISMATCH)

Source and destination descriptors are incompatible:

1. Verify both sides registered compatible memory types
2. Check that descriptor sizes match between source and destination
3. Ensure the transfer operation type (read/write) is correct for the direction

### Backend Error During Transfer (NIXL_ERR_BACKEND)

A backend-specific failure occurred during the transfer:

1. Enable `NIXL_LOG_LEVEL=DEBUG` to see detailed backend error messages
2. Check backend-specific logs and connectivity
3. Verify backend configuration (see Backend-Specific Issues below)

### Remote Disconnect (NIXL_ERR_REMOTE_DISCONNECT)

The remote agent became unreachable during the operation:

1. Check network connectivity to the remote host
2. Verify the remote agent process is still running
3. Check firewall rules between the two hosts
4. For RDMA backends, verify InfiniBand/RoCE connectivity with `ibv_devinfo`

## Backend-Specific Issues

<Note>
See [Backend Selection](../user-guide/backend-selection) for backend requirements and memory type compatibility.
</Note>

### UCX

**Connection timeout:** Verify RDMA devices are available with `ibv_devinfo`. Check that both hosts can reach each other on the RDMA network.

**Memory registration warning after 5s:** If UCX logs a warning about slow memory registration, you can adjust the timeout:

```bash
export NIXL_UCX_WARNING_TIMEOUT=10000  # milliseconds
```

Alternatively, verify that the device supports the memory type being registered.

### Libfabric

**Provider selection issues:** Check available providers with `fi_info`. Ensure the correct provider is being selected for your network fabric.

**CUDA address workaround:** If you encounter issues with CUDA memory addresses on certain providers:

```bash
export NIXL_DISABLE_CUDA_ADDR_WA=1  # Disable the workaround if it causes issues
```

### GDS (GPUDirect Storage)

**cuFile initialization fails:** Verify the GDS driver is loaded and the cuFile configuration is accessible:

1. Check that `CUFILE_ENV_PATH_JSON` points to a valid cuFile configuration
2. Verify the filesystem is ext4 or XFS (GDS requirement)
3. Ensure the GDS kernel module is loaded: `lsmod | grep nvidia_fs`

### S3 / Object Storage

**Authentication errors:** Verify your credentials are set:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

**Custom endpoint:** For non-AWS S3-compatible storage (MinIO, Ceph, etc.):

```bash
export AWS_ENDPOINT_OVERRIDE=http://your-s3-endpoint:9000
```

### Azure Blob Storage

**Connection fails:** Verify your storage account URL is set:

```bash
export AZURE_STORAGE_ACCOUNT_URL=https://youraccount.blob.core.windows.net
```

For local testing with Azurite:

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=...;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
```

## etcd Connection Issues

<Tip>
See [Metadata Exchange with etcd](../user-guide/etcd-metadata-exchange) for the full etcd setup and configuration guide.
</Tip>

### etcd Not Connecting

Verify the etcd endpoint is set and the server is reachable:

```bash
# Set the endpoint
export NIXL_ETCD_ENDPOINTS="http://localhost:2379"

# Test connectivity
etcdctl --endpoints=http://localhost:2379 endpoint health
```

### Metadata Not Found for Remote Agent

The remote agent may not have published its metadata yet:

1. Check that the remote agent has called `sendLocalMD()` after registering memory
2. Verify both agents use the same etcd namespace (`NIXL_ETCD_NAMESPACE`, default: `/nixl/agents/`)
3. Inspect etcd keys directly: `etcdctl get --prefix /nixl/agents/`

### Stale Metadata After Agent Restart

When an agent restarts, its previous metadata may still be in etcd:

1. Agents overwrite their metadata on republish via `sendLocalMD()`
2. Remote agents with cached metadata will be notified via the watcher mechanism
3. If issues persist, manually clear the agent's keys: `etcdctl del --prefix /nixl/agents/{agent_name}/`

<Warning>
After clearing an agent's etcd keys, all remote agents that previously fetched its metadata will need to re-fetch it.
</Warning>
