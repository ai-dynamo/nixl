# NIXL IBM Storage Scale Plugin

This backend enables high-performance file I/O for IBM Storage Scale (formerly GPFS)
filesystems using Linux `io_uring` for asynchronous, low-overhead data transfer.
The plugin registers under the name `IBM_SCALE` and supports `FILE_SEG` and `DRAM_SEG`
memory types.

## Features

- **Per-request `io_uring` rings**: each transfer handle owns its own ring, enabling
  fully parallel concurrent transfers with no mutex contention.
- **Descriptor coalescing**: consecutive page-aligned descriptors are merged at
  filesystem block boundaries, collapsing thousands of 4 KiB SQEs into a handful of
  block-aligned reads/writes (e.g. 2560 → 2 SQEs for a 10 MiB file on 8 MiB blocks).
- **Short-I/O handling**: `checkXfer` detects partial completions and re-submits the
  remaining bytes automatically.
- **Synchronous fallback**: if `io_uring_queue_init()` fails (e.g. `RLIMIT_MEMLOCK`),
  the plugin falls back to `pread()`/`pwrite()` transparently.
- **Optional GPFS hints**: when built with `-Dibm_scale_path=` pointing to an IBM
  Storage Scale installation that provides `gpfs_fcntl.h`, the plugin can fire
  `GPFS_PREFETCH` and `GPFS_ACCESS_RANGE` hints at `registerMem` time to warm the
  GPFS client cache ahead of I/O.

## Dependencies

- **liburing** (required): detected automatically by the top-level meson build.
- **IBM Storage Scale client libraries** (optional): provide `gpfs_fcntl.h` and
  `libgpfs.so` for GPFS hint support.

## Build Configuration

```bash
# Minimal build (io_uring only, no GPFS hints)
meson setup build
ninja -C build

# Build with GPFS hints (requires IBM Storage Scale client installed)
meson setup build -Dibm_scale_path=/usr/lpp/mmfs
ninja -C build
```

## Configuration

Parameters are passed as a `nixl_b_params_t` map when calling `createBackend("IBM_SCALE", ...)`,
or as environment variables for benchmark tools that do not populate `customParams`.

### Backend Parameters

| Parameter | Environment variable | Description | Default |
|-----------|---------------------|-------------|---------|
| `nixl_scale_ring_size` | `NIXL_SCALE_RING_SIZE` | `io_uring` queue depth per request | `128` |
| `nixl_scale_disable_mar_hints` | `NIXL_SCALE_DISABLE_MAR_HINTS` | `1` = skip per-descriptor GPFS `MULTIPLE_ACCESS_RANGE` hints | `1` |

### Priority order (highest to lowest)

1. `nixl_b_params_t` (programmatic API)
2. Environment variables
3. Built-in defaults

### Usage example (LMCache)

```python
backend_params = {
    "file_path": "/gpfs/fs1/lmcache",
    "use_direct_io": "false",
    "nixl_scale_ring_size": "128",
}
agent.create_backend("IBM_SCALE", backend_params)
```

## Transfer Model

The plugin is local-only (`supportsRemote() = false`). The NIXL transfer model maps to:

- `FILE_SEG` → on-disk files (remote descriptors)
- `DRAM_SEG` → host memory buffers (local descriptors)
- `NIXL_READ` → file → DRAM (load from storage)
- `NIXL_WRITE` → DRAM → file (store to storage)

### Sequence

1. **`registerMem`** (FILE_SEG): opens the file (path-mode or fd passthrough),
   samples the filesystem block size via `fstatfs()`, allocates `nixlScaleFileMD`.
2. **`prepXfer`**: validates parameters, coalesces descriptors, allocates the
   per-request `io_uring` ring.
3. **`postXfer`**: submits all SQEs; returns `NIXL_IN_PROG`.
4. **`checkXfer`**: polls CQEs non-blocking; handles short I/O by re-submitting;
   returns `NIXL_SUCCESS` when all descriptors are complete.
5. **`releaseReqH`**: frees the request handle and its `io_uring` ring.
6. **`deregisterMem`**: frees the file metadata (closes owned fds).
