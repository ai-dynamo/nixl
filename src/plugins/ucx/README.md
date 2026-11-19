# NIXL UCX Plugin

This plugin is the primary network backend of NIXL. It provides peer-to-peer
transfers between agents over UCX (RDMA, shared memory, and other UCX
transports), covering host memory (`DRAM_SEG`) and GPU memory (`VRAM_SEG`),
for both remote (cross-node) and local (same-node) transfers, with
notifications and trace-context propagation support.

## Overview

- **Memory types**: `DRAM_SEG`, `VRAM_SEG` (CUDA and ROCm devices, via the
  memory types supported by the UCX build)
- **Capabilities**: remote and local transfers, notifications, trace contexts
- **Connection model**: one UCX endpoint per (remote agent, worker) pair;
  packed remote keys (rkeys) are cached per endpoint and exchanged through
  NIXL metadata
- **GPU device API support**: when NIXL is built against a UCX with the GPU
  device API, memory views can be handed to CUDA kernels for device-initiated
  communication (see `prep_mem_view` in the Python API)

## Engine Variants

The engine implementation is selected at backend creation:

| Condition | Engine | Description |
| --------- | ------ | ----------- |
| `device_proxy=true` | proxy engine | Host CPU threads relay device-issued `PUT`/`ATOMIC_ADD` commands from per-peer rings to UCX operations. Used for device-initiated transfers when the UCX GPU device API is not available. |
| `num_threads > 0` | thread pool engine | Each thread drives a dedicated UCX worker; large transfers are split into chunks (`split_batch_size`) and submitted in parallel. |
| progress thread enabled (default agent configuration) | thread engine | A single progress thread advances all shared UCX workers and delivers notifications. |
| otherwise | base engine | Progress happens inside status checks on the calling thread. |

`num_threads` and `device_proxy` are mutually exclusive.

## Dependencies

- **UCX**: built and tested with UCX `v1.23.x`; see the
  [root README](../../../README.md#ucx) for build instructions. Standard
  `UCX_*` environment variables are honored; NIXL only sets values that were
  not already configured.
- **GDRCopy**: optional, recommended for maximum RDMA performance with CUDA
  GPUs. NIXL and UCX work without it.
- **CUDA or ROCm**: required for `VRAM_SEG` support (UCX must be built with
  `--with-cuda` or `--with-rocm` respectively).

## Backend Initialization Options

Options can be inspected with `get_plugin_params("UCX")` (Python) or
`getPluginParams("UCX", ...)` (C++) and passed to backend creation:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `num_workers` | `1` | Total number of UCX workers (shared plus dedicated). Shared workers are assigned round-robin to calling threads. |
| `device_list` | *(all devices)* | Comma-separated list of UCX device names to restrict the context to. |
| `engine_config` | *(empty)* | Extra UCX engine configuration string. |
| `ucx_num_device_channels` | `4` | Number of channels per device for the UCX GPU device API. |
| `num_threads` | `0` | Number of dedicated transfer threads (enables the thread pool engine). |
| `split_batch_size` | `4096` with UCX SGL, otherwise `1024` | Chunk size (number of descriptors) used to split large transfers across the thread pool; lower bound is `num_threads`. |
| `ucx_error_handling_mode` | `peer` | UCX endpoint error handling policy: `peer` or `none`. See the [backend guide](../../../docs/BackendGuide.md) for the trade-offs; `none` trades remote failure visibility for transport availability. |
| `ucx_vram_memtype_hint` | `auto` | Memory type hint passed to UCX when registering VRAM: `auto`, `none`, `cuda`, `cuda-managed`, `rocm`, `ze-device`. Matching is case-sensitive. See the [backend guide](../../../docs/BackendGuide.md). |
| `device_proxy` | `false` | Enable the device proxy for device-initiated transfers. |
| `proxy_channel_count` | `4` | Number of proxy channels. |
| `proxy_thread_count` | `proxy_channel_count` | Number of proxy CPU progress threads. |
| `proxy_max_peers` | `8` | Maximum number of remote peers to provision rings for; set to the real peer capacity. |
| `proxy_ring_depth` | `256` | Ring slots per (channel, peer); must be a power of two. |
| `proxy_drain_timeout_ms` | `60000` | A proxy drain that takes longer aborts the process; `0` waits forever. |

## Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `NIXL_UCX_SGL_ENABLE` | `true` when NIXL is built with UCX SGL support, otherwise ignored | Toggle UCX scatter-gather list offload of multi-descriptor transfers. |
| `NIXL_UCX_WARNING_TIMEOUT` | `5000` ms | How long a memory-view request waits for a connection before logging a warning (duration with unit, e.g. `10s`). |
| `NIXL_UCX_EXPECTED_SONAME` | unset | If set, the backend fails to load unless it bound to a libucx whose path contains this substring; useful to catch wrong UCX installations. |
| `NIXL_UCX_DEEPBIND` | unset | Load the UCX plugin with `RTLD_DEEPBIND` to prefer plugin-local symbols. |
| `UCX_*` | — | Standard UCX environment variables; applied before NIXL defaults. |

## Building

The plugin is built automatically when UCX is found. Point the build at a
custom UCX installation with `meson setup <build_dir> -Ducx_path=/path/to/ucx`.
UCX SGL support is detected at compile time (it requires the
`ucp_dt_local_sgl_t` type) and enables the write-path SGL offload by default
(`NIXL_UCX_SGL_ENABLE`).

## Example Usage

```python
import nixl

# Create the agent without backends, then instantiate UCX with parameters
agent = nixl.nixl_agent(backends=[])
params = agent.get_plugin_params("UCX")   # advertised options and defaults
params["num_workers"] = "4"
agent.create_backend("UCX", params)

# Or simply use the default engine configuration
# agent = nixl.nixl_agent()               # UCX backend is created by default
```

For transfers between two agents, exchange metadata (directly or through a
metadata service), then follow the standard NIXL transfer flow — see
[examples/python/basic_two_peers.py](../../../examples/python/basic_two_peers.py).

## Tests

- `test/unit/plugins/ucx/ucx_backend_test` — backend behavior
- `test/unit/plugins/ucx/ucx_backend_multi` — multi-agent transfers
- `test/unit/plugins/ucx/ucx_worker_test` — worker management
- `test/gtest/` — tracing context propagation and hardware warning tests
