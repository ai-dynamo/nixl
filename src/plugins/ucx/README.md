# NIXL UCX Plugin

UCX is the default network backend for the NIXL Python agent. It supports
transfers within an agent and between agents on the same node or across nodes,
using UCX transports such as RDMA, shared memory, and TCP. See the
[UCX user guide](../../../fern/docs/pages/user-guide/backends/ucx.md).

## Overview

- **Memory types** include host memory (`DRAM_SEG`) and GPU memory (`VRAM_SEG`)
  supported by the UCX context.
- **Capabilities** include local and remote transfers and notifications.
  Transfer notifications can carry sampled trace contexts.
- **Connections** use one endpoint per local worker for each peer. NIXL
  exchanges a packed remote key (rkey) for each registered region. The
  receiver stores one unpacked rkey per endpoint for that region.
- **GPU memory views** prepared with `prep_mem_view` let CUDA kernels initiate
  transfers. The `UCX_DIRECT` and `PROXY` device execution modes both require
  a loaded NIXL device operations implementation. `UCX_DIRECT` also requires
  UCX GPU device API support at build time.

## Engine Variants

Backend creation checks these conditions in order.

| Condition | Engine | Description |
| --------- | ------ | ----------- |
| `device_proxy=true` | proxy engine | The GPU executes `PUT` directly when both buffers have direct pointers, and `ATOMIC_ADD` when the counter has one. If a required pointer is missing, host threads relay the command through UCX using per-channel, per-peer rings. |
| `num_threads > 0` | thread pool engine | Each transfer thread drives a dedicated UCX worker. Batches reaching `split_batch_size` are divided across those workers. |
| progress thread enabled (Python default) | thread engine | One background thread advances all shared UCX workers and processes incoming notifications. |
| progress thread disabled (C++ default) | base engine | Polling transfer status or notifications drives progress on the calling thread. |

With `device_proxy=true`, disable the agent progress thread
(`enable_prog_thread=False` in Python, `useProgThread=false` in C++) and leave
`num_threads` at `0`. Omit `ucx_error_handling_mode` to use `none`, or set it
to `none` explicitly. The proxy constructor throws if you pass the plugin's
advertised default of `peer`.

## Dependencies

- For the tested **UCX** version and build instructions, see the
  [root README](../../../README.md#ucx).
- **GDRCopy** is optional and recommended for CUDA GPU transfers.
- For **CUDA** or **ROCm** buffers, build UCX with `--with-cuda` or
  `--with-rocm`, respectively.

## Backend Initialization Options

`get_plugin_params("UCX")` in Python and `getPluginParams("UCX", ...)` in C++
advertise defaults for `ucx_devices`, `num_workers`, `ucx_error_handling_mode`,
and `ucx_vram_memtype_hint`. The engine ignores
`ucx_devices`. Pass `device_list` to `create_backend` or `createBackend` to
select network devices.

| Option | Default | Description |
| ------ | ------- | ----------- |
| `num_workers` | `1` | Total shared and dedicated workers. The engine keeps at least one shared worker and assigns shared workers to calling threads round-robin by default. |
| `device_list` | unset (UCX device selection) | Separate bare device names with the exact delimiter `", "` (comma and space). NIXL appends `:1` to each item when setting `NET_DEVICES`. |
| `engine_config` | *(empty)* | Comma-separated UCX `KEY=VALUE` settings, such as `MAX_RMA_RAILS=1`. |
| `ucx_num_device_channels` | `4` | Sets the default for `RC_GDA_NUM_CHANNELS` with UCX 1.21 or newer. |
| `num_threads` | `0` | Number of dedicated transfer threads (enables the thread pool engine). |
| `split_batch_size` | `4096` with SGL enabled, otherwise `1024` | Minimum descriptor count for splitting a transfer across dedicated workers. The effective threshold is at least `num_threads`. |
| `ucx_error_handling_mode` | `peer` (proxy uses `none`) | `peer` requests endpoint failure reporting. `none` permits transports that lack peer failure reporting. See the [backend guide](../../../docs/BackendGuide.md#ucx-backend-initialization-options). |
| `ucx_vram_memtype_hint` | `auto` | VRAM registration hint policy. Accepted values are `auto`, `none`, `cuda`, `cuda-managed`, `rocm`, and `ze-device`. Matching is case-sensitive. |
| `device_proxy` | `false` | Enables the `proxy_*` options. The proxy uses one worker per ring (`proxy_channel_count * proxy_max_peers`) and rejects larger `num_workers` values. |
| `proxy_channel_count` | `4` | Positive number of proxy channels. |
| `proxy_thread_count` | `proxy_channel_count` | Positive number of CPU progress threads. It must divide the channel count or be a multiple of the channel count that divides the total ring count. |
| `proxy_max_peers` | `8` | Positive peer capacity. Each channel provisions one ring per peer slot. |
| `proxy_ring_depth` | `256` | Slots per ring. Must be a nonzero power of two. |
| `proxy_drain_timeout_ms` | `60000` | A drain that reaches this deadline aborts the process. Set `0` to wait without a deadline. |

For example, `device_list="mlx5_0, mlx5_1"` sets `NET_DEVICES` to
`mlx5_0:1,mlx5_1:1`. With `"mlx5_0,mlx5_1"`, NIXL sees one item and passes
`mlx5_0,mlx5_1:1` to UCX.

NIXL applies `device_list` before `engine_config`. If both
`NIXL_UCX_NET_DEVICES` and `UCX_NET_DEVICES` are unset, a `NET_DEVICES` entry
in `engine_config` overrides `device_list`. If either variable is set, NIXL
skips that entry and an explicit `device_list` overrides the environment.

## Environment Variables

The four plugin settings below also accept their former `NIXL_UCX_` names
when the corresponding `NIXL_PLUGIN_UCX_` setting is absent. Using a former
name as a fallback logs a deprecation warning.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `NIXL_PLUGIN_UCX_SGL_ENABLE` | `true` with UCX SGL support, otherwise `false` | Enables scatter-gather list (SGL) offload for writes on builds with SGL support. |
| `NIXL_PLUGIN_UCX_WARNING_TIMEOUT` | `5000` ms | Warning interval while NIXL retries remote device memory-list creation on `UCS_ERR_NOT_CONNECTED`. Use integer milliseconds, such as `10000`. |
| `NIXL_PLUGIN_UCX_EXPECTED_SONAME` | unset | A nonempty value must appear in the library path for `ucp_get_version_string` or plugin initialization fails. |
| `NIXL_PLUGIN_UCX_DEEPBIND` | `false` | Requests `RTLD_DEEPBIND` for the dynamic UCX plugin on platforms that support it. An invalid boolean value logs a warning and enables the flag. |

For UCX protocol and transport settings, NIXL checks `NIXL_UCX_<X>` before
`UCX_<X>`. For example, `NIXL_UCX_TLS` overrides `UCX_TLS` for NIXL's UCX
context. Either environment setting takes precedence over NIXL tuning defaults
and matching `engine_config` entries. Process-wide settings such as
`UCX_LOG_LEVEL` use the global `UCX_` prefix.

## Building

With the default plugin selection, Meson builds the plugin when it finds UCX.
Point the build at a custom UCX installation with
`meson setup <build_dir> -Ducx_path=/path/to/ucx`. See the
[UCX build options](../../../fern/docs/pages/user-guide/backends/ucx.md#build-options).
UCX SGL support is detected at compile time (it requires the
`ucp_dt_local_sgl_t` type) and enables the write-path SGL offload by default
(`NIXL_PLUGIN_UCX_SGL_ENABLE`).

## Example Usage

```python
from nixl import nixl_agent, nixl_agent_config

# Create the agent without backends, then instantiate UCX with parameters
agent = nixl_agent("agent", nixl_agent_config(backends=[]))
params = agent.get_plugin_params("UCX").copy()
params["num_workers"] = "4"
agent.create_backend("UCX", params)
```

For the default UCX configuration, use `nixl_agent("agent")`.

For transfers between two agents, exchange metadata (directly or through a
metadata service), then follow the standard NIXL transfer flow — see
[examples/python/basic_two_peers.py](../../../examples/python/basic_two_peers.py).

## Tests

- Intra-agent and inter-agent transfers — [ucx_backend_test.cpp](../../../test/unit/plugins/ucx/ucx_backend_test.cpp)
- One-sided connection and disconnection between two agents — [ucx_backend_multi.cpp](../../../test/unit/plugins/ucx/ucx_backend_multi.cpp)
- Endpoint reads, writes, and flushes with two workers — [ucx_worker_test.cpp](../../../test/unit/plugins/ucx/ucx_worker_test.cpp)
- Trace-context propagation and hardware warnings — [ucx_trace_context_test.cpp](../../../test/gtest/ucx_trace_context_test.cpp) and [hw_warning_test.cpp](../../../test/gtest/hw_warning_test.cpp)
