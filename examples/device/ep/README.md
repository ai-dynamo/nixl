# NIXL EP: Expert-Parallel Communication Example

## Overview

NIXL EP is a complete example implementation of expert-parallel communication for Mixture of Experts (MoE) models built on top of [NIXL](https://github.com/ai-dynamo/nixl)'s device API. It provides elastic scaling capabilities, enabling dynamic addition and removal of processes (ranks) during runtime without disrupting existing connections, and leverages NIXL's RDMA and NVLink support for optimal performance.

## Features
- **Dispatch and Combine support**: Supports dispatch and combine operations for MoE inference
- **RDMA and NVLink support**: Utilizes NIXL's abstractions to support both RDMA and NVLink transports for optimal performance
- **Elastic Scaling**: Dynamically add or remove ranks during runtime

## Buffer Initialization

NIXL EP provides a flexible buffer initialization pattern that supports dynamic rank management:

```python
import nixl_ep

# Initialize buffer with dynamic rank support
buffer = nixl_ep.Buffer(rank, explicitly_destroy=True)
buffer.update_memory_buffers(num_ranks, num_experts_per_rank, rdma_bytes)
buffer.connect_ranks(initial_ranks)

# Dispatch & Combine calls
buffer.dispatch(...)
buffer.combine(...)

# Later: Connect new ranks dynamically
buffer.connect_ranks(ranks)

# Dispatch & Combine calls
buffer.dispatch(...)
buffer.combine(...)

# Disconnect ranks when scaling down
buffer.disconnect_ranks(ranks)
```

## Key APIs

- `Buffer(rank_id, ...)`: Initialize the NIXL communication buffer
- `update_memory_buffers(num_ranks, num_experts_per_rank, num_rdma_bytes, num_nvl_bytes=0)`: Prepare buffers for up to `num_ranks` ranks and `num_experts_per_rank` experts
- `connect_ranks(remote_ranks, activate=True)`: Establish NIXL connections to new peers (can be called multiple times); in low-latency mode, use `activate=False` to keep new peers masked until explicitly unmasked.
- `disconnect_ranks(remote_ranks)`: Clean up connections to departing peers

## GPU Device API path

NIXL EP can run over either GPU Device API path. Select it when the agent
starts:

- `NIXL_EP_DEVICE_MODE=direct` uses the direct UCX GPU Device API. This is the
  default when the variable is unset.
- `NIXL_EP_DEVICE_MODE=proxy` routes device-side transfers through the CPU
  proxy, for hardware or transports where the direct path is unavailable.

The proxy belongs to the UCX backend and is configured with backend
parameters. EP translates these environment variables into them:

- `NIXL_EP_PROXY_CHANNELS` (falls back to `NIXL_EP_NUM_CHANNELS`, default 4)
  becomes `proxy_channel_count`.
- `NIXL_EP_PROXY_WORKER_COUNT` (defaults to the channel count) becomes
  `proxy_thread_count`, the number of CPU proxy progress threads.
- Peer capacity (`proxy_max_peers`) is EP's `max_num_ranks`. The UCX worker
  count is derived by the backend as channels x peers and must not be passed
  explicitly in proxy mode.

Kernels and Device API calls are identical in both modes.

Proxy view release waits for queued operations and UCX transport cleanup. Stop
and synchronize producers before releasing views or deregistering their buffers.
Disconnect releases old views before invalidating metadata or removing peer resources.
A dead peer need not receive the data, but cleanup can wait for transport retry
timeouts (including connection setup). `UCX_RC_TIMEOUT`, `UCX_RC_RETRY_COUNT`, and
`UCX_UD_TIMEOUT` affect this delay; EP does not change their production defaults.

There is no library-level environment override for this. Outside EP, the proxy
is enabled by passing `device_proxy=true`, plus any `proxy_*` parameters, to
`createBackend("UCX", ...)`.

## Testing

The elastic test suite in `tests/elastic/` validates dynamic scaling capabilities:
- Plan files define scaling phases (representing an orchestrator)
- Tests validate correctness and measure bandwidth between scaling phases

**Example Plan** (`expansion_contraction.json`):
```json
[
  [0, 1, 2, 3],
  [0, 1, 2, 3, 4, 5, 6, 7],
  [0, 1, 2, 3, 4, 5]
]
```
This plan defines three phases:
- **Phase 0**: Initial state with ranks 0-3
- **Phase 1**: Ranks 4-7 are added dynamically (launched independently from initial ranks)
- **Phase 2**: Ranks 6-7 are removed dynamically

## Getting Started

#### Build NIXL with NIXL EP:

First, configure the pkg-config paths (only needed when dependencies are installed to non-default paths)

```bash
export PKG_CONFIG_PATH=<path to rdma-core install>/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=<path to UCX install>/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=<path to DOCA install>/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
```

Then, configure the NIXL plugin directory so it can find UCX plugin, and set the LD_LIBRARY_PATH so UCX can find rdma-core:
```bash
export NIXL_PLUGIN_DIR=<path to NIXL install directory>/lib/x86_64-linux-gnu/plugins
export LD_LIBRARY_PATH=<path to rdma-core install>/lib:$LD_LIBRARY_PATH
```

Build and install:

```bash
meson setup build \
    -Ducx_path=<path to UCX install> \
    -Dprefix=<path to NIXL install directory> \
    -Dbuildtype=release \
    -Dbuild_nixl_ep=true

cd build
ninja install
```


Finally, configure PYTHONPATH to use NIXL EP:
```bash
export PYTHONPATH=<path to NIXL build directory>/examples/device/ep
```

Refer to [tests/elastic/README.md](tests/elastic/README.md) for detailed instructions on how to run the elastic test suite.
