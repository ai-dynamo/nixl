---
title: "NIXL-EP: Expert-Parallel Communication"
description: Expert-parallel communication for Mixture of Experts (MoE) models built on NIXL's device API with elastic scaling, RDMA and NVLink support.
---

NIXL EP is a complete example implementation of expert-parallel communication for Mixture of Experts (MoE) models built on top of [NIXL](https://github.com/ai-dynamo/nixl)'s device API. It provides elastic scaling capabilities, enabling dynamic addition and removal of processes (ranks) during runtime without disrupting existing connections. Source: [`examples/device/ep/`](https://github.com/ai-dynamo/nixl/tree/main/examples/device/ep)

## Features

- **Dispatch and Combine support**: Dispatch and combine operations for MoE inference
- **RDMA and NVLink support**: Utilizes NIXL's abstractions for both RDMA and NVLink transports
- **Elastic Scaling**: Dynamically add or remove ranks during runtime

## Building NIXL-EP

Configure pkg-config paths (only needed when dependencies are installed to non-default paths):

```bash
export PKG_CONFIG_PATH=<path to rdma-core install>/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=<path to UCX install>/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=<path to DOCA install>/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
```

Set the NIXL plug-in directory and library path:

```bash
export NIXL_PLUGIN_DIR=<path to NIXL install>/lib/x86_64-linux-gnu/plugins
export LD_LIBRARY_PATH=<path to rdma-core install>/lib:$LD_LIBRARY_PATH
```

Build and install with NIXL-EP enabled:

```bash
meson setup build \
    -Ducx_path=<path to UCX install> \
    -Dprefix=<path to NIXL install> \
    -Dbuildtype=release \
    -Dbuild_nixl_ep=true

cd build
ninja install
```

Add NIXL-EP to your Python path:

```bash
export PYTHONPATH=<path to NIXL build>/examples/device/ep
```

## Python API

```python title="Python"
import nixl_ep

# Initialize buffer with dynamic rank support
buffer = nixl_ep.Buffer(rank=rank, explicitly_destroy=True)
buffer.update_memory_buffers(num_ranks, num_experts_per_rank, num_rdma_bytes)
buffer.connect_ranks(initial_ranks)

# Dispatch tokens to experts
recv_x, recv_count, handle, event, hook = buffer.dispatch(
    x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts
)

# ... process tokens through experts ...

# Combine results back
combined_x, event, hook = buffer.combine(
    x, topk_idx, topk_weights, handle
)

# Later: Connect new ranks dynamically (elastic scaling)
buffer.connect_ranks(new_ranks)

# Disconnect ranks when scaling down
buffer.disconnect_ranks(departing_ranks)

# Explicit cleanup
buffer.destroy()
```

### Key Methods

| Method | Description |
|--------|-------------|
| `Buffer(disable_ll_nvlink=False, explicitly_destroy=False, rank=0, group=None, comm=None, tcp_store_group=None)` | Initialize the NIXL communication buffer |
| `update_memory_buffers(num_ranks, num_experts_per_rank, num_rdma_bytes)` | Allocate remote memory for communication |
| `connect_ranks(remote_ranks)` | Establish NIXL connections to new peers (can be called multiple times) |
| `disconnect_ranks(remote_ranks)` | Clean up connections to departing peers |
| `dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, ...)` | Low-latency dispatch with NIXL device API |
| `combine(x, topk_idx, topk_weights, handle, ...)` | Low-latency combine (reduce with weights) via NIXL device API |
| `get_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)` | Get minimum RDMA buffer size requirement |
| `clean_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)` | Zero-initialize buffer (required before reuse) |
| `destroy()` | Explicitly release resources (when `explicitly_destroy=True`) |

## C++ Core

The C++ implementation in `csrc/` provides the high-performance backend:

- `nixl_ep.hpp` / `nixl_ep.cpp` -- Core Buffer implementation with Transfer Agent management
- `kernels/` -- CUDA kernels for dispatch and combine operations
- `config.hpp` -- Configuration constants
- `event.hpp` -- CUDA event handling

The C++ core is exposed to Python via pybind11. Direct C++ usage requires building with meson and linking against the NIXL library.

## Elastic Scaling

NIXL-EP supports dynamic addition and removal of ranks during runtime. The `connect_ranks()` method can be called multiple times to add new peers, and `disconnect_ranks()` removes departing peers.

Example scaling plan from the test suite:

```json
[
  [0, 1, 2, 3],
  [0, 1, 2, 3, 4, 5, 6, 7],
  [0, 1, 2, 3, 4, 5]
]
```

- **Phase 0**: Initial state with ranks 0-3
- **Phase 1**: Ranks 4-7 added dynamically
- **Phase 2**: Ranks 6-7 removed dynamically

## Testing

The elastic test suite in `tests/elastic/` validates dynamic scaling capabilities.

Single-node example (8 ranks, 4 to 8 expansion):

```bash
python3 tests/elastic/elastic.py \
    --plan tests/elastic/single_expansion.json \
    --num-processes 8
```

Multi-node example with 2 nodes:

**Node 1** (launches the first phase with 4 ranks):

```bash
python3 tests/elastic/elastic.py \
    --plan tests/elastic/single_expansion.json \
    --num-processes 4
```

**Node 2** (joins the second phase with additional 4 ranks):

```bash
python3 tests/elastic/elastic.py \
    --plan tests/elastic/single_expansion.json \
    --num-processes 4 \
    --tcp-server $MASTER_IP
```

### Available Test Plans

| Plan File | Description |
|-----------|-------------|
| `no_expansion.json` | Static 4 ranks (baseline) |
| `single_expansion.json` | 4 to 8 ranks (single expansion) |
| `double_expansion.json` | 4 to 6 to 8 ranks (two expansions) |
| `expansion_contraction.json` | 4 to 8 to 6 ranks (scale up then down) |

<Tip>
NIXL-EP builds on the [NIXL Device API](../api-reference/device-api). For general NIXL concepts, see [Architecture](../getting-started/architecture).
</Tip>
