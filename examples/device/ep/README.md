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

- `Buffer(rank_id, nvlink_backend, explicitly_destroy)`: Initialize the NIXL communication buffer
- `update_memory_buffers(num_ranks, num_experts_per_rank, num_rdma_bytes)`: Prepare buffers for up to `num_ranks` ranks and `num_experts_per_rank` experts
- `connect_ranks(remote_ranks)`: Establish NIXL connections to new peers (can be called multiple times)
- `disconnect_ranks(remote_ranks)`: Clean up connections to departing peers

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
