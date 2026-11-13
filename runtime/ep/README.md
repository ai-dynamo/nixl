<!--
SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This file incorporates material from the DeepSeek project, licensed under the MIT License.
The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.

SPDX-License-Identifier: MIT AND Apache-2.0
-->

# NIXL EP V2: Adaptive Expert-Parallel Communication

## Overview

NIXL EP V2 is an expert-parallel communication implementation for Mixture-of-Experts (MoE) models built on top of NIXL's device API.

Compared with NIXL EP V1 ([examples/device/ep](../../examples/device/ep/README.md)), V2 adds topology-aware direct and hybrid kernels.

## Features

- **Dispatch and Combine support**: Low-latency and high-throughput expert-parallel communication for MoE inference.
- **Topology-Aware**: One kernel for both topologies, with the data path chosen at kernel entry from a device-side topology descriptor: direct mode within one NVLink or Multi-Node NVLink (MNNVL) domain, hybrid mode with RDMA between separate NVLink domains and NVLink within them. The mode is selected automatically from the active topology. A rescale can switch between Multi-Node NVLink and RDMA over separate NVLink domains, with no buffer re-creation.
- **CUDA Graph Reuse**: Captured graphs remain valid across topology and direct/hybrid mode changes.
- **Elastic Scaling**: Ranks can be connected, disconnected, activated, or deactivated dynamically.
- **GEMM Agnosticism**: Dispatch outputs expert-major compact, expert-major padded, or rank-major layouts, selected per call; combine accepts the same layout back.
- **NVFP4 Support**: NVFP4 on the wire in dispatch, alongside BF16 and FP8.
- **Resource Optimization**: The kernels are optimized to run on few SMs: an automatic SM count from a bandwidth model of the live world, a fixed launch grid whose idle blocks exit at entry, and one receive region shared by dispatch and combine.

## Buffer Initialization

NIXL EP V2 uses an `ElasticBuffer` sized for the maximum expected topology:

```python
import torch.distributed as dist
from nixl.runtime import ep

buffer = ep.ElasticBuffer(
    group=dist.group.WORLD,
    num_max_tokens_per_rank=4096,
    hidden=7168,
    num_topk=8,
    num_experts_per_rank=4,
    explicitly_destroy=True,
)

# Dispatch and combine
buffer.dispatch(...)
buffer.combine(...)

# Change the active world.
# The communication mode is updated automatically.
buffer.set_active_ranks(active_rank_ids)

buffer.dispatch(...)
buffer.combine(...)

buffer.destroy()
```

More complex scaling operations can establish or remove peer connections separately:

```python
# remote_mds, peer_infos: the new peers' get_local_metadata() and get_local_peer_info()
buffer.connect_ranks(new_rank_ids, remote_mds, peer_infos)
buffer.stage_topology(active_ranks=new_active_rank_ids)
buffer.commit_topology()

# After departed ranks are no longer active:
buffer.disconnect_ranks(departed_rank_ids)
buffer.commit_topology()
```

## Key APIs

- `ElasticBuffer(...)`: Allocate communication resources for the maximum topology and initialize the current cohort.
- `dispatch(...)`: Route token activations and expert selections to the destination ranks.
- `combine(...)`: Return and reduce expert outputs.
- `connect_ranks(...)`: Establish NIXL connections to new peers.
- `disconnect_ranks(...)`: Remove connections to departed peers.
- `stage_topology(...)`: Prepare a new active topology.
- `commit_topology()`: Make the staged topology visible to subsequent kernels.
- `set_active_ranks(...)`: Stage and commit a new live world in one call. The direct or hybrid mode is derived automatically.
- `set_num_active_sms(...)`: Change the active kernel SM count without changing the live world.

The docstrings in `nixl_ep_v2/buffers/elastic.py` describe the arguments and the rules for topology changes and CUDA graphs.

## Requirements

- To build: CUDA 13.0 or newer, a C++20 compiler and standard library with `<format>` (for example GCC 13 or newer), PyTorch for the `python3` that meson finds, pybind11, and UCX with its GPU device API and CUDA IPC support. If one is missing, `meson setup` warns `... skipping NIXL-EP V2 build` and builds NIXL without V2.
- To run: the CUDA toolkit (`nvcc` and `cuobjdump`), rdma-core development headers, and nvidia-ml-py (a NIXL wheel installs it, a `meson install` does not). MIG is not supported. The automatic SM count runs `nvidia-smi nvlink -s` and, with several NVLink domains, `ibstat` (from infiniband-diags).

NIXL EP V2 compiles its kernels on first use with the CUDA toolkit of `CUDA_HOME` (else `CUDA_PATH`, the `nvcc` on `PATH`, or `/usr/local/cuda`) and caches them under `~/.nixl_ep_v2`. RDMA between NVLink domains needs the rdma-core header `infiniband/mlx5dv.h` in `/usr/include`, `/usr/local/include`, or a directory on `CPATH` or `CPLUS_INCLUDE_PATH`.

## Getting Started

Configure paths for dependencies installed outside the system defaults, both to build NIXL and to run NIXL EP V2:

```bash
export PKG_CONFIG_PATH=<rdma-core-prefix>/lib/pkgconfig:<ucx-prefix>/lib/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=<nixl-libdir>/plugins
export LD_LIBRARY_PATH=<ucx-prefix>/lib:<rdma-core-prefix>/lib:$LD_LIBRARY_PATH
```

`<ucx-prefix>/lib` comes first so that this UCX is loaded rather than an older one, such as the HPC-X UCX of NVIDIA's PyTorch containers, with which NIXL EP V2 fails (UCX warns `UCP API version is incompatible`). `<nixl-libdir>` is `<nixl-prefix>/<libdir>` with meson's `libdir`, for example `lib/x86_64-linux-gnu` on Ubuntu; after `meson setup`, `meson configure build` shows it.

Build NIXL with NIXL EP enabled:

```bash
meson setup build \
    -Ducx_path=<ucx-prefix> \
    -Dprefix=<nixl-prefix> \
    -Dbuildtype=release \
    -Dbuild_nixl_ep=true

meson compile -C build
meson install -C build
```

With a `meson install` (a NIXL wheel needs neither), also set:

```bash
export PYTHONPATH=<nixl-python-dir>:<nixl-source>/src/bindings/python/nixl-meta:$PYTHONPATH
export LD_LIBRARY_PATH=<nixl-libdir>:$LD_LIBRARY_PATH
```

`<nixl-python-dir>` is the parent of the `nixl_cu<cuda-major>` directory that `find <nixl-prefix> -type d -name 'nixl_cu[0-9]*'` prints, for example `<nixl-prefix>/lib/python3/dist-packages` with the system Python of Debian or Ubuntu.

After installation:

```python
from nixl.runtime import ep
```

## Environment Variables

Set these before importing `nixl.runtime.ep`:

| Variable | Default | Effect |
| --- | --- | --- |
| `NIXL_EP_NIC_NAME` | `mlx5_0` | The adapter whose `ibstat` rate the automatic SM count uses with several NVLink domains; it must be one that `ibstat -l` lists |
| `NIXL_EP_ADAPTIVE_MODE` | unset | `hybrid` runs the hybrid kernels also when a single NVLink domain is live |
| `NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS` | `26` | The launch grid of the kernels (even, at most 64) |
| `NIXL_EP_JIT_CACHE_DIR` | `~/.nixl_ep_v2` | The JIT cache |
| `NIXL_EP_NIXL_ROOT_DIR`, `NIXL_EP_UCX_ROOT_DIR` | the installed package | The roots of the NIXL and UCX headers that the JIT compiles against |
| `NIXL_EP_BUFFER_DEBUG` | unset | `1` prints the buffer sizes, the SM count model and transport details |

The constructor sets `UCX_IB_PCI_RELAXED_ORDERING=n`, `UCX_IB_PATH_MTU=4096` and `UCX_UD_LINGER_TIMEOUT=1s` when they are unset, and refuses other values of the first two.

## Testing

The test suite under `runtime/ep/tests` validates fixed and elastic expert-parallel communication. Run the tests from `runtime/ep` in the environment of [Getting Started](#getting-started):

```bash
python3 tests/elastic/test_adaptive_elastic.py --num-processes 4 --num-tokens 512 --hidden 7168 --num-topk 8 --num-experts 192
python3 tests/elastic/test_ep.py --num-processes 4 --test-first-only --skip-perf-test
python3 tests/elastic/test_adaptive_modes.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 8 --num-experts 256
```

`test_ep.py` checks dispatch and combine on a fixed world against a `torch.distributed` reference, `test_adaptive_modes.py` checks direct and hybrid mode and CUDA-graph replay across mode and world switches, and `test_adaptive_elastic.py` runs a plan.

### Plans

Plan files define a sequence of active worlds. Each phase validates dispatch and combine correctness, active membership, and the automatically selected communication mode.

Example plan, for two NVLink domains of four GPUs (rank ids 0-3 and 4-7):

```json
[
  [0, 1, 2, 3],
  [0, 1, 2, 3, 4, 5, 6, 7],
  [4, 5, 6, 7]
]
```

This plan defines:

- **Phase 0**: One NVLink domain is active; direct mode is selected.
- **Phase 1**: Two domains are active; hybrid RDMA+NVLink mode is selected.
- **Phase 2**: One domain remains active; the mode returns to direct.

`--plan` takes a plan file or inline JSON, such as `--plan '[[0,1,2,3],[0,1,2,3,4,5,6,7],[4,5,6,7]]'` for the plan above. Available test plans:

- `tests/elastic/plans/two_domains.json`: two NVLink domains of four GPUs, 7 phases that switch between direct and hybrid mode.
- No `--plan`: the full world and smaller worlds built from the job (a single domain, fewer positions per domain, or the first position left out).

### Multiple Nodes

To run a plan on several nodes, start the test once per node with `WORLD_SIZE` (the number of nodes), `RANK` (the node index), `MASTER_ADDR` and `MASTER_PORT` (an address of node 0 and a free port on it). For example, with four GPUs on each of two nodes that do not share a Multi-Node NVLink domain:

```bash
# Node 0 (ranks 0-3)
WORLD_SIZE=2 RANK=0 MASTER_ADDR=<node-0-address> MASTER_PORT=<port> \
    python3 tests/elastic/test_adaptive_elastic.py --num-processes 4 --plan tests/elastic/plans/two_domains.json
# Node 1 (ranks 4-7)
WORLD_SIZE=2 RANK=1 MASTER_ADDR=<node-0-address> MASTER_PORT=<port> \
    python3 tests/elastic/test_adaptive_elastic.py --num-processes 4 --plan tests/elastic/plans/two_domains.json
```

### Other Tests

- `tests/elastic/test_elastic_warmup.py`: background connection and warm-up over repeated grow and shrink cycles.
- `tests/elastic/test_adaptive_processes.py`: processes that join late and exit, launched as separate steps against an external TCPStore (see its docstring).
- `tests/elastic/test_zero_copy.py`, `tests/elastic/test_nvfp4_dispatch.py` and `tests/elastic/test_deterministic_sort.py`: zero-copy dispatch, NVFP4 dispatch, and `EPHandle.deterministic_sort`.
- `tests/elastic/test_barrier.py` and `tests/utils/test_gate.py`: the barrier and the gate-score generator.
- `tests/elastic/sweep_sms.py`: a sweep of the active SM count and the token count; counts above the launch grid (`NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS`) are skipped.
- `tests/elastic/profile_dispatch.py` and `tests/elastic/profile_epilogues.py`: dispatch phase and epilogue timings, with `NIXL_EP_ADAPTIVE_PROFILE=1`.
- `tests/elastic/compile_probe_adaptive.cu`: a compile-only check of the kernel templates, built by hand with `nvcc`.
