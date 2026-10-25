<!--
SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This file incorporates material from the DeepSeek project, licensed under the MIT License.
The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.

SPDX-License-Identifier: MIT AND Apache-2.0
-->

# NIXL-EP V2

## Overview

NIXL-EP V2 provides the adaptive `ElasticBuffer` API from DeepEP V2 on top of
the NIXL GPU transport. It supports expert-parallel dispatch and combine over
NVLink and RDMA, elastic membership, and topology changes without reallocating
the communication buffer.

## Features

- **Dispatch and combine:** High-throughput expert-parallel communication.
- **NVLink and RDMA:** Direct mode within one NVLink domain and hybrid mode
  across domains.
- **Elastic membership:** Connect, disconnect, or temporarily deactivate ranks.
- **CUDA graph reuse:** Change the active topology between graph replays.

## Buffer initialization

Create an `ElasticBuffer` from an initialized PyTorch process group:

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

# Dispatch and combine calls
buffer.dispatch(...)
buffer.combine(...)

# Update the live world between launches
buffer.set_active_ranks(active_rank_ids)

buffer.destroy()
```

The original NIXL-EP implementation is temporarily available through
`import nixl_ep`. It provides `nixl_ep.Buffer` and will be removed in a future
release.

## Key APIs

- `ElasticBuffer(...)`: Allocate the communication buffer and connect the
  initial cohort.
- `dispatch(...)` / `combine(...)`: Exchange token data between expert ranks.
- `connect_ranks(...)` / `disconnect_ranks(...)`: Stage peer membership
  changes.
- `set_active_ranks(...)`: Commit a new live world. The mode follows it:
  direct when exactly one NVLink domain is live, hybrid otherwise.
- `set_num_active_sms(...)`: Change the active SM count of the live world.

## Requirements

NIXL-EP V2 requires CUDA 12.9 or newer with its tools (`nvcc` and `cuobjdump`),
PyTorch, a UCX build with its GPU device API and CUDA IPC support, and
rdma-core development headers. The first operation that needs a JIT kernel
compiles it and caches the result under `~/.nixl_ep_v2`.

Runtime and JIT settings use the `NIXL_EP_*` environment-variable prefix. The
packaged NIXL and UCX headers are used by default; advanced users can override
their roots with `NIXL_EP_NIXL_ROOT_DIR` and `NIXL_EP_UCX_ROOT_DIR`.

## Getting started

Configure paths for dependencies installed outside the system defaults:

```bash
export PKG_CONFIG_PATH=<rdma-core-prefix>/lib/pkgconfig:<ucx-prefix>/lib/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=<nixl-libdir>/plugins
export LD_LIBRARY_PATH=<rdma-core-prefix>/lib:$LD_LIBRARY_PATH
```

`<nixl-libdir>` is `<nixl-prefix>/<libdir>`, where `libdir` is meson's, usually
`lib/x86_64-linux-gnu` on Debian and Ubuntu (`lib/aarch64-linux-gnu` on Arm) and
`lib64` on RHEL-like systems. After `meson setup`, `meson configure build` shows it.

Build and install NIXL with both NIXL-EP implementations enabled:

```bash
meson setup build \
    -Ducx_path=<ucx-prefix> \
    -Dprefix=<nixl-prefix> \
    -Dbuildtype=release \
    -Dbuild_nixl_ep=true

meson compile -C build
meson install -C build
```

## Local tests

Run these commands from `runtime/ep` after installing NIXL. With a `meson install`,
also put the installed Python packages and the `nixl` dispatcher on `PYTHONPATH`,
and the NIXL libraries on `LD_LIBRARY_PATH` (a pip-installed wheel needs none of them).
`<nixl-python-dir>` is the directory meson installed the `nixl_cu<cuda-major>` package
into. It follows the Python that meson was configured with:
`<nixl-prefix>/lib/python3/dist-packages` with the Python of Debian or Ubuntu (also in
a virtual environment made from it), `<nixl-prefix>/lib64/python3.<minor>/site-packages`
with the Python of RHEL-like systems, `<nixl-prefix>/lib/python3.<minor>/site-packages`
with other Pythons, such as a uv-managed one. `find <nixl-prefix> -type d -name 'nixl_cu[0-9]*'`
prints `<nixl-python-dir>/nixl_cu<cuda-major>`.

```bash
export PYTHONPATH=<nixl-python-dir>:<nixl-source>/src/bindings/python/nixl-meta:$PYTHONPATH
export LD_LIBRARY_PATH=<nixl-libdir>:$LD_LIBRARY_PATH
python3 tests/elastic/test_adaptive_elastic.py --num-processes 4 --num-tokens 512 --hidden 7168 --num-topk 8 --num-experts 192
python3 tests/elastic/test_ep.py --num-processes 4 --test-first-only --skip-perf-test
python3 tests/elastic/test_adaptive_modes.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 8 --num-experts 256
```

`tests/elastic/test_adaptive_elastic.py` is an elastic, correctness and
performance test. It runs a plan of live worlds (phases) that grow and shrink
the expert-parallel world over NVLink and RDMA without reallocating the buffer.
In every phase it checks dispatch + combine bit-exactly, both eagerly and by
replaying a CUDA graph captured in an earlier phase. By default it also prints
the dispatch + combine bandwidth of every phase. Add `--kineto` for per-kernel
timings, or `--save-output results.json` to also save the results as JSON.

The other scripts under `tests/` are listed below:

- `tests/elastic/test_ep.py`: dispatch and combine correctness tests on a fixed
  world, checked against a `torch.distributed` reference.
- `tests/elastic/test_adaptive_modes.py`: direct and hybrid mode, handle
  sizing, and CUDA-graph replay across mode and world switches.
- `tests/elastic/test_barrier.py` and `tests/utils/test_gate.py`: the barrier
  kernel and the gate-score generator.
- `tests/elastic/test_adaptive_processes.py`: processes that join late and
  exit, launched as separate steps against an external TCPStore.
- `tests/elastic/test_elastic_warmup.py`: background connection and warm-up
  over repeated grow and shrink cycles.
- `tests/elastic/profile_dispatch.py` and `tests/elastic/profile_epilogues.py`:
  kernel phase timings; they need `NIXL_EP_ADAPTIVE_PROFILE=1`.
- `tests/elastic/test_zero_copy.py`: zero-copy dispatch (direct mode only),
  checked against the staged dispatch on the same buffer.
- `tests/elastic/test_nvfp4_dispatch.py`: NVFP4 dispatch (uint8 codes and
  block scales, BF16 combine), checked bitwise against the reference.
- `tests/elastic/test_deterministic_sort.py`: `EPHandle.deterministic_sort`,
  captured into a CUDA graph and replayed with other receive counts.
- `tests/elastic/sweep_sms.py`: a benchmark that sweeps the active SM count and
  the token count; it needs `NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS`.
- `tests/elastic/compile_probe_adaptive.cu`: a compile-only check of the kernel
  templates, built by hand with `nvcc`.
