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
- `set_active_ranks(...)`: Commit a new live world.
- `set_topology(...)`: Switch between direct and hybrid communication modes.

## Requirements

NIXL-EP V2 requires CUDA tools (`nvcc` and `cuobjdump`), PyTorch, a UCX build
with its GPU device API and CUDA IPC support, and rdma-core development
headers. The first operation that needs a JIT kernel compiles it and caches
the result under `~/.nixl_ep_v2`.

Runtime and JIT settings use the `NIXL_EP_*` environment-variable prefix. The
packaged NIXL and UCX headers are used by default; advanced users can override
their roots with `NIXL_EP_NIXL_ROOT_DIR` and `NIXL_EP_UCX_ROOT_DIR`.

## Getting started

Configure paths for dependencies installed outside the system defaults:

```bash
export PKG_CONFIG_PATH=<rdma-core-prefix>/lib/pkgconfig:<ucx-prefix>/lib/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=<nixl-prefix>/lib/x86_64-linux-gnu/plugins
export LD_LIBRARY_PATH=<rdma-core-prefix>/lib:$LD_LIBRARY_PATH
```

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
also put the installed Python packages and the `nixl` dispatcher on `PYTHONPATH`
(a pip-installed wheel needs neither):

```bash
export PYTHONPATH=<nixl-prefix>/lib/python3.<minor>/site-packages:<nixl-source>/src/bindings/python/nixl-meta:$PYTHONPATH
python tests/elastic/test_ep.py --num-processes 4 --test-first-only --skip-perf-test
python tests/elastic/test_adaptive_elastic.py --num-processes 4 --num-tokens 512 --hidden 7168 --num-topk 8 --num-experts 192
python tests/elastic/test_adaptive_modes.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 8 --num-experts 256
```
