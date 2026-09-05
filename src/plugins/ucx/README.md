<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL UCX Plugin

This plugin provides the default communication backend for NIXL using
[UCX](https://github.com/openucx/ucx). It supports both DRAM (`DRAM_SEG`) and
GPU (`VRAM_SEG`) memory, and selects the best available transport for each
transfer: RDMA (InfiniBand/RoCE) and `cuda_ipc` for intra-node GPU-to-GPU
transfers, falling back to TCP where nothing faster is available.

## Known limitations

### PyTorch `expandable_segments:True` / CUDA VMM memory falls back to the slow path

CUDA buffers allocated through the CUDA Virtual Memory Management API
(`cuMemCreate` / `cuMemMap`) — which is what PyTorch uses when
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set — **cannot be exported
through the legacy `cuIpcGetMemHandle` mechanism** that UCX's `cuda_ipc`
transport relies on.

Such regions still register successfully with NIXL. However, on an intra-node
transfer where `cuda_ipc` would otherwise be used (e.g. two processes on the
same node sharing GPUs over NVLink) and **no RDMA NIC is available as an
alternative**, UCX silently falls back to its software-emulated path over TCP.
The data stays correct, but throughput can drop by orders of magnitude (e.g.
from hundreds of GB/s on `cuda_ipc` to well under 1 GB/s on the emulated path).
Because the fallback is silent and the result is correct, this is difficult to
diagnose in production.

This commonly affects RL / large-model training and inference colocation
setups, where `expandable_segments:True` is a popular fragmentation-mitigation
setting and trainer/inference processes are colocated on a single node.

The UCX plugin emits a one-time `NIXL_WARN` when it registers a CUDA region
that is not legacy CUDA IPC-capable, naming the region and the consequence.
Warnings are visible at the default log level.

> UCX can share CUDA VMM memory across processes via *fabric handles*, but that
> requires fabric-enabled allocations plus IMEX, which PyTorch does not produce
> and which is unavailable on most deployments. See
> [ai-dynamo/nixl#1754](https://github.com/ai-dynamo/nixl/issues/1754) for the
> tracking issue, which also covers supporting POSIX-FD VMM handles intra-node.

#### Workaround

Allocate the buffer that NIXL registers and transfers **outside** expandable
segments, so it comes from a plain `cudaMalloc` allocation that `cuda_ipc` can
export:

```python
import torch

# Carve the NIXL transfer/staging buffer out of a non-expandable allocation.
torch.cuda.memory._set_allocator_settings("expandable_segments:False")
bucket = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
torch.cuda.memory._set_allocator_settings("expandable_segments:True")
```

This restores the `cuda_ipc` zero-copy fast path for the registered buffer while
leaving the rest of the process on expandable segments.

## Troubleshooting

Enable UCX and NIXL logging to inspect transport selection:

```bash
# Print the protocol/transport UCX selects for each operation.
export UCX_PROTO_INFO=y

# NIXL debug logging.
export NIXL_LOG_LEVEL=debug
```

With `UCX_PROTO_INFO=y`, a healthy intra-node GPU transfer reports
`zero-copy | cuda_ipc/cuda`, whereas the degraded fallback described above
reports `software emulation | tcp/...`.
