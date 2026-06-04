<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Local VRAM Staging Design

This document describes a staged local GPU-to-GPU transfer mode for NIXL on single-node systems
where CUDA peer access or GPU Direct P2P is unavailable or not effective. The target use case is
single-node prefill/decode disaggregation on consumer GPUs, where both sides already have fixed
KV-cache VRAM pools registered at startup.

The intended application-facing model is unchanged:

- The application continues registering `VRAM_SEG`.
- The application continues using the existing NIXL agent API.
- The transfer completion is reported only after the destination GPU memory has been updated.
- SGLang does not need to parse backend metadata or add a separate copy path.

## Problem

On consumer GPU systems, two local GPUs may not have CUDA peer access:

```text
cudaDeviceCanAccessPeer(src_gpu, dst_gpu) == 0
```

In this case a direct GPU-to-GPU copy either fails, falls back internally, or has unpredictable
performance. The desired local staged data path is:

```text
source GPU -> pinned host staging -> target GPU
```

This is similar to the remote UCX VRAM staging path, but there is no network RDMA stage:

```text
remote staged path:
source GPU -> source pinned DRAM -> UCX host RDMA -> target pinned DRAM -> target GPU

local staged path:
source GPU -> pinned DRAM -> target GPU
```

## Goals

- Preserve the NIXL API and `VRAM_SEG` surface.
- Support single-node PD-separated processes on different GPUs.
- Avoid CUDA P2P requirements.
- Keep pinned host staging buffers resident after memory registration; do not allocate or register
  staging memory for every transfer.
- Support `NIXL_WRITE` first, because the current SGLang disaggregation path is expected to be
  WRITE-heavy.
- Provide transfer-only benchmark timing that excludes pattern generation, CUDA allocation,
  metadata exchange, memory registration, and cleanup.
- Avoid changing OS routes or machine-level network configuration.

## Non-Goals

- Do not implement `NIXL_READ` in the first local staged version.
- Do not require SGLang source changes in the first version.
- Do not rely on direct CUDA peer access.
- Do not make the first version depend on a specific network HCA. Local payload transfer should not
  use the network data plane.

## Backend Placement

The short-term implementation should extend the current UCX backend staged mode with a local fast
path instead of adding NIXL core logic.

Reasons:

- SGLang can continue selecting the `UCX` backend.
- The existing staged WRITE state machine already owns request completion, notification ordering,
  slot leasing, and ACK semantics.
- UCX active messages can still be used as a control plane between local processes.
- The local data plane can skip UCX RMA when both agents are on the same host and shared staging is
  available.

Longer term, this could be split into a dedicated `LOCAL_STAGED` or `CUDA_STAGING` backend. That is
cleaner architecturally, but it would likely require SGLang configuration changes and more backend
selection work.

## Two Local Modes

### Mode A: Reuse Existing UCX Staged Path

This is the lowest-risk correctness mode and may already work when both agents are local:

```text
source GPU
  -> source process private pinned staging
  -> UCX shm/cma/memory-copy transport
  -> target process private pinned staging
  -> target GPU
```

Pros:

- Reuses the current remote staged WRITE implementation.
- Reuses target-side slot lease and ACK logic.
- Requires minimal code changes.

Cons:

- Adds an extra host-to-host copy.
- Still goes through UCX data movement even though the peer is local.
- Performance can be lower than the optimal local staging path.

This mode is useful as Phase 1 validation and as a fallback when shared pinned staging cannot be
attached across processes or containers.

### Mode B: Shared Pinned Staging Fast Path

The optimized local mode should use target-owned shared pinned staging slots:

```text
source GPU
  -> target-owned shared pinned host slot
  -> target GPU
```

The target process creates a shared-memory staging pool at `VRAM_SEG` registration time. The source
process maps that shared memory after loading target metadata, registers its local virtual mapping
with CUDA, and copies source GPU data directly into the target-owned staging slot.

The target then copies from the same physical host pages into the target GPU.

This removes both UCX RMA and the source-private staging buffer from the local data path.

## Proposed Parameters

Backend parameters:

```text
vram_local_staging=true
local_staging_mode=auto          # auto | shared_pinned | ucx_staged | off
local_staging_chunk_size=16777216
local_staging_slots_per_gpu=4
local_staging_slot_request_window=32
local_staging_cuda_copy_streams=1
local_staging_shm_dir=/dev/shm/nixl
local_staging_host_id=<optional stable host id>
```

Environment variables:

```text
NIXL_UCX_VRAM_LOCAL_STAGING=1
NIXL_UCX_LOCAL_STAGING_MODE=auto
NIXL_UCX_LOCAL_STAGING_CHUNK_SIZE=16777216
NIXL_UCX_LOCAL_STAGING_SLOTS=4
NIXL_UCX_LOCAL_STAGING_SLOT_REQUEST_WINDOW=32
NIXL_UCX_LOCAL_STAGING_CUDA_COPY_STREAMS=1
NIXL_UCX_LOCAL_STAGING_SHM_DIR=/dev/shm/nixl
NIXL_UCX_LOCAL_STAGING_HOST_ID=<optional stable host id>
```

`auto` should prefer the shared pinned fast path when both of these are true:

- local and remote metadata report the same host id;
- the source process can open, mmap, and CUDA-register the target shared staging object.

If either check fails, `auto` should fall back to the existing UCX staged path if
`NIXL_UCX_VRAM_STAGING=1` is also enabled.

## Same-Host Detection

Metadata should include a stable host id:

```text
host_id
pid
boot_id
container_hint
```

Default host-id selection:

1. `NIXL_UCX_LOCAL_STAGING_HOST_ID`, if provided.
2. `/etc/machine-id`, if available.
3. hostname as a fallback.

Container deployments need special care. Two containers on the same physical host may have
different hostnames or private IPC namespaces. For shared pinned staging, both processes need to see
the same shared-memory object. Practical deployment options:

- run with shared IPC namespace, such as `--ipc=host`;
- or use a bind-mounted staging directory, such as `/tmp/nixl-local-staging`, that is visible to both
  containers.

If the shared object cannot be opened by the source process, the backend must not silently use an
invalid pointer. It should fall back or fail clearly.

## Metadata Design

Public staged local metadata should be versioned:

```text
magic = "NIXL_UCX_LOCAL_STAGED_V1"
mode = "shared_pinned"
host_id
pid
region_id
gpu_base
gpu_len
gpu_dev_id
slot_size
slot_count
shared_object:
  type = "posix_shm" or "file_mmap"
  name_or_path
  total_size
  data_offset
capabilities:
  write = true
  read = false
```

Private metadata should store:

```text
gpu_base
gpu_len
gpu_dev_id
local mapped host addresses
CUDA host registration state
slot state table
lease table
copy streams
copy events
shared object ownership and cleanup state
```

Raw host virtual addresses should not be treated as portable between processes. The source process
must map the target shared object and use its own local virtual address for CUDA D2H.

## WRITE Protocol

The optimized shared-pinned WRITE protocol is:

```text
initiator:
  load target metadata
  mmap target shared staging pool
  cudaHostRegister(mapped staging pages)

initiator:
  STAGED_LOCAL_SLOT_REQ(target_gpu_addr, gpu_dev, size, transfer_id, chunk_id)

target:
  find target staged region
  reserve one target-owned shared slot
  STAGED_LOCAL_SLOT_GRANT(slot_id, lease_id, status)

initiator:
  cudaMemcpyAsync D2H from source GPU into mapped target shared slot
  wait/query D2H event
  STAGED_LOCAL_WRITE_READY(slot_id, lease_id, target_gpu_addr, size)

target:
  validate lease
  cudaMemcpyAsync or cudaMemcpy H2D from shared slot into target GPU
  wait/query H2D completion
  release slot
  STAGED_LOCAL_ACK(status)

initiator:
  mark chunk complete after ACK
```

Important semantic rule:

```text
NIXL completion must wait until target H2D has completed.
```

This preserves the application-visible semantics currently used by the remote staged path.

## Slot Ownership

The target process owns local shared staging slots. Slot state should reuse the current staged slot
lease model:

```text
FREE
LOCAL_D2H
REMOTE_RESERVED
REMOTE_RDMA_READY
REMOTE_H2D
ERROR
```

For local shared staging, the names can be interpreted as:

- `REMOTE_RESERVED`: slot granted to an initiator and waiting for source D2H into shared memory.
- `REMOTE_RDMA_READY`: source D2H has completed and the target can copy H2D.
- `REMOTE_H2D`: target H2D is in progress.

The target must validate:

```text
slot_id
lease_id
owner agent
transfer_id
chunk_id
target GPU address range
target GPU dev id
size <= slot_size
```

before starting H2D.

## Pipeline Strategy

The local shared-pinned path has two data stages:

```text
D2H: source GPU -> shared pinned host slot
H2D: shared pinned host slot -> target GPU
```

There is no RDMA write and no UCX flush. The primary pipeline depth controls are:

- chunk size;
- target slot count;
- slot request window;
- CUDA copy stream count.

Initial suggested values:

```text
chunk_size = 16 MiB
slots_per_gpu = 4 or 8
slot_request_window = 32
cuda_copy_streams = 1
```

The expected performance ceiling is roughly bounded by:

```text
min(source D2H bandwidth, target H2D bandwidth)
```

On the current 4090 measurements, pinned D2H/H2D are around 24 to 25 GiB/s. A well-pipelined local
staged path should therefore aim for the low-to-mid 20 GiB/s range before considering more invasive
optimizations.

## CUDA Stream Semantics

NIXL does not currently receive the application's CUDA stream. The local staged path has the same
semantic caveat as the remote staged path:

- The application must make source GPU data visible before posting the NIXL transfer.
- The application must not consume target GPU data before NIXL completion or notification.

The first version can use conservative synchronization for correctness. The performance version
should use backend-owned CUDA streams and events:

```text
source D2H async -> event query -> READY
target H2D async -> event query -> ACK
```

Future API-level stream/event integration would be cleaner but would likely require application
changes, so it is out of scope for the first local staged implementation.

## Verification Plan

Baseline checks:

```text
cudaDeviceCanAccessPeer(src, dst)
nvidia-smi topo -m
pinned D2H bandwidth
pinned H2D bandwidth
cudaMemcpyPeer or cudaMemcpyDefault behavior when P2P is unavailable
```

Correctness smoke:

```text
single process, two GPUs, one transfer
two local processes, two GPUs, one transfer
two local processes, concurrency=8, slots=1
two local processes, concurrency=8, slots=4
multi-descriptor KV-page-like writes
two initiators writing different target offsets
```

Performance matrix:

```text
chunk=8MiB,  slots=4
chunk=16MiB, slots=4
chunk=16MiB, slots=8
chunk=32MiB, slots=4
chunk=32MiB, slots=8
```

Counters and logs:

- Network payload counters should not grow materially for shared-pinned local payload transfer.
- UCX may still carry control messages if the UCX backend is used as the control plane.
- Transfer-only timing should exclude setup, memory registration, metadata exchange, verification,
  and cleanup.

Do not change `ip route` or host routing during validation.

## Implementation Phases

### Phase 0: Capability and Baseline

- Confirm no CUDA P2P between the selected GPUs.
- Measure D2H and H2D bandwidth with pinned host memory.
- Measure current UCX staged behavior between two local processes.
- Confirm whether containers share `/dev/shm` or a bind-mounted staging directory.

### Phase 1: Local Correctness Through Existing UCX Staged Path

- Run the current staged WRITE smoke on one node with two processes and two GPUs.
- Force local UCX transports for validation, for example `UCX_TLS=shm,self` if compatible with the
  environment.
- Verify correctness, transfer-only timing, and that network payload counters do not carry the main
  transfer.

### Phase 2: Shared Pinned Pool Registration

- Add target-owned shared staging pool allocation at `VRAM_SEG` registration time.
- Support POSIX shared memory or file-backed mmap staging objects.
- Register mapped pages with CUDA using `cudaHostRegister`.
- Publish versioned local staged metadata.
- Add source-side attach, mmap, CUDA host registration, and detach paths.

### Phase 3: Local Shared-Pinned WRITE

- Add same-host detection.
- Add local staged request states.
- Reuse target-side slot lease validation.
- Implement one chunk:
  `source D2H into target shared slot -> READY -> target H2D -> ACK`.
- Ensure completion waits for target H2D.

### Phase 4: Multi-Chunk Pipeline

- Add chunk scheduling.
- Add target slot request window.
- Add CUDA event polling for source D2H and target H2D.
- Benchmark sync target H2D versus target H2D worker. The remote staged result showed the worker did
  not help there, but the local path has a different control/data balance and should be measured
  again.

### Phase 5: SGLang Local PD Validation

- Run a small local PD-separated SGLang path with the local staged mode enabled.
- Confirm transfer operation type is `NIXL_WRITE`.
- Confirm descriptor shape is compatible with the staged implementation.
- Confirm no transfer-handle repost behavior conflicts with the staged request state.
- Compare TTFT/TPOT and transfer-only NIXL timing against the existing path.

### Phase 6: Robustness

- Add stale shared-object cleanup.
- Add lease timeout and ERROR-slot recovery.
- Add attach failure fallback.
- Add telemetry for:
  - local staged bytes;
  - D2H time;
  - H2D time;
  - slot wait time;
  - attach failures;
  - fallback count.

## Risks

- Shared memory visibility can fail across containers with private IPC namespaces.
- Pinned memory can grow too large if every registered VRAM region allocates a separate staging
  pool. A global per-device pool may be needed for SGLang deployments with many registrations.
- `cudaHostRegister` on shared mappings must be validated on the target machines.
- Target slots are held while source D2H is in progress. Enough slots are needed to keep D2H and
  H2D overlapped.
- Without application stream integration, correctness depends on the application respecting NIXL
  post/completion ordering.

## Immediate Next Steps

1. Add a local two-process smoke mode to validate current UCX staged behavior on one node.
2. Prototype shared pinned pool allocation and cross-process attach with `cudaHostRegister`.
3. Implement single-chunk local shared-pinned WRITE.
4. Extend the smoke benchmark to compare:
   - current UCX staged local path;
   - shared-pinned local path;
   - CUDA peer copy behavior when P2P is unavailable.
5. Only after the local smoke is stable, run SGLang local PD with the mode enabled.
