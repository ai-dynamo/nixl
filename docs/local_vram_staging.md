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

For the SGLang single-node and multi-node PD validation runbook, see
[`sglang_pd_staging_integration.md`](sglang_pd_staging_integration.md).

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

The optimized local mode should use shared pinned staging so that the payload does not need UCX RMA
or an extra host-to-host copy. There are two ownership options.

#### Option B1: Source-Owned Shared Staging

This is the preferred performance path:

```text
source GPU
  -> source-owned shared pinned host slot
  -> target GPU
```

The source process creates a shared-memory staging pool at `VRAM_SEG` registration time. The target
process maps that shared memory after loading source metadata, registers its local virtual mapping
with CUDA, and copies from the source-owned shared slot into the target GPU after a READY message.

This preserves the important optimization proven in the remote staged path: source D2H can be
prefetched before any target-side slot is acquired. The source copies into its own shared staging
slot, waits for the D2H event, then sends READY with the source slot id and target GPU address. The
target H2D reads from the mapped source shared slot and ACKs when target GPU memory is updated.

#### Option B2: Target-Owned Shared Staging

This is the simpler lease-oriented path:

```text
source GPU
  -> target-owned shared pinned host slot
  -> target GPU
```

The target process creates the shared staging pool. The source maps the target pool and copies D2H
directly into a granted target slot. This matches the existing target-side lease model, but it has a
performance downside: the target slot is held while source D2H is still running. That is the exact
pipeline gap that `source_d2h_prefetch` removed in the remote staged path.

The implementation should therefore use source-owned shared staging as the first optimized path and
keep target-owned shared staging as a comparison point or fallback when source-owned attach is not
available.

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
local_staging_fallback=true
local_staging_owner=source        # source | target
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
NIXL_UCX_LOCAL_STAGING_FALLBACK=1
NIXL_UCX_LOCAL_STAGING_OWNER=source
NIXL_UCX_LOCAL_STAGING_HOST_ID=<optional stable host id>
```

`NIXL_UCX_VRAM_LOCAL_STAGING=1` now automatically enables UCX VRAM staging if
`NIXL_UCX_VRAM_STAGING` or `vram_staging` was left off. The backend logs this once during engine
construction. This avoids the common misconfiguration where local staging is requested but
`VRAM_SEG` registration falls back to the direct UCX VRAM path.

Debug-only fault injection for validation:

```text
NIXL_UCX_LOCAL_STAGING_FORCE_ATTACH_FAIL=1
```

Set this only on the target process. It forces the local shared READY handler to ACK an attach error
so the initiator fallback path can be tested deterministically.

`auto` should prefer the shared pinned fast path when all of these are true:

- local and remote metadata report the same host id;
- the peer process can open and `mmap` the advertised shared staging object;
- the local CUDA runtime can `cudaHostRegister` the mapped pages;
- the selected owner mode is compatible with the transfer direction.

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
2. `/proc/sys/kernel/random/boot_id`, if available.
3. `/etc/machine-id`, if available.
4. hostname as a fallback.

Container deployments need special care. Two containers on the same physical host may have
different hostnames or private IPC namespaces. For shared pinned staging, both processes need to see
the same shared-memory object. Practical deployment options:

- run with shared IPC namespace, such as `--ipc=host`;
- or use a bind-mounted staging directory, such as `/tmp/nixl-local-staging`, that is visible to both
  containers.

If the shared object cannot be opened by the peer process, the backend must not silently use an
invalid pointer. It should fall back or fail clearly.

The target also constrains advertised local shared paths to the configured staging directory. It
canonicalizes the configured directory and source path, opens the file with `O_NOFOLLOW`, and
rejects READY messages whose path or slot does not match metadata previously loaded for that source
agent.

For Phase 1 validation through the existing UCX staged path, use UCX shared-memory transports such as
`UCX_TLS=sm,self` or explicit components like `UCX_TLS=posix,sysv,cma,self`, depending on the UCX
build. `self` alone is not enough for two separate Prefill/Decode processes.

## Metadata Design

Public staged local metadata should be versioned:

```text
magic = "NIXL_UCX_LOCAL_STAGED_V1"
mode = "shared_pinned"
owner = "source" or "target"
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

Raw host virtual addresses should not be treated as portable between processes. The backend must not
publish or consume a `cudaMallocHost` pointer from another process. Cross-process shared staging
requires a real shared object:

```text
shm_open or memfd_create or file-backed mmap
ftruncate to the staging pool size
mmap(MAP_SHARED)
cudaHostRegister(mapped address, size, ...)
```

Metadata should publish the shared object identity and slot offsets, not process-local host
pointers. Every process that attaches to the object gets its own virtual address and must use that
local mapping for CUDA copies.

## WRITE Protocol

### Source-Owned Shared Staging WRITE

The preferred optimized protocol is:

```text
initiator:
  create or reuse source-owned shared staging pool during VRAM registration
  publish shared object name/path and slot offsets in local metadata

target:
  load initiator metadata
  mmap initiator shared staging pool
  cudaHostRegister(mapped staging pages)

initiator:
  acquire source shared slot
  cudaMemcpyAsync D2H from source GPU into source shared slot
  wait/query D2H event
  STAGED_LOCAL_WRITE_READY(source_slot_id, source_slot_offset, target_gpu_addr, size)

target:
  validate source slot, target GPU address, transfer id, and chunk id
  cudaMemcpyAsync or cudaMemcpy H2D from mapped source shared slot into target GPU
  wait/query H2D completion
  STAGED_LOCAL_ACK(status)

initiator:
  release source shared slot after ACK
  mark chunk complete
```

This keeps source D2H prefetch independent from target slot availability.

### Target-Owned Shared Staging WRITE

The target-owned comparison path is:

```text
initiator:
  load target metadata
  mmap target shared staging pool
  cudaHostRegister(mapped staging pages)

target:
  reserve one target-owned shared slot after STAGED_LOCAL_SLOT_REQ
  STAGED_LOCAL_SLOT_GRANT(slot_id, lease_id, status)

initiator:
  cudaMemcpyAsync D2H from source GPU into mapped target shared slot
  wait/query D2H event
  STAGED_LOCAL_WRITE_READY(slot_id, lease_id, target_gpu_addr, size)

target:
  validate lease
  cudaMemcpyAsync or cudaMemcpy H2D from target shared slot into target GPU
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

Source-owned and target-owned local shared staging have different slot ownership.

For source-owned staging, the initiator owns the staging slots. A source slot is released only after
target ACK confirms H2D completion. This is enough to prevent the source from overwriting a shared
slot while the target is still reading from it.

For target-owned staging, the target process owns the staging slots. Slot state can reuse the
current staged slot lease model:

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

The target must validate the target GPU write regardless of owner mode:

```text
owner agent
transfer_id
chunk_id
target GPU address range
target GPU dev id
size <= slot_size
```

before starting H2D.

For source-owned staging, the READY message should additionally carry:

```text
source_slot_id
source_slot_generation
source_slot_offset
```

The target must validate that the advertised source slot belongs to the registered source shared
region. The source must not reuse that slot until ACK.

For target-owned staging, the READY message should additionally carry:

```text
target_slot_id
lease_id
```

The target must validate that the lease matches the granted target-owned slot.

## Pipeline Strategy

The local shared-pinned path has two data stages:

```text
D2H: source GPU -> shared pinned host slot
H2D: shared pinned host slot -> target GPU
```

There is no RDMA write and no UCX flush. The primary pipeline depth controls are:

- chunk size;
- shared staging slot count;
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

Source-owned staging is expected to be the better first performance target because it can overlap
source D2H before the target starts H2D. Target-owned staging should be measured as a comparison,
especially for NUMA-sensitive GPU pairs where target-local host pages might improve H2D enough to
offset the earlier target slot acquisition.

NUMA placement may dominate some GPU pairs. The long-term implementation should consider:

```text
local_staging_numa_policy=source|target|auto
```

The first implementation can use default allocation policy, but benchmarks must record GPU pair and
CPU/PCIe topology.

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
shm_open or file mmap + cudaHostRegister feasibility
shared mmap D2H and H2D bandwidth
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
- Verify `shm_open` or file-backed `mmap(MAP_SHARED)` can be registered with `cudaHostRegister`.
- Measure D2H and H2D bandwidth through the registered shared mapping.
- Measure current UCX staged behavior between two local processes.
- Confirm whether containers share `/dev/shm` or a bind-mounted staging directory.

The first local probe is `local_vram_staging_probe`. It validates file-backed shared mmap plus
`cudaHostRegister` independently from NIXL:

```text
local_vram_staging_probe \
  --mode single \
  --source-gpu 0 \
  --target-gpu 1 \
  --bytes 256M \
  --iters 4 \
  --path /dev/shm/nixl-local-vram-staging-probe.bin
```

For the two-process source-owned shape, start the target first:

```text
local_vram_staging_probe \
  --mode target \
  --target-gpu 1 \
  --bytes 256M \
  --iters 4 \
  --path /dev/shm/nixl-local-vram-staging-probe.bin
```

Then start the source:

```text
local_vram_staging_probe \
  --mode source \
  --source-gpu 0 \
  --target-gpu 1 \
  --bytes 256M \
  --iters 4 \
  --path /dev/shm/nixl-local-vram-staging-probe.bin
```

Both processes must see the same `--path`. If they run in different containers, use a shared IPC
namespace or a bind-mounted staging directory. The probe prints:

```text
d2h_gib_per_sec
h2d_gib_per_sec
serial_copy_gib_per_sec
verification=passed
```

The shared-pinned local fast path is worth implementing only if shared mmap registration succeeds
and D2H/H2D through the shared mapping are close to normal pinned-memory copy bandwidth.

Current Phase 0 result on `sglang-rdma-0-26`, GPU4 -> GPU5:

```text
nvidia-smi topo: GPU4/GPU5 = PIX
cudaDeviceCanAccessPeer(4, 5) = 0

single-process shared mmap probe, 64 MiB x 2:
  d2h_gib_per_sec = 24.43
  h2d_gib_per_sec = 25.10
  serial_copy_gib_per_sec = 12.38
  verification = passed

two-process source-owned shared mmap probe, 64 MiB x 2:
  source d2h_gib_per_sec = 24.43
  target h2d_gib_per_sec = 25.07
  verification = passed

two-process source-owned shared mmap probe, 512 MiB x 4:
  source d2h_gib_per_sec = 24.23
  target h2d_gib_per_sec = 25.17
  verification = passed
```

This validates the key prerequisite for the source-owned shared-pinned fast path on this host:
cross-process file-backed shared mappings can be CUDA-registered and sustain normal pinned-memory
D2H/H2D bandwidth.

### Phase 1: Local Correctness Through Existing UCX Staged Path

- Run the current staged WRITE smoke on one node with two processes and two GPUs.
- Force local UCX transports for validation, for example `UCX_TLS=shm,self` if compatible with the
  environment.
- Do not use `self` alone for two-process PD. Cross-process traffic needs shared-memory transports
  such as `sm`, `posix`, `sysv`, or `cma`.
- Verify correctness, transfer-only timing, and that network payload counters do not carry the main
  transfer.

Current Phase 1 result on `sglang-rdma-0-26`, GPU4 -> GPU5:

```text
NIXL_UCX_VRAM_STAGING=1
NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH=1
NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW=32

UCX_TLS=sm,self:
  failed during UCX endpoint creation
  reason: selected shared-memory transports do not provide the peer-failure/AM capability required
          by the current NIXL UCX endpoint setup on this UCX build

default UCX transport selection, 64 MiB x 2 transfers, 2 descriptors:
  transfer_loop_gib_per_sec = 16.15
  target verification = passed
```

This means the existing UCX staged path can be used as a same-node correctness fallback, but pure
UCX shared-memory transport is not available with the current endpoint configuration. The optimized
local shared-pinned path should therefore avoid depending on UCX for payload movement and use UCX
only as a control plane if needed.

## Current Implementation Status

The current branch has a first source-owned local shared staging implementation inside the UCX
backend:

- Added opt-in `vram_local_staging` / `NIXL_UCX_VRAM_LOCAL_STAGING`.
- Added `local_staging_shm_dir` / `NIXL_UCX_LOCAL_STAGING_SHM_DIR`, defaulting to `/dev/shm/nixl`.
- Added `local_staging_fallback` / `NIXL_UCX_LOCAL_STAGING_FALLBACK`, defaulting to enabled.
- `NIXL_UCX_VRAM_LOCAL_STAGING=1` now auto-enables `vram_staging` and logs a warning once if the
  user forgot to set `NIXL_UCX_VRAM_STAGING=1`.
- `registerMem(VRAM_SEG)` can now allocate source-owned file-backed shared staging pools, `mmap`
  them, `cudaHostRegister` the pool, and still UCX-register each slot for remote staged fallback.
- Public staged metadata now includes `host_id`, local shared enablement, shared region id, shared
  path, and mapping size.
- Initiator selects the local shared path only when local staging is enabled, the source slots are
  shared, target metadata reports the same host id, and target metadata also indicates local shared
  staging support.
- Added internal `STAGED_LOCAL_WRITE_READY` AM. The READY message carries the source shared path,
  source shared region id, source slot id, source slot generation, slot offset, mapping size, and
  target GPU address.
- Target validates READY against metadata previously loaded for the source agent before opening or
  mapping the advertised file. It checks source agent, region id, path, slot id, slot offset,
  mapping size, and chunk size.
- Target constrains source shared paths to `local_staging_shm_dir` using canonical path checks and
  opens attached files with `O_NOFOLLOW`.
- Target maps and CUDA-registers the source shared file, then performs H2D and ACKs only after
  target GPU memory has been updated.
- Added a target-side attachment cache, so each source shared file is opened, mmapped, and
  CUDA-registered once per loaded source region instead of once per chunk.
- Local shared attachments are removed when their remote staged metadata is unloaded, when a remote
  agent disconnects, or when the backend is destroyed.
- If local shared READY returns an ACK error and fallback is enabled, the initiator retries that
  chunk through the existing UCX staged host path using the already-filled source staging slot.
- Profile output now includes local shared chunks/bytes, local ACK errors, fallback count, target
  local errors, attachment cache hit/miss/failure counts, and attachment time.
- `deregisterMem` now rejects staged VRAM regions while any staging slot is active, avoiding
  unmap/unregister races with in-flight local or remote staged transfers.

Current limitations:

- Only `NIXL_WRITE` is supported.
- Target-owned shared staging is not implemented yet.
- Stale shared-file cleanup after process crash still needs robustness work.
- Local-to-UCX staged fallback is implemented for ACK-error recovery and has a debug-only attach
  failure injection path for deterministic validation.
- Target H2D still runs synchronously in the AM callback path.
- There is no READ support and no explicit application CUDA stream integration.

Current benchmark result on `sglang-rdma-0-26`, GPU4 -> GPU5:

```text
NIXL_UCX_VRAM_STAGING=1
NIXL_UCX_VRAM_LOCAL_STAGING=1
NIXL_UCX_LOCAL_STAGING_SHM_DIR=/dev/shm/nixl
NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH=1
NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW=32

64 MiB x 2 transfers, 2 descriptors, 16 MiB chunks:
  before target attach cache: transfer_loop_gib_per_sec = 5.46
  after target attach cache:  transfer_loop_gib_per_sec = 11.08
  target verification = passed

16 MiB x 1 transfer, 1 descriptor, profile enabled:
  slot_req_sent = 0
  slot_grant_success = 0
  rdma_write_posted = 0
  flush_posted = 0
  ready_sent = 1
  ack_received = 1
  target verification = passed

512 MiB x 8 transfers, 8 descriptors, 16 MiB chunks, 4 slots:
  transfer_loop_gib_per_sec = 22.62
  transfer_loop_sec = 0.1768
  target verification = passed

local staging disabled regression smoke, 16 MiB x 1:
  NIXL_UCX_VRAM_LOCAL_STAGING=0
  transfer_loop_gib_per_sec = 8.34
  target verification = passed

post-profile/fallback validation, 512 MiB x 8 transfers, 16 MiB chunks, 4 slots:
  transfer_loop_gib_per_sec = 21.99
  transfer_loop_sec = 0.1819
  target verification = passed

post-safety-boundary validation, 512 MiB x 8 transfers, 16 MiB chunks, 4 slots:
  includes metadata-bound READY validation, path boundary checks, and attachment lifecycle cleanup
  transfer_loop_gib_per_sec = 23.65
  transfer_loop_sec = 0.1691
  target verification = passed

auto-enable smoke, 16 MiB x 1:
  NIXL_UCX_VRAM_LOCAL_STAGING = 1
  explicit vram_staging / NIXL_UCX_VRAM_STAGING disabled
  backend auto-enabled UCX VRAM staging
  initiator rdma_write_posted = 0
  initiator flush_posted = 0
  target verification = passed

profile smoke, 16 MiB x 1:
  initiator local_shared_chunks = 1
  initiator local_shared_fallbacks = 0
  initiator rdma_write_posted = 0
  initiator flush_posted = 0
  target attach_cache_misses = 1
  target attach_failures = 0
  target attach_total_us = 6133
  target h2d_total_us = 652

forced attach-failure fallback smoke, 16 MiB x 1:
  target NIXL_UCX_LOCAL_STAGING_FORCE_ATTACH_FAIL = 1
  initiator local_shared_chunks = 1
  initiator local_shared_ack_errors = 1
  initiator local_shared_fallbacks = 1
  initiator rdma_write_posted = 1
  initiator flush_posted = 1
  initiator ready_sent = 2
  target local errors = 1
  target attach_failures = 1
  target verification = passed
```

The profile counters confirm the local shared path skips target slot lease, UCX RMA write, and UCX
flush. UCX is still used for connection setup and internal READY/ACK control messages.

## Current Optimization Position

The current local shared-pinned result is already close to the measured copy ceiling for this GPU
pair:

```text
raw shared-mmap D2H = 24.23 GiB/s
raw shared-mmap H2D = 25.17 GiB/s
NIXL local shared staged WRITE = 22.62 GiB/s
```

This is roughly 90 percent of the limiting D2H/H2D stage. At this point the first local staged
implementation is good enough to move from microbenchmark-only work to SGLang local PD validation.
Further optimization is still possible, but it should be profile-driven and should not change the
default path until it wins on the same transfer-only benchmark.

Current interpretation:

- The payload path is correct for local shared staging: there is no target slot lease, no UCX RMA
  write, and no UCX flush in the hot data path.
- UCX remains a control plane for setup, READY, and ACK.
- If attach fails on the target, the initiator can fall back per chunk to the UCX staged host path
  without rerunning source D2H.
- The remaining gap to raw D2H/H2D bandwidth is expected from chunk scheduling, READY/ACK latency,
  source slot lifetime until ACK, and synchronous target H2D in the AM callback path.
- Larger chunks are not automatically better. The current 16 MiB chunk size keeps enough scheduling
  granularity to overlap D2H and H2D.
- UCX HCA selection and network routing are not relevant to the local shared-pinned payload path.

Optimization priority after SGLang local PD smoke:

1. Use the local staged profile counters to split source D2H time, target H2D time, READY-to-ACK
   time, source slot wait, local fallback count, and attachment cache behavior.
2. If READY-to-ACK or target callback time dominates, test a local target H2D worker with reused
   CUDA streams and event pools. Keep synchronous H2D as the default unless the worker improves
   transfer-loop bandwidth.
3. Verify robust same-host fallback behavior under forced attach failure: if shared attach or
   `cudaHostRegister` fails, the existing UCX staged path should take over or fail clearly when
   fallback is disabled.
4. Add stale shared-file cleanup and guard against deregistering a VRAM region while local staged
   transfers are still in flight.
5. Evaluate a global per-device shared staging pool if SGLang registers many VRAM regions, to avoid
   excessive pinned-memory growth.
6. Implement target-owned shared staging only as an A/B comparison path. Source-owned remains the
   preferred path because it preserves source D2H prefetch.

## SGLang `shm_pinned` Reference

The local SGLang branch `pr1-shm-pinned-sync-kv-aux` contains a working single-node pinned-shm PD
backend under:

```text
python/sglang/srt/disaggregation/shm_pinned/
```

The relevant implementation points are:

- The decode side creates a POSIX shared-memory data buffer, a metadata buffer, and semaphores for
  free/ready/slot locking.
- Both decode and prefill map the same shared-memory object and call `cudaHostRegister` on their
  local virtual mapping.
- The data ring is resident for the session. Shared memory and CUDA host registration are done at
  setup/attach time, not per transfer chunk.
- The hot path is synchronous:
  `prefill D2H into shared slot -> post READY -> decode H2D from shared slot -> post FREE`.
- The implementation is target-owned: decode owns the ring slots and prefill waits for a free slot
  before D2H.
- The default SGLang parameters are `slot_count=32` and `chunk_tokens=512`.
- The PR1 scope is KV plus aux transfer only. State transfer is explicitly not supported.

Benchmark notes from that branch on a two RTX 4090 PCIe/no-NVLink machine:

```text
Qwen3-8B-FP8, 1 prefill + 1 decode, ShareGPT:
  output=300: shm_pinned total throughput +19.2 percent over NIXL/UCX loopback
  output=100 SLO frontier: +36.8 percent conservative, +52.9 percent aggressive
  TTFT: shm_pinned is 11x to 32x lower under QPS scan
```

The useful lessons for NIXL local staging are:

- Persistent shared pinned staging is the right shape. The NIXL local shared path should keep
  staging pools and peer attachments resident after `VRAM_SEG` registration/metadata attach.
- Cross-process shared mappings plus `cudaHostRegister` are already validated in a real SGLang
  deployment, matching the NIXL probe result.
- Attachment caching matters. Registering or mapping per chunk would destroy the latency advantage.
- A CPU protocol test is valuable. The SGLang unit test monkeypatches the CUDA memcpy path with
  `ctypes.memmove` to test slot metadata, semaphore ordering, and cleanup without GPU CI.
- End-to-end PD measurements should include TTFT and SLO frontier, not only raw transfer bandwidth.
  The SGLang result shows the transfer backend can move the bottleneck from handoff/TTFT to decode
  compute.

The parts that should not be copied directly into NIXL:

- The SGLang backend is application-specific and transfers page-indexed KV chunks. NIXL needs to
  keep descriptor-based `VRAM_SEG` semantics and should not depend on SGLang KV page layout.
- The SGLang ring is target-owned. That is simple and works, but it holds a target slot while source
  D2H is running. The NIXL source-owned fast path is preferred because it preserves source D2H
  prefetch and only requires target ACK before source slot reuse.
- POSIX semaphores are not required for the NIXL control plane. NIXL can continue using backend
  control messages and request state, with shared memory used only for the payload slots.
- SGLang's `chunk_tokens=512` is a page/token-level parameter. NIXL should continue tuning in bytes
  and currently uses 16 MiB chunks as the measured sweet spot.

### Phase 2: Shared Pinned Pool Registration

- Add source-owned shared staging pool allocation at `VRAM_SEG` registration time.
- Support POSIX shared memory or file-backed mmap staging objects.
- Register mapped pages with CUDA using `cudaHostRegister`.
- Publish versioned local staged metadata.
- Add peer attach, mmap, CUDA host registration, and detach paths.
- Keep target-owned shared pool support as an optional comparison path after source-owned works.

### Phase 3: Local Shared-Pinned WRITE

- Add same-host detection.
- Add local staged request states.
- Implement source-owned one chunk:
  `source D2H into source shared slot -> READY -> target H2D -> ACK`.
- Release the source slot only after ACK.
- Reuse target-side lease validation only for the target-owned comparison path.
- Ensure completion waits for target H2D.

### Phase 3B: Target-Owned Comparison Path

- Implement one chunk:
  `request target slot -> source D2H into target shared slot -> READY -> target H2D -> ACK`.
- Compare target-owned against source-owned on the same GPU pair and chunk/slot settings.
- Keep target-owned disabled by default unless it wins on a specific topology.

### Phase 4: Multi-Chunk Pipeline

- Add chunk scheduling for source-owned shared slots.
- Add a source local-ready queue so the scheduler does not busy poll when all source slots are full.
- Drive a steady pipeline:
  `D2H event ready -> READY -> H2D event ready -> ACK -> source slot release`.
- Add target slot request window only for target-owned mode.
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
- In target-owned mode, target slots are held while source D2H is in progress. This is why
  source-owned mode is preferred for the first performance implementation.
- Source-owned mode requires source slot lifetime to extend until target ACK. A source crash can
  leave the target with a mapped shared object that needs cleanup.
- NUMA placement can make source-owned and target-owned staging perform differently on different GPU
  pairs.
- Without application stream integration, correctness depends on the application respecting NIXL
  post/completion ordering.

## Immediate Next Steps

1. Commit the current local shared-pinned staging implementation and probe once the local diff is
   reviewed.
2. Run SGLang local PD smoke with:
   `NIXL_UCX_VRAM_STAGING=1`,
   `NIXL_UCX_VRAM_LOCAL_STAGING=1`,
   `NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH=1`, and 16 MiB chunks.
3. Confirm the SGLang local PD path uses `NIXL_WRITE`, compatible descriptor shapes, and no staged
   handle repost pattern.
4. Run with `NIXL_UCX_STAGING_PROFILE=1` and use the local staged counters to decide whether target
   H2D worker/event-pool work is worth enabling.
5. Add stale shared-file cleanup before wider local PD use.
