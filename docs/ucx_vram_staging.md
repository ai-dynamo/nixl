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

# UCX VRAM Staging Design

This document describes a proposed staged VRAM transfer mode for the NIXL UCX backend. The target
use case is high-throughput GPU-to-GPU transfer on systems where GPU Direct RDMA is unavailable or
not effective, such as consumer GPUs connected to RDMA-capable hosts.

The mode keeps the NIXL application-facing memory type as `VRAM_SEG`, while the backend internally
moves data through pinned host staging buffers:

```text
source GPU -> source pinned DRAM -> UCX host RDMA -> target pinned DRAM -> target GPU
```

The main goal is to avoid changing inference framework code that already uses the standard NIXL
agent API.

## Goals

- Preserve the existing NIXL API surface for applications.
- Keep the backend name `UCX` and continue exposing `VRAM_SEG`.
- Use host DRAM RDMA for the network stage instead of relying on direct GPU memory RMA.
- Return transfer completion only after the target GPU memory is updated.
- Preserve user notification ordering and visibility semantics.
- Start with `NIXL_WRITE` support, then add `NIXL_READ` only if required by the application path.

## Non-Goals

- Do not change the NIXL core request model in the first implementation.
- Do not fork or modify UCX as the first implementation step.
- Do not require SGLang or another application to parse backend metadata internals.
- Do not attempt to make the first version fully optimized; correctness comes first.
- Do not rely on UCX's direct VRAM registration path when staging is enabled.

## Existing Path

The NIXL core selects one backend that is common to the local and remote descriptor lists, then
delegates request lifecycle operations to that backend:

- `prepXfer`
- `postXfer`
- `checkXfer`
- `releaseReqH`

The current UCX backend declares support for `DRAM_SEG` and `VRAM_SEG`. During `registerMem`, it
registers the provided address with UCX and publishes a packed rkey. During transfer, it calls
UCX RMA operations against the descriptor addresses and metadata:

```text
local descriptor addr + local UCX memh
remote descriptor addr + remote UCX rkey
```

On consumer GPU systems this can fall back to TCP or otherwise fail to use the desired RDMA path.
Staged mode must therefore change the UCX backend's behavior for `VRAM_SEG` registration and
transfer. It is not enough to add CUDA copies around the existing direct VRAM RMA path.

## Proposed Backend Mode

Add a staged mode inside the UCX backend. The UCX plugin keeps the same public backend name and
memory capabilities, but uses a different internal engine or request implementation when staging is
enabled.

Suggested backend parameters:

```text
vram_staging=true
staging_chunk_size=16777216
staging_slots_per_gpu=4
staging_force_progress_thread=true
staging_cuda_copy_streams=1
```

Suggested environment variables:

```text
NIXL_UCX_VRAM_STAGING=1
NIXL_UCX_STAGING_CHUNK_SIZE=16M
NIXL_UCX_STAGING_SLOTS=4
NIXL_UCX_STAGING_FORCE_PROGRESS_THREAD=1
NIXL_UCX_STAGING_CUDA_COPY_STREAMS=1
```

The existing direct UCX path remains the default. Staged mode is opt-in.

## Implementation Status

Current local implementation status:

- Added opt-in staged configuration through UCX backend parameters and environment variables.
- Added staged `VRAM_SEG` registration that allocates pinned host slots, registers them as
  `DRAM_SEG` with UCX, and publishes staged metadata instead of a direct GPU rkey.
- Added internal UCX active messages for staged WRITE ready and ACK. These messages are separate
  from user-visible notifications.
- Added a staged `NIXL_WRITE` correctness path:
  `cudaMemcpy` D2H, UCX host RDMA write, UCX flush, target H2D, target ACK, then NIXL completion.
- Added multi-chunk WRITE scheduling with per-chunk ready/ACK messages, local slot reuse, and
  remote slot reuse within one transfer after the corresponding target H2D ACK.
- Added local slot busy tracking so one initiator does not reuse a staging slot before its RDMA
  write has completed.
- Kept staged mode opt-in. The default direct UCX path remains unchanged.

Current limitations:

- `NIXL_READ` is not implemented for staged mode.
- The staged WRITE path still supports exactly one descriptor, but the descriptor may be larger
  than one staging slot and will be split into chunks.
- Staged v1 uses synchronous CUDA copies for correctness.
- Staged v1 uses UCX worker 0 for internal ready/ACK messages.
- Remote slot leasing is not implemented yet. Multi-initiator or highly concurrent writes can
  collide on target staging slots and require the Phase 3 slot lease protocol.
- Target H2D is still performed synchronously in the internal active-message handler.

## Registration Model

When staged mode is disabled, `VRAM_SEG` registration keeps the current behavior.

When staged mode is enabled, `registerMem(VRAM_SEG)` should not publish a direct GPU rkey as the
transfer target. Instead, it should:

1. Store the original GPU address, length, and device id.
2. Allocate or attach a pinned host staging pool for the relevant GPU device.
3. Register the host staging slots with UCX as host memory.
4. Publish staged metadata containing the remote staging slots and enough information to map a
   descriptor offset back to the original GPU region.

The descriptor address remains the original GPU address. The staged backend computes:

```text
gpu_offset = descriptor.addr - registered_gpu_base
```

and uses that offset when copying to or from the correct GPU region.

### Private Metadata

Private metadata should include:

```text
gpu_base
gpu_len
gpu_dev_id
staging_pool
ucx_mem_handles
free_slot_queue
cuda_copy_streams
cuda_events
```

The private metadata is used by local request execution and is not serialized directly.

### Public Metadata

Public metadata should be versioned so staged peers can reject incompatible metadata early:

```text
magic = "NIXL_UCX_STAGED_V1"
mode = "vram_host_staging"
gpu_base
gpu_len
gpu_dev_id
slot_size
slot_count
slots:
  slot_id
  host_addr
  rkey_blob
capabilities:
  write=true
  read=false
```

The first implementation can publish a fixed slot pool per registered region. A later version can
deduplicate pools across registrations on the same GPU.

## WRITE Protocol

The first implementation should support `NIXL_WRITE`.

For each descriptor pair:

1. Split the transfer into chunks no larger than `staging_chunk_size`.
2. Acquire a local staging slot.
3. Acquire or reserve a remote staging slot.
4. Copy source GPU data to local pinned DRAM.
5. RDMA write local pinned DRAM into the remote staging slot.
6. Notify the target backend that the chunk is present in remote staging.
7. Target backend copies remote staging into the target GPU.
8. Target backend sends an ack after H2D completion.
9. Initiator releases local and remote slot state.
10. The request completes only after all chunk acks have been received.

Single-chunk correctness mode can use conservative synchronization before introducing overlap.

### Internal Control Messages

UCX active messages can be reused for internal staged protocol messages. These messages should be
separate from user notifications.

Current implemented message types:

```text
STAGED_WRITE_READY
STAGED_ACK
```

Phase 3A should add target-side slot lease messages:

```text
STAGED_SLOT_REQ
STAGED_SLOT_GRANT
STAGED_SLOT_RELEASE
```

The key protocol change is that the initiator must not choose the target staging slot from public
metadata by itself. The target owns its staging slots and must grant a lease before the initiator
can RDMA-write into one.

Suggested WRITE chunk flow after Phase 3A:

```text
initiator -> target: STAGED_SLOT_REQ(transfer_id, chunk_id, gpu_addr, gpu_dev, size)
target -> initiator: STAGED_SLOT_GRANT(transfer_id, chunk_id, slot_id, lease_id, status)
initiator:          D2H into local staging slot
initiator:          UCX host RDMA write into granted target slot
initiator:          UCX flush
initiator -> target: STAGED_WRITE_READY(transfer_id, chunk_id, slot_id, lease_id,
                                        gpu_addr, gpu_dev, size)
target:             validate lease, H2D from target slot into target GPU
target -> initiator: STAGED_ACK(transfer_id, chunk_id, lease_id, status)
initiator:          release local slot and mark chunk done
```

`STAGED_SLOT_RELEASE` is used when an initiator has received a grant but cannot complete the chunk
path, for example D2H failure, UCX write failure, UCX flush failure, READY send failure, request
release, cancellation, or remote disconnect handling.

The target side must process internal messages even when the application is not calling
`getNotifs`. Staged mode should therefore require a progress thread or provide a backend-owned
progress path that is always driven while transfers are in progress.

## Request Completion Semantics

NIXL request completion must mean that application-visible memory has been updated.

For staged `NIXL_WRITE`, `checkXfer` must return `NIXL_SUCCESS` only after:

- all local D2H copies for the request have completed,
- all UCX host RDMA operations and required flushes have completed,
- all target H2D copies have completed,
- all target acks have been received by the initiator.

If the request has a user notification, the user notification must be delivered only after target
H2D completion. Delivering it after host RDMA completion would be incorrect because the target GPU
could still contain old data.

## Slot Ownership

The staging pool needs explicit ownership tracking. Publishing staging slots in metadata is not
enough, because multiple in-flight transfers or multiple initiators can otherwise write into the
same slot.

Current Phase 2 ownership is only transfer-local:

- local source slots are protected by `nixlUcxStagedPrivateMetadata::slotBusy`,
- target slots are tracked by `nixlUcxStagedBackendReqH::remoteSlotBusy`,
- `remoteSlotBusy` is private to one request handle.

This is enough for one transfer's chunk pipeline, but not enough for concurrent transfers. Two
handles in the same initiator process, or two different initiators, can both select target slot 0
because they have independent local views of remote slot availability.

For SGLang-style concurrent use, target-side slot leasing is required. Slot state should live in
the target's `nixlUcxStagedPrivateMetadata`, because that process owns the pinned host staging
slots. The same state table should cover local D2H use and remote incoming RDMA use so a process
that is both initiator and target cannot reuse one physical host slot for incompatible purposes.

Suggested state model:

```text
FREE
LOCAL_D2H
REMOTE_RESERVED
REMOTE_RDMA_READY
REMOTE_H2D
ERROR
```

Suggested lease fields:

```text
state
owner_agent
transfer_id
chunk_id
lease_id
gpu_addr
gpu_dev
size
deadline
```

`STAGED_WRITE_READY` must validate the lease before H2D:

```text
slot_id exists
lease_id matches
owner_agent matches
transfer_id matches
chunk_id matches
size <= slot_size
gpu range is inside the registered VRAM region
slot state is REMOTE_RESERVED or REMOTE_RDMA_READY
```

Every failure path must release or poison affected slots so later transfers cannot observe stale
state. At minimum:

- initiator D2H failure after grant sends `STAGED_SLOT_RELEASE`,
- UCX write or flush failure after grant sends `STAGED_SLOT_RELEASE`,
- READY send failure after grant sends `STAGED_SLOT_RELEASE`,
- target malformed READY releases a matching lease or marks the slot `ERROR`,
- target H2D failure releases the slot and sends an error ACK,
- initiator request release sends release for any chunk that holds a target lease but has not
  received ACK.

Target H2D should not run while holding the global staged region mutex. The callback should find the
region and lease, copy the needed slot pointer and GPU range, update slot state, release the region
lock, perform H2D, then reacquire the slot lock to release or poison the lease and send ACK.

## CUDA Synchronization

The NIXL API does not currently carry an application CUDA stream. Staged mode therefore cannot
automatically order backend copies with application kernels on arbitrary streams.

The first version should use a conservative synchronization policy:

- Ensure source GPU writes are visible before D2H.
- Ensure target H2D completes before ack.
- Treat NIXL completion as the application-visible synchronization point.

Possible first-version implementation choices:

- Use backend-owned nonblocking CUDA streams plus explicit events.
- Use `cudaStreamSynchronize` at key boundaries for correctness testing.
- Document that the application must not modify the source buffer while the request is active and
  must not consume the target buffer until NIXL completion or notification.

A later API extension may pass CUDA streams or events into NIXL, but that would likely require
application changes and is outside the first staged implementation.

## READ Protocol

`NIXL_READ` is more complex and should not block the first `WRITE` implementation.

A staged read would require:

1. Initiator asks target to stage the source GPU range.
2. Target performs D2H into its staging slot.
3. Target sends ready.
4. Initiator RDMA reads from target staging into local staging.
5. Initiator performs H2D into local GPU destination.
6. Initiator releases the target slot.

This requires additional control messages:

```text
STAGE_READ_PREPARE
STAGE_READ_READY
STAGE_READ_RELEASE
```

Implement this only after confirming the target application path requires `NIXL_READ`.

## Phased Plan

### Phase 0: Host RDMA Baseline

Before implementing staging, confirm that host DRAM transfers on the target systems use RDMA rather
than TCP.

Checks:

- NIXL `DRAM_SEG -> DRAM_SEG` transfer succeeds between the two hosts.
- UCX transport selection uses an RDMA-capable transport.
- NIC counters increase during the run.
- Host DRAM throughput is high enough to justify the staged pipeline.

If host DRAM transfer falls back to TCP, staged VRAM mode will not solve the performance problem.

Observed baseline on `sglang-rdma-0-26 -> sglang-rdma-0-41`:

```text
Command shape: /workspace/nixl_test/nixl_vram_bw.py
Environment:  UCX_TLS=rc,ud,self
Payload:      256 MiB x 64 iterations

DRAM WRITE:   367.226 Gb/s
DRAM READ:    262.030 Gb/s
```

The `UCX_TLS=rc,ud,self` setting excludes TCP. mlx5 port counters increased during the WRITE run on
the expected transmit/receive direction, so the host staging network stage is viable on these
hosts.

Control plane and data plane can remain separated. In the smoke tests below, NIXL listener metadata
used the ordinary `bond0` address (`10.159.0.41` on the target), while UCX data transfer was
restricted to the high-speed RDMA devices with `ucx_devices=mlx5_4,mlx5_5` and
`UCX_TLS=rc,ud,self`. This does not require changing `ip route`.

### Phase 1: WRITE Correctness

Implement the smallest correct staged write:

- staged `VRAM_SEG` registration,
- fixed pinned host staging slots,
- single chunk per request,
- internal ready and ack messages,
- conservative CUDA synchronization,
- request completion after target H2D ack.

This phase may restrict staged mode to one in-flight request per peer.

Current smoke-test status:

```text
Hosts:        sglang-rdma-0-26 -> sglang-rdma-0-41
Control IP:   10.159.0.41 (target bond0)
UCX devices:  mlx5_4,mlx5_5
UCX TLS:      rc,ud,self
Operation:    NIXL_WRITE, single VRAM descriptor

Python smoke:
  4 MiB WRITE:  target GPU verification passed
  16 MiB WRITE: target GPU verification passed

C++/CUDA smoke, no Python or torch dependency:
  1 MiB WRITE:  target GPU byte verification passed
  16 MiB WRITE: target GPU byte verification passed
  64 MiB WRITE: target GPU byte verification passed, 4 chunks x 16 MiB
  80 MiB WRITE: target GPU byte verification passed, 5 chunks x 16 MiB, slot reuse required
```

Additional correctness smoke matrix:

```text
4 B WRITE:        passed
49,380 B WRITE:   passed
1 MiB WRITE:      passed
4 MiB WRITE:      passed
16 MiB WRITE:     passed
16 MiB repeat:    passed
```

For the 16 MiB run, the mlx5 counters increased on both selected rails. On the initiator, both
`mlx5_4` and `mlx5_5` transmit counters increased by roughly 2.135M counter units; on the target,
the corresponding receive counters increased by roughly the same amount. Since the IB data counters
are word-scaled, the two rails together account for the 16 MiB payload plus protocol overhead.

The staged path was also checked with a direct-path negative control. With the same host-only UCX
build and `vram_staging=false`, `VRAM_SEG` registration failed while UCX attempted to register the
GPU address directly:

```text
ucp_mem_map: Input/output error
ibv_reg_mr(address=<gpu_addr>, length=1048576) failed: Bad address
target registerMem: NIXL_ERR_BACKEND
```

The successful `VRAM_SEG` transfers therefore depend on the staged registration path, where NIXL
allocates pinned host slots and registers those host slots with UCX instead of publishing a direct
GPU rkey.

### Phase 2: Multi-Slot Pipeline

Add overlap:

```text
chunk 0: D2H
chunk 0: RDMA, chunk 1: D2H
chunk 0: H2D, chunk 1: RDMA, chunk 2: D2H
```

Initial tuning defaults:

```text
chunk_size = 16 MiB or 32 MiB
slots_per_gpu = 4
cuda_copy_streams = 1 per direction
```

Tune chunk size and slot count based on PCIe copy bandwidth, RDMA bandwidth, and CPU overhead.

Current Phase 2 status:

- Implemented chunk splitting for one large `VRAM_SEG` descriptor.
- Added `chunk_id` to internal `STAGED_WRITE_READY` and `STAGED_ACK` messages.
- Initiator can keep multiple chunks in flight up to the available local and remote staging slot
  count.
- Initiator releases a local staging slot after the chunk's UCX write and flush complete.
- Initiator releases a remote staging slot after the target sends the chunk ACK, which is sent
  after target H2D completes.

Observed 80 MiB counter validation with `chunk_size=16 MiB` and `slots_per_gpu=4`:

```text
0-26 mlx5_4 port_xmit_data: +10,675,817
0-26 mlx5_5 port_xmit_data: +10,675,200
0-41 mlx5_4 port_rcv_data:  +10,676,032
0-41 mlx5_5 port_rcv_data:  +10,675,200
```

The two rails together account for the 80 MiB staged host RDMA payload plus protocol overhead.

### Phase 3: Robustness

Add production safety:

- transfer id allocation and wrap handling,
- target-side slot leasing,
- multi-peer slot ownership,
- timeout handling,
- remote disconnect handling,
- cancellation and abort,
- partial failure cleanup,
- telemetry for D2H, RDMA, H2D, and ack latency.

#### Phase 3A: Target-Side Slot Lease

Phase 3A should be completed before SGLang concurrent testing. The Phase 2 pipeline proves a single
large transfer can reuse slots safely within one request handle, but it does not protect target
staging slots across multiple request handles or initiators.

Recommended implementation order:

1. Reject staged request repost until reset support exists.
   Ordinary UCX requests may be reposted after completion, but the staged handle currently owns
   chunk state such as `chunks`, `completedChunks`, `nextChunkToPost`, and per-chunk ACK state. A
   conservative first patch should return `NIXL_ERR_NOT_ALLOWED` when a staged handle is posted
   outside the initial state.
2. Add `STAGED_SLOT_REQ`, `STAGED_SLOT_GRANT`, and `STAGED_SLOT_RELEASE`.
   The target grants concrete staging slots and assigns a `lease_id`; the initiator uses only the
   granted slot for the chunk RDMA write.
3. Validate the lease in `STAGED_WRITE_READY`.
   The target must verify owner, transfer id, chunk id, slot id, lease id, GPU range, and size
   before running H2D.
4. Release or poison slots on every failure path.
   Slot leaks will deadlock later transfers; stale slot reuse can corrupt target GPU data.
5. Add concurrent smoke tests.
   Correctness must be shown for multiple in-flight WRITE requests and at least two initiator
   processes writing different offsets into one target allocation.

Suggested internal slot lease structures:

```cpp
enum class StagedSlotState {
    FREE,
    LOCAL_D2H,
    REMOTE_RESERVED,
    REMOTE_RDMA_READY,
    REMOTE_H2D,
    ERROR,
};

struct StagedSlotLease {
    StagedSlotState state;
    std::string ownerAgent;
    uint64_t transferId;
    uint64_t chunkId;
    uint64_t leaseId;
    uintptr_t gpuAddr;
    uint64_t gpuDev;
    size_t size;
    std::chrono::steady_clock::time_point deadline;
};
```

The C++ smoke test should be extended with:

```text
--concurrency N
--iters N
--bytes N
--chunk-size N
--slots N
--offset-stride N
--initiator-id N
```

Suggested acceptance matrix:

```text
slots=1, concurrency=2,  bytes=16 MiB
slots=1, concurrency=8,  bytes=4 MiB
slots=4, concurrency=8,  bytes=80 MiB
slots=4, concurrency=16, bytes=1 MiB
two initiator processes writing distinct offsets into one target allocation
```

The target should allocate one larger GPU buffer. Each transfer writes a different offset, and the
data pattern should encode `initiator_id`, transfer index, and byte offset so target verification can
detect cross-transfer slot overwrites.

Suggested commit split:

```text
staging: reject staged request repost until reset is implemented
staging: add target-side slot lease control messages
staging: validate lease before H2D and release on failure
examples: add concurrent staged vram write smoke
```

### Phase 4: READ Support

Implement staged `NIXL_READ` only if the application path requires it.

## Implementation Locations

Likely files to modify:

```text
src/plugins/ucx/ucx_backend.h
src/plugins/ucx/ucx_backend.cpp
src/plugins/ucx/ucx_plugin.cpp
src/plugins/ucx/ucx_utils.h
src/plugins/ucx/ucx_utils.cpp
src/plugins/ucx/meson.build
```

The UCX engine factory can select a staged engine when `vram_staging=true`.

The NIXL core should remain unchanged for the first implementation because the current core already
delegates the whole transfer lifecycle to the selected backend.

## Open Questions

- Does the target application path use `NIXL_READ`, or is `NIXL_WRITE` sufficient?
- Does the application provide any implicit CUDA stream ordering before calling NIXL?
- How many concurrent transfers can exist per peer and per registered GPU region?
- What lease timeout should be used before a target slot is marked `ERROR` or reclaimed?
- Should target H2D remain in the UCX active-message callback for Phase 3A, or should it move to a
  target-side staging worker queue immediately after lease validation?
- Should staged mode force a progress thread, or can all needed progress be driven by existing
  application polling in the target process?
- Which UCX transports are selected for host DRAM transfers on the target hosts?

## Summary

The staged UCX mode is feasible because NIXL backend metadata is private to the backend and NIXL
request completion is delegated to the selected backend. The important boundary is that staged mode
must manage the whole data path and completion protocol itself:

```text
GPU D2H -> host RDMA -> remote H2D -> ack -> NIXL completion
```

The first implementation should be narrow: opt-in UCX staged mode, `NIXL_WRITE` only, conservative
synchronization, and correctness before pipelining.
