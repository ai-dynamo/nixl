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
staging_slot_request_window=32
staging_batch_flush=false
staging_target_h2d_worker=false
```

Suggested environment variables:

```text
NIXL_UCX_VRAM_STAGING=1
NIXL_UCX_STAGING_CHUNK_SIZE=16M
NIXL_UCX_STAGING_SLOTS=4
NIXL_UCX_STAGING_FORCE_PROGRESS_THREAD=1
NIXL_UCX_STAGING_CUDA_COPY_STREAMS=1
NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW=32
NIXL_UCX_STAGING_BATCH_FLUSH=0
NIXL_UCX_STAGING_TARGET_H2D_WORKER=0
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
- Added target-side slot leases. The initiator now requests a target staging slot for every chunk,
  and the target grants a concrete `slot_id` and `lease_id` before the RDMA write.
- Added lease validation before target H2D and release messages for failure cleanup after a grant.
- Added local slot state tracking so one process does not reuse a staging slot before its D2H/RDMA
  use has completed.
- Added concurrent C++ smoke coverage for multiple in-flight staged WRITE requests, including two
  initiator processes writing different offsets into one target VRAM registration.
- Fixed UCX data-plane device selection so the backend accepts the documented `ucx_devices`
  parameter and remains compatible with the older `device_list` parameter. Both forms use
  comma-separated lists and trim whitespace around each device name.
- Added transfer-only timing output and staged profile counters for initiator control messages,
  D2H/write/flush/READY/ACK timing, and target H2D callback timing.
- Added bounded slot-request scheduling through `staging_slot_request_window` /
  `NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW`.
- Added an experimental `staging_batch_flush` / `NIXL_UCX_STAGING_BATCH_FLUSH` switch. It is
  disabled by default because current measurements show it is a regression with synchronous target
  H2D and four target slots.
- Added an experimental `staging_target_h2d_worker` /
  `NIXL_UCX_STAGING_TARGET_H2D_WORKER` switch. It moves target H2D out of the UCX active-message
  callback into a target-side worker thread, then sends ACK after H2D completion. It is disabled by
  default because current measurements do not show a throughput gain.
- Added staged WRITE support for multiple descriptor pairs. Each local/remote descriptor pair must
  have the same nonzero length; the backend splits each pair into staged chunks and uses the same
  target-side slot lease protocol per chunk.
- Added C++ smoke coverage for the prepped transfer path (`prepXferDlist + makeXferReq`) used by
  SGLang's fast KV transfer path.
- Kept staged mode opt-in. The default direct UCX path remains unchanged.

Current limitations:

- `NIXL_READ` is not implemented for staged mode.
- Staged v1 uses synchronous CUDA copies for correctness.
- Staged v1 uses UCX worker 0 for internal ready/ACK messages.
- Target H2D is still performed synchronously in the internal active-message handler by default.
- Batch flush is experimental and disabled by default.
- Target-side H2D worker is experimental and disabled by default.
- Malformed READY cleanup, lease timeout, remote disconnect recovery, and poison/reclaim policy for
  `ERROR` slots are still future robustness work.

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
STAGED_SLOT_REQ
STAGED_SLOT_GRANT
STAGED_SLOT_RELEASE
STAGED_WRITE_READY
STAGED_ACK
```

The key protocol property is that the initiator must not choose the target staging slot from public
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

Current ownership is target-side for incoming writes:

- local source slots are reserved in `nixlUcxStagedPrivateMetadata` as `LOCAL_D2H`,
- target incoming slots are reserved in the target's `nixlUcxStagedPrivateMetadata` as
  `REMOTE_RESERVED`,
- the target assigns a `lease_id` for each granted chunk,
- `STAGED_WRITE_READY` must present the matching owner, transfer id, chunk id, slot id, lease id,
  GPU address, device id, and size before H2D begins.

The same slot state table covers local D2H use and remote incoming RDMA use so a process that is
both initiator and target cannot reuse one physical host slot for incompatible purposes.

Current state model:

```text
FREE
LOCAL_D2H
REMOTE_RESERVED
REMOTE_H2D
ERROR
```

Current lease fields:

```text
state
owner_agent
transfer_id
chunk_id
lease_id
gpu_addr
gpu_dev
size
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
slot state is REMOTE_RESERVED
```

Every failure path should release or poison affected slots so later transfers cannot observe stale
state. Current Phase 3A covers the normal initiator-side failure paths after a grant and H2D
failure on the target. Remaining malformed-message and timeout cleanup are future robustness work.
Current covered paths:

- initiator D2H failure after grant sends `STAGED_SLOT_RELEASE`,
- UCX write or flush failure after grant sends `STAGED_SLOT_RELEASE`,
- READY send failure after grant sends `STAGED_SLOT_RELEASE`,
- target H2D failure marks the slot `ERROR` and sends an error ACK,
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

The backend also accepts the older `device_list` key for compatibility. `ucx_devices` is preferred;
if it is unset or empty, `device_list` is used.

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

- Implemented chunk splitting for staged `VRAM_SEG` descriptor pairs.
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

Phase 3A has been implemented and smoke-tested with one initiator process using multiple in-flight
WRITE request handles. The Phase 2 pipeline proved a single large transfer could reuse slots safely
within one request handle; Phase 3A protects target staging slots across multiple request handles by
moving remote slot ownership into the target process.

Implemented changes:

1. Reject staged request repost until reset support exists.
   Ordinary UCX requests may be reposted after completion, but the staged handle currently owns
   chunk state such as `chunks`, `completedChunks`, and per-chunk grant/ACK state. The staged path
   returns `NIXL_ERR_NOT_ALLOWED` when a handle is posted outside the initial state.
2. Add `STAGED_SLOT_REQ`, `STAGED_SLOT_GRANT`, and `STAGED_SLOT_RELEASE`.
   The target grants concrete staging slots and assigns a `lease_id`; the initiator uses only the
   granted slot for the chunk RDMA write.
3. Validate the lease in `STAGED_WRITE_READY`.
   The target must verify owner, transfer id, chunk id, slot id, lease id, GPU range, and size
   before running H2D.
4. Release or poison slots on the covered Phase 3A failure paths.
   Initiator-side failure after a grant sends `STAGED_SLOT_RELEASE`; request release sends release
   for held remote leases; target H2D failure marks the slot `ERROR` and sends an error ACK.
5. Add concurrent smoke tests.
   Correctness has been shown for multiple in-flight WRITE requests from one initiator process.

Current internal slot lease structures:

```cpp
enum class StagedSlotState {
    FREE,
    LOCAL_D2H,
    REMOTE_RESERVED,
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
};
```

The C++ smoke test has been extended with:

```text
--concurrency N
--iters N
--bytes N
--chunk-size N
--slots N
--offset-stride N
--initiator-id N
--initiator-count N
--descriptors N
--prepped 0|1
--skip-desc-merge 0|1
```

Observed Phase 3A acceptance matrix on `sglang-rdma-0-26 -> sglang-rdma-0-41`:

```text
slots=1, concurrency=2,  bytes=16 MiB: passed
slots=1, concurrency=8,  bytes=4 MiB:  passed
slots=4, concurrency=8,  bytes=80 MiB: passed
slots=4, concurrency=16, bytes=1 MiB:  passed
```

The target allocates one larger GPU buffer. Each transfer writes a different offset, and the data
pattern encodes `initiator_id`, transfer index, and byte offset so target verification can detect
cross-transfer slot overwrites.

#### Phase 3B: Multi-Initiator Smoke

Phase 3B extends the C++ smoke test so one target can wait for multiple initiator agents and send
the same target VRAM descriptor to all of them. Each initiator writes a distinct offset range:

```text
global_transfer = (initiator_id - 1) * local_transfer_count + local_transfer
remote_offset = global_transfer * offset_stride
```

The target waits for all DONE notifications and verifies the whole target allocation with a pattern
that includes `initiator_id`, transfer index, and byte offset.

Observed Phase 3B acceptance matrix on `sglang-rdma-0-26 -> sglang-rdma-0-41`:

```text
initiators=2, slots=1, concurrency=1 each, bytes=16 MiB: passed
initiators=2, slots=1, concurrency=4 each, bytes=4 MiB:  passed
initiators=2, slots=4, concurrency=4 each, bytes=80 MiB: passed
```

All runs used the target control address `10.159.0.41`, `ucx_devices=mlx5_4,mlx5_5`, and
`UCX_TLS=rc,ud,self`.

Post-fix smoke matrix after enabling `ucx_devices` parsing:

```text
slots=1, concurrency=2, bytes=16 MiB:                  passed
slots=1, concurrency=8, bytes=4 MiB:                   passed
slots=4, concurrency=8, bytes=80 MiB:                  passed
initiators=2, slots=1, concurrency=4 each, bytes=4 MiB:  passed
initiators=2, slots=4, concurrency=4 each, bytes=80 MiB: passed
```

For the single-initiator 80 MiB x 8 run after the parameter fix, mlx5 counters increased on both
selected rails:

```text
0-26 mlx5_4 port_xmit_data: +85,427,614
0-26 mlx5_5 port_xmit_data: +85,401,600
0-41 mlx5_4 port_rcv_data:  +85,427,882
0-41 mlx5_5 port_rcv_data:  +85,401,600
```

#### SGLang NIXL Path Audit

The installed SGLang package on `sglang-rdma-0-26` is `sglang==0.5.10` under
`/opt/venv/lib/python3.10/site-packages`. Its NIXL disaggregation path is:

```text
sglang/srt/disaggregation/nixl/conn.py
```

Observed properties:

- The KV cache, sliced KV cache, aux data, and state transfer paths all call
  `agent.initialize_xfer("WRITE", ...)`.
- No SGLang NIXL path found in this package calls `NIXL_READ`.
- Each transfer creates a fresh NIXL xfer handle, posts it once with `agent.transfer(handle)`, stores
  the handle, and polls it with `agent.check_xfer_state(handle)`. No repost/reset pattern was found.
- KV cache WRITE is not single descriptor in the general case. The code groups contiguous source and
  destination block indices, then emits descriptors for each layer and K/V tensor. For a standard
  MHA model with one contiguous block group, this is already one descriptor per layer for K plus one
  descriptor per layer for V. Additional non-contiguous block groups increase descriptor count.
- `send_kvcache_slice` can emit even more VRAM descriptors because it creates per-token head-slice
  descriptors for each layer K/V pair.
- Aux transfer uses DRAM descriptors and is unaffected by VRAM staging.
- State transfer can also use VRAM descriptors depending on `state_type`.

This means current staged WRITE correctness is aligned with SGLang on operation and handle
lifecycle. Phase 3C adds the missing descriptor-shape support: staged WRITE now accepts multiple
local/remote descriptor pairs, validates pair counts and sizes, and expands each pair into staged
chunks internally. This keeps SGLang's existing multi-descriptor WRITE requests on the same NIXL API
surface.

#### Phase 3C: Multi-Descriptor WRITE

Phase 3C has been implemented for staged WRITE. The operation remains WRITE-only and staged handle
repost remains rejected. The backend now accepts a descriptor list, splits every descriptor pair
into chunks, and runs the target-side slot lease protocol per chunk. Request completion and user
notification still wait for all descriptor chunks to receive target H2D ACKs.

The C++ smoke test now has `--descriptors N`, which splits every transfer's local and remote xfer
dlists into `N` descriptors while the registered VRAM region remains contiguous. It also has
`--prepped 1` for the `prepXferDlist + makeXferReq` path and `--skip-desc-merge 1` to keep the test
descriptor list from being merged back into one contiguous descriptor.

Observed Phase 3C acceptance matrix on `sglang-rdma-0-26 -> sglang-rdma-0-41`:

```text
descriptors=1, slots=1, concurrency=2, bytes=4 MiB:         passed
skip_desc_merge=1, descriptors=8, slots=4,
  concurrency=2, bytes=16 MiB:                               passed
skip_desc_merge=1, descriptors=4, slots=4,
  concurrency=2, bytes=80 MiB:                               passed
skip_desc_merge=1, descriptors=72, slots=4,
  concurrency=1, bytes=16 MiB:                               passed
initiators=2, skip_desc_merge=1, descriptors=8, slots=1,
  concurrency=2 each, bytes=16 MiB:                          passed
prepped=1, skip_desc_merge=1, descriptors=8,
  slots=4, concurrency=2, bytes=16 MiB:                      passed
prepped=1, skip_desc_merge=1, descriptors=72,
  slots=4, concurrency=1, bytes=16 MiB:                      passed
```

The `descriptors=4, bytes=80 MiB` case validates both multi-descriptor and multi-chunk behavior:
each 20 MiB descriptor is split across the 16 MiB staging chunk boundary.

Counter validation for a `skip_desc_merge=1, descriptors=4, bytes=80 MiB, concurrency=2` run with
`ucx_devices=mlx5_4,mlx5_5`:

```text
0-26 mlx5_4 port_xmit_data: +21,356,058
0-26 mlx5_5 port_xmit_data: +21,350,400
0-41 mlx5_4 port_rcv_data:  +21,356,324
0-41 mlx5_5 port_rcv_data:  +21,350,400
```

#### Phase 3D: Transfer-Only Bandwidth Measurement

The smoke test has been extended with timing output so staged bandwidth can be measured with a
PD-like scope. The `transfer_loop` timing starts after:

- the source and target VRAM buffers already exist,
- both VRAM regions have been registered with NIXL,
- staged pinned host slots have been allocated and registered,
- remote metadata and target descriptors have been exchanged.

It excludes CPU pattern generation, CUDA allocation, initial source-buffer population, metadata
handshake, full target verification, deregistration, and free. This is the closest current smoke
test proxy for the SGLang PD path where prefill and decode register long-lived VRAM pools at
startup and runtime transfers only write page/range descriptors inside those pools.

Observed transfer-only results on `sglang-rdma-0-26 -> sglang-rdma-0-41`, GPU0 to GPU0:

```text
payload:               4 GiB
descriptors:           8
prepped:               1
skip_desc_merge:       1
ucx_devices:           mlx5_4,mlx5_5
UCX_TLS:               rc,ud,self
poll_sleep_us:         0
```

Best observed setting:

```text
bytes=512 MiB, concurrency=8, iters=1
chunk_size=16 MiB, slots_per_gpu=4
transfer_loop_us=230,256
transfer_loop_gib_per_sec=17.37
target verification: passed
```

The corresponding mlx5 counter deltas were aligned with the 4 GiB payload and showed data-plane
traffic on both selected rails:

```text
0-26 mlx5_4 port_xmit_data: +547,548,283
0-26 mlx5_5 port_xmit_data: +546,570,240
0-41 mlx5_4 port_rcv_data:  +547,548,551
0-41 mlx5_5 port_rcv_data:  +546,570,240
```

Since the IB data counters are in 4-byte units, these deltas correspond to roughly 4.07 GiB of
wire-level counter growth across the two rails, including protocol/control overhead.

The shell `real` time for this smoke should not be used as the staged data-path bandwidth because
it includes benchmark setup. In the same run the initiator spent about 5.56 seconds in
`buffer_prepare`, mostly generating and uploading a 4 GiB byte pattern. The staged data-path timing
was only about 0.23 seconds:

```text
initiator setup_sec:          0.58
initiator buffer_prepare_sec: 5.56
initiator register_sec:       0.03
initiator metadata_sec:       0.29
initiator transfer_loop_sec:  0.23
initiator cleanup_sec:        0.01
```

Tuning results so far:

```text
chunk=16 MiB,  slots=4, concurrency=4:  15.50 GiB/s
chunk=16 MiB,  slots=4, concurrency=8:  17.37-17.75 GiB/s
chunk=16 MiB,  slots=4, concurrency=12: 16.21 GiB/s
chunk=16 MiB,  slots=4, concurrency=16: 16.62 GiB/s
chunk=64 MiB,  slots=4, concurrency=8:  16.25 GiB/s
chunk=128 MiB, slots=4, concurrency=8:  16.93 GiB/s
chunk=256 MiB, slots=4, concurrency=8:  16.78 GiB/s
chunk=16 MiB,  slots=8, concurrency=4:  13.08 GiB/s
```

Bounded slot-request window measurements:

```text
profile enabled, window=unbounded: 15.82 GiB/s, slot_req_sent=13,800, grant_inprog=13,544
profile enabled, window=4:         15.08 GiB/s, slot_req_sent=256,    grant_inprog=0
profile enabled, window=8:         14.87 GiB/s, slot_req_sent=425,    grant_inprog=150
profile enabled, window=16(auto):  15.64 GiB/s, slot_req_sent=1,055,  grant_inprog=871
profile enabled, window=32:        18.52 GiB/s, slot_req_sent=3,113,  grant_inprog=2,868
profile off,     window=32:        18.34-18.63 GiB/s
```

The best current tuned point is `chunk_size=16 MiB`, `slots_per_gpu=4`,
`staging_slot_request_window=32`, and around eight concurrent WRITE requests. A window exactly equal
to the target slot count removes most control-plane retries but underfills the pipeline. A wider
window keeps more work available and is faster even though the target still returns some
`NIXL_IN_PROG` slot grants.

Batch-flush measurements with `staging_slot_request_window=32`:

```text
batch_flush=false, profile off: 18.63 GiB/s
batch_flush=true,  profile off: 11.27 GiB/s
batch_flush=true,  profile on:  11.03 GiB/s
```

The current batch-flush experiment reduces flush calls from 32 per 512 MiB request to about 8, but
it holds target staging leases longer and delays WRITE_READY/H2D/ACK into bursts. With synchronous
target H2D and only four target slots this is a net regression, not a path to 20 GiB/s. Keep
`staging_batch_flush=false` unless specifically profiling this behavior.

Target H2D worker measurements with `staging_slot_request_window=32` and
`staging_batch_flush=false`:

```text
target_h2d_worker=false, profile off: 18.97 GiB/s
target_h2d_worker=true,  profile off: 18.60 GiB/s
target_h2d_worker=true,  profile on:  18.04 GiB/s
```

The H2D worker does what it is designed to do: target callback time dropped from about 169 ms total
for 256 READY messages to about 95 us total. H2D itself remained about 168 ms total for the same
4 GiB payload. Because the worker currently performs synchronous H2D on one thread, this removes
UCX callback blocking but does not create more H2D overlap. Keep
`staging_target_h2d_worker=false` for the current first version; use
`NIXL_UCX_STAGING_TARGET_H2D_WORKER=1` only when profiling callback interference.

Host and local PCIe baselines measured in the same containers:

```text
ib_write_bw mlx5_4:       196.08 Gb/s
ib_write_bw mlx5_5:       196.08 Gb/s
ib_write_bw dual rail:    about 392 Gb/s aggregate
GPU pinned H2D/D2H:       about 25.2 / 24.6 GiB/s per host
staged transfer-only:     about 17.4 GiB/s
bounded-window staged:    about 18.6 GiB/s
```

The remaining gap is therefore not explained by pinned memory registration or TCP fallback. The
current staged v1 path still uses synchronous `cudaMemcpy` and performs target H2D in the UCX
active-message callback by default. The next performance work should focus on:

1. converting D2H/H2D to `cudaMemcpyAsync` plus CUDA events,
2. using one or more target-side CUDA streams so H2D can overlap with UCX progress and ACK send,
3. replacing busy retry scheduling with a lease-grant queue or target-side pending queue,
4. revisiting flush coalescing only after target H2D is asynchronous and slots are not held idle.

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
