# DOCA_MEMOS splitting PoC

This extension targets the upstream DOCA_MEMOS backend. It splits logical
values into ordinary NVMe-KV operations without changing the NIXL API or copying
payloads. This is a fixed-geometry format, with the limitations below.

## Behavior

The backend expands each logical READ/WRITE descriptor into contiguous payload
slices. Each slice becomes an ordinary DOCA KV task, using the existing task
queue, submission backpressure and completion/error handling. There are no
payload copies, stored headers, manifests or internal metadata validation.

- Chunk 0 keeps the original resolved key, K.
- Continuation i uses BLAKE2b-128 of `"nixl-memos-chunk-v1" || key_length_u8 || K || i_le64`.
  This reuses the existing libb2 dependency and produces a 16-byte key.
- At initialization, discover the namespace maximum value length (MXVL) through
  DOCA and the kernel single-command transfer limit from the NVMe queue's
  `max_hw_sectors_kb`. Default chunk bytes are the smaller of these limits.
  Standard `/dev/ngXnY` and `/dev/nvmeXnY` names and resolvable aliases are
  supported; discovery uses `/sys/block/nvmeXnY/queue/max_hw_sectors_kb`.
- Optional backend initialization parameter `chunk_size` (advertised default `0`,
  meaning automatic discovery) overrides chunk bytes when nonzero,
  capped by both discovered limits. If the kernel limit is unavailable, an
  explicit override is required and is capped by MXVL; the caller must ensure
  it fits the actual transport. A zero namespace maximum is rejected.
  NIXLBench submits logical objects without specifying chunk size.
- EXIST checks only K using the upstream backend's device query.
- READ/WRITE succeed only when all children succeed. Missing reads return the
  upstream `NIXL_ERR_BACKEND` for both split and unsplit objects, whether some
  or all chunks are absent. Other child failures also fail the logical request.
  Upstream removed `ignore_read_not_found` and `query_mem_mode`; this patch does
  not reintroduce them or change NIXLBench.

**Key-collision assumption:** application keys are assumed to be independently
randomly generated, rather than deliberately selected to match internal keys.
Continuation keys are 128-bit BLAKE2b digests and share the physical key space
with application keys. Accidental collision risk depends on the total number of
stored application and continuation keys; for the intended random-key workload,
that risk is accepted as negligible.

This format does not isolate deliberately chosen application keys. In raw
hexadecimal mode, a caller can compute another object's continuation key and
write a separate object at that key, overwriting the continuation. A subsequent
read may return mixed data with a success status. Hash-input domain separation
does not prevent this chosen-key overwrite. Protection against such callers is
outside this PoC's scope. The accepted random-key assumption requires no key-format
change or data repopulation.

All participants must use the same effective chunk size, supported by their
interfaces, and data must have been written with that geometry. The backend does
not handle unaligned geometry, validate object generations/lengths, roll back partial writes, or clean up stale
continuations after a size change. Concurrent updates may mix chunks. EXIST can
succeed while a continuation is absent. A failed read may already have modified
part of the destination. Use freshly populated test values; existing unsplit
large values are not migrated.

## Verification

The backend unit suite compiles the actual backend and progress-engine sources
against the repository's DOCA mocks and compatible public DOCA KV headers.
Coverage includes multi-chunk byte-for-byte round trips and overwrites, short
final chunks, missing continuations, first-key EXIST, failed child submissions,
namespace key-width limits, request reuse and both progress modes. Transfer-limit
tests use a temporary sysfs fixture to cover discovery, explicit overrides and
missing or malformed limits.

Cancellation tests cover READ and WRITE with two chunks submitted and three
pending. Release remains deferred until submitted I/O drains; unsubmitted chunks
remain untouched, and the same request handle can subsequently be reused.
The tests also retain upstream cancellation, error-priority and parameter-parsing
coverage. Mocks establish these code-path properties, not device compatibility
or performance.

Earlier revisions passed live large-value round trips and task-pool backpressure
checks. Those historical results do not qualify the current upstream port; the
following Linux build and hardware checks must be rerun on the exact candidate.

## Simple live test plan

Use NIXLBench: direct FIO access to NVMe-KV bypasses this NIXL backend and cannot
test this implementation. An existing FIO setup remains useful as a control for
physical, unsplit transfers.

### Build in a compatible Linux SDK environment

Use a DOCA KV SDK compatible with this branch, plus the repository's normal build
dependencies (including libb2 and GoogleTest). From this worktree:

```bash
meson setup build-poc --buildtype=debugoptimized -Dbuild_tests=true \
  -Denable_plugins=DOCA_MEMOS -Dwith_trace=false \
  --prefix="$PWD/install-poc" --libdir=lib
meson compile -C build-poc
meson test -C build-poc --suite doca_memos --print-errorlogs
meson install -C build-poc

meson setup buildbench-poc benchmark/nixlbench \
  -Dnixl_path="$PWD/install-poc" --libdir=lib
meson compile -C buildbench-poc
```

These Linux build commands are a plan and have not been run locally. Ensure the
benchmark loads this installation's library and plugin, rather than the existing
system plugin:

```bash
export LD_LIBRARY_PATH="$PWD/install-poc/lib:${LD_LIBRARY_PATH:-}"
export NIXL_PLUGIN_DIR="$PWD/install-poc/lib/plugins"
```

### Smoke test

Use an approved test namespace/device. A single-process storage test does not require etcd.
Hugepages must already be available. Replace the device and NGUID
variables with the test environment's values. Run one size per invocation so the
READ path repopulates values using the same geometry before reading them.

```bash
NIXL_LOG_LEVEL=DEBUG ./buildbench-poc/nixlbench \
  --backend=DOCA_MEMOS --op_type=READ \
  --doca_memos_device_name="$KV_DEVICE" --doca_memos_nguid="$NGUID" \
  --total_buffer_size=67108864 \
  --start_block_size=16777216 --max_block_size=16777216 \
  --start_batch_size=2 --max_batch_size=2 --num_threads=1 \
  --storage_enable_direct=true --use_hugepages=true \
  --num_iter=10 --warmup_iter=1 --large_blk_iter_ftr=1 --enable_pt=false
```

The existing READ path first writes the keys, retrieves them once, then performs
warmup and timed reads. Require successful prepopulation, retrieval and benchmark
completion. Its verification messages check operation success, **not payload
equality**; `--check_consistency` is unsupported for DOCA_MEMOS. Byte comparison is
covered by the mock round-trip tests above; a live byte comparison would require
a small dedicated NIXL client or a separate benchmark verification change.

Read the effective `chunk_size` (C bytes) from the backend initialization log;
automatic discovery does not guarantee 8 MiB. Choose aligned logical sizes from
the following matrix, with both `--enable_pt=false` and `true`. Set
`--total_buffer_size` to at least four times the logical size for batch 2
(the smoke command above uses a 64 MiB budget for 16 MiB objects).

| Logical size | Physical chunks per object |
| ---: | ---: |
| C / 2 (aligned) | 1 |
| C | 1 |
| 2C | 2 |
| 2C + C / 2 (aligned) | 3, short final chunk |
| 3C | 3 |

For batch 2, require the debug preparation message to show twice the per-object
physical task count. Repeat **3C**, batch 2, with `--doca_memos_num_tasks=2`:
six physical tasks exceed the two-task pool and exercise backpressure. Ensure
sufficient approved storage and host memory before selecting these sizes.
Compare the two unsplit sizes against the unchanged build, using identical
sizes and settings; these short runs are not performance qualification.

Missing-child and partial-write error tests are already deterministic in the
mock suite. Avoid deleting production keys merely to reproduce these failures.
