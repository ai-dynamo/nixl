# NIXL EP v2

This fork adapts DeepEP V2's BF16/FP8 dispatch and BF16 combine to a fixed
launch grid. Direct and hybrid modes share one kernel family; kernels read
the mode and active block count from a device topology descriptor. The
NIXL/UCX transport supports planned membership changes between
completed dispatch/combine pairs, including replay of retained CUDA graphs.

## Quick start

### Requirements

- Hopper (SM90) or newer GPUs with the required TMA instructions.
- PyTorch 2.14 or newer, with a matching Python and CUDA installation.
- CUDA 13.0 or newer, supporting the target GPU and the selected NIXL/UCX builds.
- NIXL and UCX builds with the GPU device API (`nixlAtomicAdd` and UCX
  device channels), including CUDA IPC support.
- RDMA development headers on JIT hosts, including `infiniband/mlx5dv.h`
  from rdma-core or OFED, for UCX's GPU RDMA device implementation.
- `nvidia-ml-py>=13.580.65` (installed with the package).
- NVLink within scale-up domains and RDMA between domains.

### Installation

Build from the NIXL repository root with NIXL EP enabled:

```bash
meson setup build -Dbuild_nixl_ep=true -Ducx_path=/path/to/ucx/install
meson compile -C build
meson install -C build
```

`nixl_cuda_arch_list` selects extension targets (for example `100` for
B200/GB200); communication kernels are JIT-compiled for the running GPU.
The build uses fmt headers from the system or the pinned Meson fallback.
`build_nixl_ep` enables both EP implementations and defaults to false.
This runtime requires CUDA 13; manylinux wheel builds default EP off on CUDA 12.
With EP disabled, its build prerequisites are not required. The wheel builder
accepts `--build-nixl-ep --torch-versions 2.14` and merges the extensions produced
for each Torch version. The container builder enables EP by default on CUDA 13.

The public import is `nixl.runtime.ep`, routed through the selected NIXL CUDA
backend wheel. For a Meson installation, put its Python site-packages and
`src/bindings/python/nixl-meta` on `PYTHONPATH`; installed wheels handle this.
NIXL's wheel builder includes the matching Torch-versioned extension and
NIXL/UCX device headers. Explicit header overrides remain available through
`NIXL_EP_NIXL_ROOT_DIR` and `NIXL_EP_UCX_ROOT_DIR` and must match the loaded libraries.

Runtime settings use `NIXL_EP_*`; compiler macros retain `EP_*`. Set
`NIXL_EP_NUM_TOPK_IDX_BITS=32` or `64` when building to choose the routing-index
width. The JIT uses the extension's compiled width; this setting has no
effect at runtime.

Construction defaults `UCX_UD_LINGER_TIMEOUT` to `1s` to avoid matching a
replacement process to a stale wire-up endpoint. This process-wide default
affects subsequent UCX initialization; set the variable before construction
to override it.

UCX transport channels per peer default to 64. For tuning, set
`NIXL_EP_NIXL_NUM_CHANNELS` to a value from 1 through 64 before construction.

### Dispatch and combine

BF16 combine requires `hidden` divisible by 256, including pairs that use
FP8 or NVFP4 dispatch.

With NIXL's metadata backend configured as described below, a process group `group`
containing two NVLink domains of eight GPUs in domain-contiguous rank order,
a shared `torch.distributed.Store` named `store`, and CUDA tensors `x` (BF16,
`[tokens, hidden]`), `topk_idx` (`nixl.runtime.ep.topk_idx_t`, `[tokens, topk]`) and
optional `topk_weights` (float32, the same shape):

```python
from nixl.runtime.ep import ElasticBuffer

domains, positions = 2, 8
world_rank = group.rank()
world = list(range(group.size()))
buffer = ElasticBuffer(
    world_rank=world_rank,
    max_num_scaleout_ranks=domains,
    max_num_scaleup_ranks=positions,
    store=store,
    num_max_tokens_per_rank=4096,
    hidden=7168,
    num_topk=8,
    num_experts_per_rank=8,  # Fixed physical routing slots per GPU.
    explicitly_destroy=True,
)
buffer.connect_ranks([rank for rank in world if rank != world_rank])
buffer.set_active_ranks(world)
recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
    x, topk_idx, topk_weights=topk_weights,
)
# Replace recv_x with the local experts' BF16 output before combining.
combined_x, _ = buffer.combine(recv_x, handle)
buffer.destroy()
```

Construction does not use the Python Store or wait for peers. Peer connection exchanges
metadata; activation selects the participating ranks. Communication and
destruction are collective over the committed active world.
When using `async_with_compute_stream=True`, wait on the returned event
before consuming the result. Deterministic sorting is supported only in eager
execution. Use `do_cpu_sync=False` for graph capture;
receive tensors then have maximum capacity, with actual counts in the
handle. See the `ElasticBuffer` method docstrings for expanded layouts and
FP8 scale-factor inputs.

## Topology and graph replay

The main dispatch/combine launch grid defaults to 24 blocks.
`NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS` selects an even grid size from 2 through 64
before construction. Each launch reads the active block count from the
topology descriptor; surplus blocks return. Keep that count unchanged
throughout a dispatch/combine pair, including its epilogues. Topology
changes are collective, between completed pairs, with updates and replay
ordered on the same stream or through explicit event dependencies.

`dispatch(num_sms=0)` remembers its top-k for automatic
SM selection. Subsequent `set_active_ranks` calls with
the default `num_active_sms=0` recompute the estimate for the proposed
membership using cached bandwidth information. Commit publishes the new
topology and count together. An explicit dispatch or topology SM count
pins manual selection until a successful `dispatch(num_sms=0)` enables
automatic selection again. Before the first automatic dispatch, topology
updates retain the existing count.

The selection policy belongs to the buffer: its most recent successful
dispatch selects automatic or manual mode for all graphs using it.
Participants must agree on that policy and its estimator inputs; joining
ranks establish their policy through their normal dispatch warm-up/capture.
Captured non-deterministic dispatch/combine graphs use the updated active count
without recapture. Their backing allocations cover the full launch grid;
Python handle fields and tensor shapes still describe capture-time metadata.
The estimator runs on the host during topology updates, never during graph
replay; no profiling or correctness checks are added to communication.

Dispatch/combine require at least two active GPUs, checked before launch
and again inside the kernels for graph replay. Smaller cohorts may still
bootstrap or tear down. `num_experts_per_rank` fixes the physical routing
slot capacity per GPU at construction; peers, including joiners, must agree on it.
Choose it for the largest per-GPU expert placement across all intended
topologies, including shrink, rather than just the initial placement.
Local expert metadata and `cumulative_local_expert_recv_stats` retain that
fixed length through shrink/expand. Without a CPU sync, received token
storage covers the declared maximum world, including expanded outputs and
alignment padding, so one captured graph can reuse those allocations.

The model's logical expert count is separate from these physical slots.
Unused slots may remain unrouted; the transport does not replicate experts
or create weights. The caller owns the mapping from logical experts to
physical slots and supplies the corresponding routing weights.

For S = `num_experts_per_rank` slots per GPU, the public expert ID is
`world_rank * S + local_expert_idx`, where `0 <= local_expert_idx < S`.
IDs range from zero through `max_num_scaleout_ranks * max_num_scaleup_ranks * S - 1`.
Slot `e` belongs to world rank `e // S`, with coordinates
`divmod(e // S, max_num_scaleup_ranks)`. IDs stay fixed across membership and
extent changes, including graph replay; holes are not packed away. `-1` means
no selection, and selections owned by inactive ranks are dropped. The caller
does not renumber IDs solely because an extent changes, but still owns logical
expert placement and routing weights. Internal source metadata continues to
use committed-extent rank encoding. The SM traffic estimate uses active-rank
counts times S, excluding holes.

`connect_ranks` prepares peer views; `set_active_ranks` validates and activates
the requested membership and prepared views. After excluding departing peers
from the active world, `disconnect_ranks` retires them and activates the
remaining peer views before returning. An empty disconnect does nothing.
Size the buffer for its maximum world at construction. Active ranks form a
product of active domains and positions within those domains. New-peer
connections may run while the committed graph replays. Activation and peer
retirement run outside graph capture. Use one CUDA device per process. Serialize control operations
on each buffer and select that device in any background setup thread. Before
activation or peer retirement, every old-world rank must finish
queued CUDA work, call `buffer.barrier(with_cpu_sync=True)` to drain the
transport, and complete a host rendezvous before any peer departs. Joining
ranks construct their buffers independently, then call `connect_ranks` with
their peers' stable world IDs. Mid-launch rank failure and transport recovery
are unsupported.

An idle joiner can commit its future topology locally and capture its graph
before cutover. Capture must skip eager communication and replay until the
old world has drained and all new-world ranks have activated. Construction
precompiles the barrier without launching it; dispatch/combine still require
two active GPUs.

`connect_ranks` exchanges agent names and region descriptors through the supplied Store,
publishing under `NIXL_EP/<world_rank>` and deleting it after the requested peers
have read it and native preparation finishes. Already-connected peers are ignored;
if no new peers remain, the call does nothing and needs no Store. New peers must request each
other in the same exchange. World IDs must be unique within the Store, and
all participants must finish an exchange before reusing its keys.

NIXL's metadata manager publishes and fetches transport metadata. Configure its
backend before buffer construction; for TCPStore, set `NIXL_TCPSTORE_ENDPOINT=host:port`
on every rank to a running TCPStore server. This can be the server backing the
Python Store and must remain available through buffer destruction.
Native publication, fetch and readiness checks release the GIL, allowing another Python thread to replay the
active graph. Publications remain available to future joiners until buffer
destruction. The manager starts its backend connection asynchronously during
construction without waiting for peers.

Invalid inputs and incompatible mappings are refused before activation.
Failures after transport mutation terminate the process; restart it instead
of attempting recovery.

## API changes from upstream V2

- Import `nixl.runtime.ep`. The component has no separate `__version__`;
  its release version is the installed NIXL distribution version.
  Root exports are `ElasticBuffer`, `EPHandle`, `EventHandle`, `EventOverlap`,
  and `topk_idx_t`; wildcard imports exclude initialization helpers.

- This fork exposes V2 dispatch/combine only. Legacy `Buffer`/`Config`,
  NVSHMEM, Engram, PP, AGRS and CPU-segment APIs are removed.
- Construction takes a stable `world_rank` and both maximum world dimensions;
  it creates only that rank's local buffer. Connect peers explicitly, then activate
  the intended world with `set_active_ranks`.
- Constructor keyword `num_experts_per_rank` fixes physical routing slots
  per GPU; it must be positive for dispatch/combine (zero permits a
  barrier-only buffer). `num_experts` is removed from `dispatch`,
  `get_theoretical_num_sms`, `EPHandle`, and native dispatch/combine; slot
  IDs span the declared maximum world. The cached `dispatch(handle=...)` path is
  removed, along with `allow_hybrid_mode`, the buffer's `group` and alignment
  exports.
- Constructor `num_bytes` is removed. Allocation is calculated from the workload
  settings and maximum world dimensions; barrier-only buffers are sized internally.
  `get_buffer_size_hint` uses explicit maxima for local memory planning, and `buffer.num_bytes`
  reports the communication allocation excluding workspace.
- `combine` takes no top-k weights and returns `(combined_x, event)`.
  Upstream used the optional weight return for routing-weight gradients in
  dispatch backward; this fork omits that path. It is independent of cached
  dispatch, which upstream used for combine backward.
- `combine(num_sms=...)` is removed: dispatch and combine use the topology
  descriptor's active block count throughout each pair, so combine has no
  independent SM setting. Set the count through dispatch or topology updates
  between completed pairs. Communicator, QP and service-level arguments are removed.
- Constructor and size-hint `use_fp8_dispatch=True` reserve smaller FP8
  dispatch capacity; peers must use matching capacity settings. These buffers
  accept FP8/NVFP4 and reject BF16 dispatch. Default BF16-capacity buffers
  accept all three wire formats, selected by dispatch input tensors.
- Active and connected world-rank queries replace static domain
  attributes/accessors; topology helpers are internal. `get_physical_domain_size` and the
  `barrier(sequential=)` argument are removed.
- `get_theoretical_num_sms` drops the zero-only `num_scaleout_topk` argument;
  its second positional argument is now `rdma_gbs`. It returns the raw estimate,
  before the launch-grid limit. `EPHandle.deterministic_sort` uses the handle's
  channel list instead of accepting a separate one. The constant-true private
  `_C.is_sm90_compiled` query is removed.

## NVFP4 dispatch

Construct the buffer with the model's logical `hidden`. Pass `dispatch`
an already quantized `(codes, scales)` tuple: contiguous uint8 codes shaped
`[tokens, hidden // 2]` and contiguous float8_e4m3fn or uint8 scales shaped
`[tokens, hidden // 16]`, with `hidden` divisible by 64. Scale storage offsets
must be divisible by four; clone unaligned scale slices before dispatch. Each code byte holds
two e2m1 values, with the even element in its low nibble. Dispatch returns
the codes unchanged and row-major float8_e4m3fn block scales. Combine remains
BF16 and requires the logical hidden size divisible by 256; use this stricter
alignment for a complete dispatch/combine pair.

No new constructor flag is needed: either existing capacity setting fits
NVFP4, and `use_fp8_dispatch=True` reserves the smaller one. Dispatch does
not quantize, apply scales, or transmit a global scale. The caller must
supply the consuming GEMM's global scale and any required scale swizzle;
using one shared GEMM scale requires inputs quantized with that same scale.
Column-major scale output is unsupported for NVFP4.

## Automatic combine optimization

Hybrid combine across multiple active domains automatically uses incremental
completion for destinations with at most 16 original input tokens, letting
the reduction epilogue wait for each token's contributions. Larger batches
and single-domain hybrid mode use final channel completion; direct mode is
unchanged. Selection uses actual input counts, independently
for each rank, rather than buffer capacity. Traversal remains forward.

The threshold follows BF16 measurements on a 2×8 GB200 topology: incremental
completion materially improved batches 1 and 16, while larger batches had
small gains or regressions. Other configurations have not been performance
validated.

## Tests

Run tests from `runtime/ep` after installing the package.
The existing tests are `tests/elastic/test_ep.py` and
`tests/elastic/test_barrier.py`. Their launcher uses
`nixl_ep_v2/utils/envs.py:init_dist`; configure it for your cluster before
running distributed tests. Run either script with `--help` for its options.
All GPU tests require the metadata backend configuration above. For tests that
serve a TCPStore on `MASTER_ADDR` / `MASTER_PORT`, set this before launching:

```bash
export NIXL_TCPSTORE_ENDPOINT="${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-8361}"
```

For tests launched with `--store-host` / `--store-port`, use that external server's
address instead. The metadata server must stay alive until all buffers are destroyed.

`tests/elastic/test_adaptive_elastic.py` accepts an inline JSON plan or a
plan file through `--plan`. `--benchmark`, `--kineto` and `--feed-pairs`
enable call, kernel and transition timing; `--save-output PATH` saves
per-rank results. Payload byte counts exclude transport overhead and are
not per-NIC bandwidth measurements.

`tests/elastic/test_elastic_warmup.py` prepares fresh joining buffers while
survivors replay, checks the first replay after each switch and serves
through departure cleanup. It reports completion gaps during preparation,
cutover and retirement. `--max-completion-gap-ms 1000` applies a one-second
acceptance bound; the default reports gaps without enforcing a bound.
Joiner capture is prepared ahead of cutover; its first execution is timed
after activation. `tests/elastic/test_runtime_tokens.py` checks BF16 payloads
and JIT reuse across token capacities.

## Phase profiling

Set `NIXL_EP_ADAPTIVE_PROFILE=1` before importing NIXL EP v2 to record diagnostic
timestamps. Dispatch and combine keep their existing arguments. Profiling
adds one buffer-owned timestamp table and a descriptor after the workspace
controls; existing control offsets and kernel signatures are unchanged.
Without the flag, timestamp recording is compiled out.

`buffer.read_profile()` returns named CPU tensors indexed by `[block, warp]`.
Each nonzero value is a device global-timer timestamp in nanoseconds; zero
means that warp did not reach the phase. The call synchronizes the device,
copies the samples, and completes their reset before returning. Discard
warm-up samples with one read, then read after each isolated pair or graph
replay. The method cannot run inside graph capture. Resetting between
replays also removes samples from blocks or roles that become inactive.

For eager and graph-replay profiles with the elastic test's correctness checks:

```bash
NIXL_EP_ADAPTIVE_PROFILE=1 python tests/elastic/test_adaptive_elastic.py --profile --num-tokens 16
```

`PROFILE` reports the first and last observed phase boundaries relative to
the earliest entry of that operation, in microseconds, plus the number of
writers. These are samples from different warps, not additive phase durations:

- Entry and barrier markers sample each participating warp. Dispatch count
  reduction/exchange markers sample block 0's notify warps; prefix markers
  sample its two prefix-sum warps.
- `copy_ready` and `reduce_ready` follow the PDL dependency wait. Incremental
  combine can still wait for remote contributions inside the reduction body.
- `*_submitted` markers mean the role issued its work; they do not wait for
  asynchronous stores to complete. `combine.main_end` likewise does not imply
  remote readiness when incremental completion is active.

No per-token counters, profiling atomics or extra kernel barriers are added.
Instrumentation still changes execution, so use profiling to locate work and
waits, and use the uninstrumented `PERF` measurements for performance claims.

## JIT cache

The JIT cache defaults to `~/.nixl_ep_v2`; `NIXL_EP_JIT_CACHE_DIR` selects another
directory. NIXL and UCX dependency headers are not fingerprinted by the cache key.

Token capacity is a runtime kernel argument, so changing it between eager
calls or new captures reuses the compiled specialization. Each existing
graph retains the capacity supplied at capture. Compilation uses `cuobjdump`
once to store the kernel symbol with the CUBIN; cached loads read that symbol
file. Clear caches created before `kernel.symbol` was introduced, or select a
fresh cache directory when upgrading.

When changing NIXL or UCX builds between runs, including build configuration or
device-header changes, clear the existing cache or set `NIXL_EP_JIT_CACHE_DIR` to a
directory dedicated to the new dependency builds before starting the process.
This requirement also applies to persistent or shared caches across container
images.
