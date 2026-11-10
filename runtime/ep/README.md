# NIXL EP v2

This fork adapts DeepEP V2's BF16/FP8 dispatch and BF16 combine to a fixed
launch grid. Direct and hybrid modes share one kernel family; kernels read
the mode and active block count from a device topology descriptor. The
NIXL/UCX transport supports planned membership changes between
completed dispatch/combine pairs, including replay of retained CUDA graphs.

## Quick start

### Requirements

- Hopper (SM90) or newer GPUs with the required TMA instructions.
- PyTorch 2.10 or newer, with a matching Python and CUDA installation.
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
`build_nixl_ep` enables EP and defaults to false. CUDA 12 builds V1 and skips
V2 with a warning; CUDA 13 builds both. With EP disabled, its build
prerequisites are not required. The wheel builder accepts
`--build-nixl-ep --torch-versions 2.10` and merges the extensions produced
for each Torch version. The container builder enables EP by default;
`--no-build-nixl-ep` disables both versions.

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

### Dispatch and combine

Given an initialized process group `group` and CUDA tensors `x` (BF16,
`[tokens, hidden]`), `topk_idx` (`nixl.runtime.ep.topk_idx_t`, `[tokens, topk]`) and
optional `topk_weights` (float32, the same shape):

```python
from nixl.runtime.ep import ElasticBuffer

buffer = ElasticBuffer(
    group,
    num_max_tokens_per_rank=4096,
    hidden=7168,
    num_topk=8,
    explicitly_destroy=True,
)
recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
    x, topk_idx, num_experts=256, topk_weights=topk_weights,
)
# Replace recv_x with the local experts' BF16 output before combining.
combined_x, _ = buffer.combine(recv_x, handle)
buffer.destroy()
```

Construction is collective over the constructing cohort; communication and
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

`dispatch(num_sms=0)` remembers its expert count and top-k for automatic
SM selection. Subsequent `stage_topology` / `set_active_ranks` calls with
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
The estimator runs on the host during topology staging, never during graph
replay; no profiling or correctness checks are added to communication.

Dispatch/combine require at least two active GPUs, checked before launch
and again inside the kernels for graph replay. Smaller cohorts may still
bootstrap or tear down. If `cumulative_local_expert_recv_stats` is supplied
during capture, allocate at least `num_experts // 2` entries; its address
stays fixed as the local-expert count changes across shrink and expansion.

`connect_ranks` / `disconnect_ranks` stage peer views; `stage_topology`
stages membership. `commit_topology` activates both. The convenience method
`set_active_ranks` stages and commits membership immediately. Exclude
departing peers in a committed topology before disconnecting them.
Size the buffer for its maximum world at construction. Active ranks form a
product of active domains and positions within those domains. Use one CUDA
device per process. Serialize control operations on each buffer and select
that device in any background setup thread. Before membership or
transport-view changes, every old-world rank must finish
queued CUDA work, call `buffer.barrier(with_cpu_sync=True)` to drain the
transport, and complete a host rendezvous before any peer departs. Joining
cohorts use an explicit Store and fresh `bootstrap_id`; see `publish` and
`connect_ranks` for metadata exchange. Mid-launch rank failure and transport
recovery are unsupported.

For process replacement, use a fresh shared publication `key_prefix`, or
rendezvous after every target's current process has completed `publish`
before calling `connect_ranks`. Store keys survive departing processes;
`bootstrap_id` isolates cohort construction, not publication keys.

Invalid inputs and incompatible mappings are refused before activation.
Failures after transport mutation terminate the process; restart it instead
of attempting recovery.

## API changes from upstream V2

- Import `nixl.runtime.ep`. The component has no separate `__version__`;
  its release version is the installed NIXL distribution version.
  Root exports are `ElasticBuffer`, `EPHandle`, `EventHandle`, `EventOverlap`,
  `topk_idx_t` and `get_logical_domain_size`; wildcard imports exclude
  initialization helpers.

- This fork exposes V2 dispatch/combine only. Legacy `Buffer`/`Config`,
  NVSHMEM, Engram, PP, AGRS and CPU-segment APIs are removed.
- `dispatch`'s `num_experts` is no longer optional and appears immediately
  after `topk_idx`. Upstream already required it for a fresh dispatch; the
  omitted value came from a cached handle. The cached `dispatch(handle=...)`
  path is removed, along with `allow_hybrid_mode`, the buffer's `group` and
  alignment exports.
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
  reject BF16 dispatch. Default BF16-capacity buffers accept either wire
  format, selected by dispatch input tensors.
- Committed masks/extents and connected world ranks replace static domain
  attributes/accessors. `get_physical_domain_size` and the
  `barrier(sequential=)` argument are removed.
- `get_theoretical_num_sms` drops the zero-only `num_scaleout_topk` argument;
  its third positional argument is now `rdma_gbs`. It returns the raw estimate,
  before the launch-grid limit. `EPHandle.deterministic_sort` uses the handle's
  channel list instead of accepting a separate one. The constant-true private
  `_C.is_sm90_compiled` query is removed.

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

When changing NIXL or UCX builds between runs, including build configuration or
device-header changes, clear the existing cache or set `NIXL_EP_JIT_CACHE_DIR` to a
directory dedicated to the new dependency builds before starting the process.
This requirement also applies to persistent or shared caches across container
images.
