# NIXL EP

NIXL EP provides BF16/FP8/NVFP4 dispatch and BF16 combine using a fixed
launch grid. Direct and hybrid modes share one kernel family; kernels read
the mode and membership from a device topology descriptor. The
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
- NVLink within scale-up domains and RDMA between domains.
- `nvidia-ml-py>=13.580.65` for NVLink domain discovery during peer connection.

### Installation

Configure Meson from the NIXL repository root with `build_nixl_ep` enabled and
`ucx_path` pointing to the UCX installation, then compile and install the build.

`nixl_cuda_arch_list` selects extension targets (for example `100` for
B200/GB200); communication kernels are JIT-compiled for the running GPU.
The build uses fmt headers from the system or the pinned Meson fallback.
`build_nixl_ep` enables both EP implementations and defaults to false.
This runtime requires CUDA 13; manylinux wheel builds default EP off on CUDA 12.
With EP disabled, its build prerequisites are not required. The wheel builder
enables EP through `--build-nixl-ep`; `--torch-versions` selects the Torch versions
whose extensions are merged. The container builder enables EP by default on CUDA 13.

The public import is `nixl.runtime.ep`, routed through the selected NIXL CUDA
backend wheel. For a Meson installation, put its Python site-packages and
`src/bindings/python/nixl-meta` on `PYTHONPATH`; installed wheels handle this.
NIXL's wheel builder includes the matching Torch-versioned extension and
NIXL/UCX device headers. Explicit header overrides remain available through
`NIXL_EP_NIXL_ROOT_DIR` and `NIXL_EP_UCX_ROOT_DIR` and must match the loaded libraries.

Runtime settings use `NIXL_EP_*`; compiler macros retain `EP_*`. Set
`NIXL_EP_NUM_TOPK_IDX_BITS` to 32 or 64 when building to choose the routing-index
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

Configure NIXL's metadata backend as described below. Construct an `ElasticBuffer`
per GPU with its stable world rank, maximum topology dimensions, workload capacities
and a shared `torch.distributed.Store`. Connect peers with `connect_ranks`, then
activate membership with `set_active_ranks`.

Pass `dispatch` BF16 token rows shaped `[tokens, hidden]`, routing indices of type
`nixl.runtime.ep.topk_idx_t` shaped `[tokens, topk]`, and optional float32 routing
weights of the same shape. Run local experts on the received rows, then pass their
BF16 output and the dispatch handle to `combine`. When using explicit destruction,
call `destroy` collectively when the buffer is no longer needed.

Construction does not use the Python Store or wait for peers. Peer connection exchanges
metadata; activation selects the participating ranks. Communication and
destruction are collective over the committed active world.
When `async_with_compute_stream` is enabled, wait on the returned event
before consuming the result. Deterministic sorting is supported only in eager
execution. Disable `do_cpu_sync` for graph capture;
receive tensors then have maximum capacity, with actual counts in the
handle. See the `ElasticBuffer` method docstrings for expanded layouts and
FP8 scale-factor inputs.

## Topology and graph replay

Constructor `num_launch_sms` fixes the main dispatch/combine block count for
this buffer (default 24; even, from 2 through 64, matching across peers).
All blocks participate; the count does not change with membership or dispatch
inputs. The topology descriptor selects the active ranks and communication
mode, so captured graphs can reuse the fixed grid through shrink/expansion.
Topology changes are collective, between completed pairs, with updates and
replay ordered on the same stream or through explicit event dependencies.
Python handle fields and tensor shapes still describe capture-time metadata.

Dispatch/combine support a single active GPU. The calling rank must be active,
checked before launch and again inside the kernels for graph replay.
The required positive `num_experts_per_rank` fixes the physical routing
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
Slot `e` belongs to world rank `e // S`. Caller-assigned rank IDs do not encode
physical topology. IDs stay fixed across membership and
extent changes, including graph replay; holes are not packed away. `-1` means
no selection, and selections owned by inactive ranks are dropped. The caller
does not renumber IDs solely because an extent changes, but still owns logical
expert placement and routing weights. Internal source metadata continues to
use committed-extent rank encoding.

`connect_ranks` prepares peer views; `set_active_ranks` validates and activates
the requested membership and prepared views. `disconnect_ranks` excludes departing
peers from an active caller's membership and retires their connections in one update;
the remaining ranks must still form a product of domains and positions. An inactive
caller only retires connections, preserving its membership descriptor.
An empty disconnect does nothing.
Size the buffer for its maximum world at construction. Active ranks form a
product of active domains and positions within those domains. New-peer
connections may run while the committed graph replays. Activation and peer
retirement run outside graph capture. Use one CUDA device per process. Serialize control operations
on each buffer and select that device in any background setup thread. Before
activation or peer retirement, every old-world rank must finish
queued CUDA work, call `barrier` with `with_cpu_sync` enabled to drain the
transport, and complete a host rendezvous before any peer departs. Joining
ranks construct their buffers independently, then call `connect_ranks` with
their peers' stable world IDs. Mid-launch rank failure and transport recovery
are unsupported.

An idle joiner can warm up and capture dispatch/combine on its own GPU before
connecting peers, then reuse the graph after coordinated activation of the new
world. Construction precompiles the barrier without launching it.

`connect_ranks` exchanges agent names and region descriptors through the supplied Store,
publishing under `NIXL_EP/<world_rank>` and deleting it after the requested peers
have read it and native preparation finishes. Already-connected peers are ignored;
if no new peers remain, the call does nothing and needs no Store. New peers must request each
other in the same exchange. World IDs must be unique within the Store, and
all participants must finish an exchange before reusing its keys.
It also exchanges GPU discovery records and agrees on private domain/position mappings.
Mappings of connected ranks are preserved. Ranks that retire all peer connections
may receive new internal coordinates on reconnection. Prepared mappings become
live only on activation.

NIXL's metadata manager publishes and fetches transport metadata. Configure its
backend before buffer construction; for TCPStore, set `NIXL_TCPSTORE_ENDPOINT`
on every rank to the server's host and port separated by a colon. This can be the
server backing the Python Store and must remain available through buffer destruction.
Native publication, fetch and readiness checks release the GIL, allowing another Python thread to replay the
active graph. Publications remain available to future joiners until buffer
destruction. The manager starts its backend connection asynchronously during
construction without waiting for peers.

Invalid inputs and mapping refusals before peer retirement preserve active views.
Failures after peer retirement or view activation terminate the process; restart
it instead of attempting recovery.

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
- Required positive constructor argument `num_experts_per_rank` fixes physical routing slots
  per GPU. `num_experts` is removed from `dispatch`,
  `EPHandle`, and native dispatch/combine; slot
  IDs span the declared maximum world. The cached `dispatch` path accepting a `handle` is
  removed, along with `allow_hybrid_mode`, the buffer's `group` and alignment
  exports.
- Constructor `num_bytes` is removed. Allocation is calculated from the workload
  settings and maximum world dimensions; positive `num_max_tokens_per_rank` and
  `hidden` are required constructor arguments.
  `get_buffer_size_hint` uses explicit maxima for local memory planning, and `buffer.num_bytes`
  reports the communication allocation excluding workspace.
- Constructor and size-hint `num_topk` are required integers from 1 through 32 for buffer sizing;
  pass `32` to reserve maximum top-k capacity. Dispatch uses the width of `topk_idx`.
- `combine` takes no top-k weights and returns `(combined_x, event)`.
  Upstream used the optional weight return for routing-weight gradients in
  dispatch backward; this fork omits that path. It is independent of cached
  dispatch, which upstream used for combine backward.
- Dispatch and combine have no `num_sms` argument. Constructor `num_launch_sms`
  fixes the block count for the buffer's lifetime. There is no separate active
  SM count or automatic estimator. Communicator, QP and service-level arguments are removed.
- Enabling constructor and size-hint `use_fp8_dispatch` reserves smaller FP8
  dispatch capacity; peers must use matching capacity settings. These buffers
  accept FP8/NVFP4 and reject BF16 dispatch. Default BF16-capacity buffers
  accept all three wire formats, selected by dispatch input tensors.
- Active and connected world-rank queries replace static domain
  attributes/accessors; topology helpers are internal. `get_physical_domain_size` and the
  `sequential` argument of `barrier` are removed.
- `get_theoretical_num_sms` is removed. `EPHandle.deterministic_sort` uses the
  handle's channel list instead of accepting a separate one. The constant-true private
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
NVFP4, and enabling `use_fp8_dispatch` reserves the smaller one. Dispatch does
not quantize, apply scales, or transmit a global scale. The caller must
supply the consuming GEMM's global scale and any required scale swizzle;
using one shared GEMM scale requires inputs quantized with that same scale.
Column-major scale output is unsupported for NVFP4.

## Automatic combine optimization

Hybrid combine across multiple active domains automatically uses incremental
completion for destinations with at most 16 original input tokens, letting
the reduction epilogue wait for each token's contributions. Larger batches
use final channel completion; direct mode is
unchanged. Selection uses actual input counts, independently
for each rank, rather than buffer capacity. Traversal remains forward.

The threshold follows BF16 measurements on a 2×8 GB200 topology: incremental
completion materially improved batches 1 and 16, while larger batches had
small gains or regressions. Other configurations have not been performance
validated.

## Tests

Run `tests/elastic/test_elastic_ep.py` from `runtime/ep` after installing the
package. Launch one invocation per node with node-level `WORLD_SIZE` and `RANK`;
`--num-processes` selects the GPUs per node. Run with `--help` for workload,
format and timing options. The test requires the metadata backend configuration
above. Before launching, set `NIXL_TCPSTORE_ENDPOINT` to the host and port selected
by `MASTER_ADDR` and `MASTER_PORT`, which default to loopback and port 8361.

`tests/elastic/test_elastic_ep.py` accepts an inline JSON plan or a
plan file through `--plan`. `--benchmark`, `--kineto` and `--feed-pairs`
enable call, kernel and transition timing; `--save-output` selects a path for saving
per-rank results. Payload byte counts exclude transport overhead and are
not per-NIC bandwidth measurements. `--dtype` selects BF16, FP8 or NVFP4
dispatch for checked and timed pairs, with matching buffer capacity.
Payload values and scales are checked byte for byte; combine stays BF16.

Each phase checks eager execution and retained CUDA graphs with both original
and changed routing. Processes remain alive throughout the membership plan;
the active topology determines direct or hybrid execution.
`--reconnect-on-leave` retires departing ranks' peer connections into local
singleton buffers that sit out until rejoining; their captured graphs are retained.
Custom reconnect plans must form a product of domains and positions after each
reconnection. Ranks in a fully retired domain may receive positions incompatible
with those retained in another domain. The default plan avoids this by restoring
the full cohort between shrinks; custom plans can do the same or keep connections.
Unsupported plans fail the normal activation check; the test neither rewrites
them nor treats rejection as a passing result.

## Phase profiling

Set `NIXL_EP_ADAPTIVE_PROFILE` to 1 before importing NIXL EP to record diagnostic
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

Enable `--profile` in the elastic test to print diagnostic timestamps alongside
the eager and graph-replay correctness checks.

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
