# NIXL EP API shell

The public exports, method signatures, defaults and argument annotations follow
the production API. Construction requires `world_rank`,
`max_num_scaleout_ranks`, `max_num_scaleup_ranks`, positive
`num_max_tokens_per_rank` and `hidden`, and keyword-only
`num_experts_per_rank`; it does not discover a group
or connect peers. The constructor has no `num_bytes` or `num_nixl_channels`
overrides. Public metadata getters and `publish` are absent; `connect_ranks`
takes a Store instead of manual metadata. Topology inspection and coordinate
helpers are private; `set_active_ranks` validates and activates membership in
one call, and `disconnect_ranks` excludes and retires peers in one call.
`get_buffer_size_hint` requires both maximum-world dimensions and queries size
locally without a process group or Store.
Method docstrings describe the
production contracts, not operations executed by this shell.

`num_experts_per_rank` is a required positive integer specifying the fixed
physical slot capacity per GPU; slots may remain unused. Choose it for the
largest per-GPU expert placement across all intended topologies, including
shrink, rather than just the initial placement.
Public expert IDs use `world_rank * num_experts_per_rank + local_expert_idx`,
where `0 <= local_expert_idx < num_experts_per_rank`. The ID range covers the
declared maximum world: `[0, max_num_scaleout_ranks * max_num_scaleup_ranks * num_experts_per_rank)`.
IDs stay fixed across membership and extent changes, including graph replay.
`-1` means no selection; production dispatch drops selections owned by inactive
ranks. The caller owns logical expert placement and routing weights, without
renumbering IDs solely because an extent changes. Internal source metadata
continues to use committed-extent rank encoding. The shell does not read or
route these IDs.

The keyword-only `num_launch_sms=24` selects the fixed main dispatch/combine
block count for the buffer's lifetime: an even integer from 2 through 64, exposed as a
read-only property and matching across peers. Dispatch, combine and captured
graph replays always use this count. Membership and direct/hybrid mode may
change between completed dispatch/combine pairs while reusing the same graph.
`dispatch` and `set_active_ranks` have no SM-count argument, and there is no
runtime SM setter or estimator. Handles do not record an SM count.

## Install and exercise the public API

Use an isolated Python environment with PyTorch installed, without the NIXL
core or backend wheels. The standalone distribution provides the namespace
`nixl.runtime.ep` and does not build the rest of this repository:

```bash
python -m pip install ./runtime/ep
```

An installed NIXL core package owns the same `nixl` namespace and may load its
native backend first. Keep the shell in its own environment. For source-only
review, run the tests below without installing the package.

```python
import torch
from nixl.runtime.ep import ElasticBuffer, topk_idx_t

buffer = ElasticBuffer(
    world_rank=0, max_num_scaleout_ranks=1, max_num_scaleup_ranks=1,
    num_experts_per_rank=8, num_max_tokens_per_rank=16,
    hidden=512, num_topk=2, explicitly_destroy=True,
)
x = torch.empty((16, 512), dtype=torch.bfloat16, device="meta")
topk = torch.empty((16, 2), dtype=topk_idx_t, device="meta")
received, indices, weights, handle, event = buffer.dispatch(x, topk)
combined, event = buffer.combine(received, handle)
assert combined.shape == (16, 512)
assert combined.device.type == "meta"
buffer.destroy()
```

## What the shell models

- Configuration, fixed expert slots per rank, connected peers, active topology,
  direct/hybrid selection, the fixed SM count and
  explicit destruction are local Python state. No other rank participates.
  Construction starts with only `world_rank` active and no connected peers.
  Dispatch/combine work in this initial single-rank state. To add peers,
  connect them explicitly, then activate them with `set_active_ranks`.
- Dispatch returns the five-item API result; combine returns its two-item
  result. Receive tensors always have zero rows. Combine's placeholder has the
  original input token count and the supplied combine input's hidden width.
- Tensors created by the shell use the `meta` device and have no values or data
  storage. With `do_handle_copy=False`, `handle.topk_idx` retains the original
  input tensor, including its device and storage.
  BF16 and quantized input forms, optional weights and expanded
  return structure can be exercised; routing, reductions, alignment, layout
  strides and numerical correctness are not simulated. Input tensors and
  cumulative counters are never read or modified.
- Events are inert. Waiting performs no synchronization. Registered hooks are
  discarded when a wait is requested, without invoking caller code. The
  communication stream is an opaque placeholder, not a usable CUDA stream.
  Explicit event release also clears `extra_tensors`.
- `num_bytes` is `0`, `get_buffer_size_hint()` returns `0`,
  and `read_profile()` returns an empty mapping. These are placeholders, not
  sizing or performance results.
- Peer connections and retirement are local state. `connect_ranks`
  ignores already-connected peers and requires a Store only for new peers,
  passed directly or at construction, but never accesses it.
  On an active local rank, `disconnect_ranks` also removes requested peers from
  membership; survivors must form a nonempty product of domains and positions.
  Direct/hybrid mode follows the surviving world and the fixed SM count stays
  unchanged. On an inactive local rank, retirement preserves its active-world
  descriptor, including when all peers in that world are retired. The entire
  request and surviving topology are validated before either state changes;
  invalid requests leave both membership and connections unchanged, and an
  empty request is a no-op. The caller must drain the old world and complete a
  host rendezvous before retiring peers; the call does not coordinate ranks.
  A disconnected peer can reconnect after `disconnect_ranks` returns. No data is exchanged.
  `init_jit()` is a no-op, and `topk_idx_t` is fixed to `torch.int64`.

The shell does not validate execution-dependent constraints, CUDA graph or
stream behavior, memory ownership, timeouts, transport completion or GPU
correctness. The existing core NIXL and EP example sources remain in this repository;
none is loaded by the standalone shell.

## Test

```bash
PYTHONPATH=runtime/ep python -m unittest discover -s runtime/ep/tests/api -v
```

Tests cover API parity and the shell's local state/placeholder behavior. They
do not require GPUs, UCX, NIXL native libraries, a process group or a Store.
