# NIXL EPv2 API shell

This branch exposes the API from `polished/dev-branch` at `6b11b2c` for interface
review and CPU-only integration tests. It contains no EPv2 kernels, native
extension, JIT compiler, transport implementation or hardware discovery.
The production implementation remains on `polished/dev-branch`.

The public exports, method signatures, defaults and argument annotations are
preserved. Method docstrings retain the production contracts for review; they
are not claims that this shell executes those operations.

## Install and exercise the public API

Use an isolated Python environment with PyTorch installed, without the NIXL
core or backend wheels. The standalone distribution provides the namespace
`nixl.runtime.ep` and does not build the rest of this repository:

```bash
python -m pip install ./runtime/ep
```

An installed NIXL core package owns the same `nixl` namespace and may load its
native backend first. Keep the shell in its own environment. For source-only
review, `PYTHONPATH=runtime/ep` also exposes the same API as `nixl_ep_v2`.

```python
import torch
from nixl.runtime.ep import ElasticBuffer, topk_idx_t

buffer = ElasticBuffer(
    rank_idx=0, num_ranks=2, num_group_scaleup_ranks=2,
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

- Configuration, fixed expert slots per rank, connected peers, staged versus
  committed topology, direct/hybrid selection, manual/automatic SM policy and
  explicit destruction are local Python state. No other rank participates.
- With no declared domain size, the cohort is treated as one domain. Group
  objects are queried only for their local `rank()` and `size()` values.
- Dispatch returns the five-item API result; combine returns its two-item
  result. Receive tensors always have zero rows. Combine's placeholder has the
  original input token count and the supplied combine input's hidden width.
- Tensor results and handle metadata use the `meta` device. They have no values
  or data storage. BF16 and quantized input forms, optional weights and expanded
  return structure can be exercised; routing, reductions, alignment, layout
  strides and numerical correctness are not simulated. Input tensors and
  cumulative counters are never read or modified.
- Events are inert. Waiting performs no synchronization. Registered hooks are
  discarded when a wait is requested, without invoking caller code. The
  communication stream is an opaque placeholder, not a usable CUDA stream.
- `get_buffer_size_hint()` returns `0`, `get_theoretical_num_sms()` returns `2`,
  and `read_profile()` returns an empty mapping. These are placeholders, not
  sizing, tuning or performance results. The shell's launch limit is `24`.
- `publish()` is a no-op; peer connections and pending disconnects are local
  state. A disconnected peer can reconnect after commit. No Store is touched
  or data exchanged. Metadata contains no usable address. `init_jit()` is a
  no-op, and `topk_idx_t` is fixed to `torch.int64`.

The shell does not validate execution-dependent constraints, CUDA graph or
stream behavior, memory ownership, timeouts, transport completion or GPU
correctness. The existing core NIXL and EPv1 sources remain in this repository;
none is loaded by the standalone shell.

## Test

```bash
PYTHONPATH=runtime/ep python -m unittest discover -s runtime/ep/tests/api -v
```

Tests cover API parity and the shell's local state/placeholder behavior. They
do not require GPUs, UCX, NIXL native libraries, a process group or a Store.
