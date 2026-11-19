# NIXL device API for CuTe

Call NIXL from a CuTe compute kernel, for example to transfer produced KV-cache
data. Host registration, connections and memory views remain ordinary NIXL calls.
The optional `nixl.cute` module links device bitcode from the selected CUDA backend.
Ordinary `import nixl` does not import CuTe.

- [cute.py](../python/nixl-meta/nixl/cute.py): Python device operations using `@cute.extern`.
- [device.cu](device.cu): thin calls to the native NIXL device API.
- [compute_put.py](../../../examples/python/cute/compute_put.py): separate two-GPU
  correctness example, including computation, completion and notification.

## Install and run

Use a NIXL wheel built with `-Dbuild_cute=true`, then install its optional
dependencies with `pip install 'nixl[cute]'`. Import the operations explicitly:

```python
from nixl import cute as device
```

Only CuTe-enabled builds include `nixl.cute` and the `[cute]` extra. The extra
installs the tested CuTe DSL 4.5.1 with CUDA 13 support.

The wheel example uses [upstream PyTorch](https://pytorch.org/get-started/previous-versions/)
2.11 (CUDA 13). Keep UCX compatible across the process: NGC 26.06 PyTorch
preloads UCX 1.21, which conflicts with this UCX 1.23 build.

## Build from source

Requires UCX device support and UCX development headers
(`--enable-devel-headers`). Use the same NIXL/UCX sources for the host library and
bitcode. The example uses CUDA 13.3, CuTe DSL 4.5.1, PyTorch and cuda-python.

In an activated virtual environment, from the repository root
(`CUDA_HOME` is the toolkit root; `UCX_PREFIX` is that UCX installation):

```sh
meson setup build --prefix="$VIRTUAL_ENV" --libdir=lib \
  -Dbuild_cute=true \
  -Ducx_path="$UCX_PREFIX" -Dcudapath_inc="$CUDA_HOME/include" \
  -Dcudapath_lib="$CUDA_HOME/lib64"
meson compile -C build
meson install -C build
pip install --no-deps build/src/bindings/python/nixl-meta/nixl-*.whl
pip install 'nvidia-cutlass-dsl[cu13]==4.5.1'
timeout --signal=TERM --kill-after=10s 120s \
  torchrun --standalone --nproc-per-node=2 examples/python/cute/compute_put.py
```

Use a Python environment that can import the installed CUDA backend; `uv` must
be available at Meson setup to build the meta wheel. Meson builds one
`libnixl_device_sm_<arch>.bc` per target in `nixl_cuda_arch_list` and installs
each file in `libdir` and the Python backend package. The wheel includes only the
package copy. Use the default list or set `-Dnixl_cuda_arch_list=90,100` for a
smaller build.

CuTe selects bitcode for the kernel's compile target, including an explicit
`--gpu-arch`. A kernel targeting `sm_90a` uses the baseline `sm_90` bitcode.
If the wheel lacks that target, compilation reports which targets it contains.

Meson uses `build.py` to prepare bitcode for libNVVM and tracks included headers.
The default LLVM 22 tool directory is `/usr/lib/llvm-22/bin`. Use
`-Dcute_llvm_path=...` for another installation that supports the requested targets.
For a backend wheel, pass the same setup options to `python -m build --wheel`
using `-Csetup-args=-D...`; the bitcode is included in its installation manifest.

Select `--level thread|warp|block`, `--defer` and `--elements N` as needed.
With `--defer`, the example polls once. If work is still pending, it calls
`device.flush(remote, channel=..., level=...)` before continuing to poll.
Flush waits on the GPU for prior work on that channel. It requires UCX direct
mode built with `ucp_device_flush`. Older UCX builds and proxy mode return
`NIXL_ERR_NOT_SUPPORTED`. If the request is still pending and flush is
unsupported, the example reports the error. Run without `--defer` on that stack.
Polling alone cannot ring the doorbell for deferred GDA work.
Gloo exchanges host metadata and coordinates validation/buffer reuse; NIXL moves
the GPU payload. The example checks three generations and guard regions.
Its host synchronization is for correctness, not a performance benchmark.
GNU `timeout` bounds the job; Gloo's timeout does not stop GPU waits. On timeout,
abort the run rather than attempting buffer reuse or recovery.

## Device contract

`put`, `atomic_add`, `get_xfer_status` and `flush` forward the
[native API](../../api/device/gpu/nixl_device.cuh), including descriptor indices,
byte offsets, channels, flags and status.

- The supported cooperation levels are `THREAD`, `WARP` and `BLOCK`. All
  participants enter each cooperative call together.
- Cooperating threads share one 64-byte request aligned to 16 bytes. The example
  zeroes it once. Each accepted submission initializes its request state, so the
  storage can be reused after completion without clearing it. Overlapping
  operations need separate requests. Keep requests, buffers and views alive
  until completion.
- Poll only after `IN_PROGRESS`, at the same cooperation level; stop on success
  or error. `get_xfer_status` performs one poll, not a completion loop.
- `request=0` forwards a native null request; it does not select blocking mode.
  Polling zero is invalid and returns `NIXL_ERR_INVALID_PARAM`.
- Submission adds no wrapper wait, fence, allocation or launch. Native
  submission may do inline work even with `DEFER`.
- Producer publication, receiver acquire and safe buffer reuse belong to the
  caller. See the example for one explicit protocol.

## Tests

With GTest/GMock installed and Meson configured using
`-Dbuild_tests=true -Dbuildtype=debugoptimized`:

```sh
meson compile -C build unit
meson test -C build unit --test-args='--gtest_filter=CuteDeviceBinding.*'
```

`pytest test/python/test_cute_packaging.py` checks lazy imports, CUDA backend
selection and missing optional artifacts without requiring a GPU.

The CPU shim tests check forwarding and progress policy, not GPU transport or
ordering. Two-GPU CUDA IPC passes at THREAD/WARP/BLOCK; a GDA-enabled build and
forced-GDA THREAD/WARP transfers also pass on the tested stack.

GDA BLOCK hangs with UCX `589be408`: its
[reservation helper](https://github.com/openucx/ucx/blob/589be40811c7b53591cbbfe8a3739b5984987cdc/src/uct/ib/mlx5/gdaki/gdaki.cuh#L148)
does not broadcast the leader's queue index. Use THREAD/WARP for GDA until this
dependency is fixed. Inter-node RDMA, performance parity and peer churn remain
unqualified.
