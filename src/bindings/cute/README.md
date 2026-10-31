# NIXL device API for CuTe

Call NIXL from a CuTe compute kernel, for example to transfer produced KV-cache
data. Host registration, connections and memory views remain ordinary NIXL calls.
This is a source-only prototype; it is not installed by the NIXL package.

- [nixl_cute.py](nixl_cute.py): Python device operations using `@cute.extern`.
- [device.cu](device.cu): thin calls to the native NIXL device API.
- [compute_put.py](../../../examples/python/cute/compute_put.py): separate two-GPU
  correctness example, including computation, completion and notification.

## Build and run

Requires an installed NIXL with UCX device support and UCX development headers
(`--enable-devel-headers`). Use the same NIXL/UCX sources for the host library and
bitcode. The tested toolchain is LLVM 20, CUDA 13.3 and CuTe DSL 4.5.1; the example
also uses PyTorch and cuda-python.

From the repository root, add these options to your NIXL Meson configuration
(`CUDA_HOME` is the toolkit root; `UCX_PREFIX` is that UCX installation):

```sh
meson setup build -Dbuild_cute=true -Dnixl_cuda_arch_list=100 \
  -Ducx_path="$UCX_PREFIX" -Dcudapath_inc="$CUDA_HOME/include" \
  -Dcudapath_lib="$CUDA_HOME/lib64"
meson compile -C build cute-device-bitcode
export PYTHONPATH="$PWD/build/src/bindings/cute:${PYTHONPATH:-}"
timeout --signal=TERM --kill-after=10s 120s \
  torchrun --standalone --nproc-per-node=2 examples/python/cute/compute_put.py
```

Meson uses `build.py` for LLVM/libNVVM normalization and tracks included headers.
Set `-Dcute_llvm_path=...` if LLVM 20 is not in `/usr/lib/llvm-20/bin`.
For an existing matching NIXL installation, `python src/bindings/cute/build.py
--ucx "$UCX_PREFIX" --arch sm_100` still works; use `src/bindings/cute` on
`PYTHONPATH` in that case.

Select `--level thread|warp|block`, `--defer` and `--elements N` as needed.
The example's `--defer` then wait protocol is validated only for CUDA IPC;
GDA may need a later nondeferred submission to flush deferred work.
Gloo exchanges host metadata and coordinates validation/buffer reuse; NIXL moves
the GPU payload. The example checks three generations and guard regions.
Its host synchronization is for correctness, not a performance benchmark.
GNU `timeout` bounds the job; Gloo's timeout does not stop GPU waits. On timeout,
abort the run rather than attempting buffer reuse or recovery.

## Device contract

`put`, `atomic_add` and `get_xfer_status` forward the
[native API](../../api/device/gpu/nixl_device.cuh), including descriptor indices,
byte offsets, channels, flags and status.

- Use a compile-time THREAD/WARP/BLOCK level; all participants enter uniformly.
- Cooperating threads share one zeroed, 64-byte request, aligned to 16 bytes;
  the native dynamic dispatcher tags it from the group leader. Independent
  operations need separate requests. Keep requests, buffers and views alive
  until completion.
- Poll only after `IN_PROGRESS`, at the same cooperation level; stop on success
  or error. `get_xfer_status` performs one poll, not a completion loop.
- `request=0` forwards a native null request; it does not select blocking mode.
  Polling zero is invalid and returns `NIXL_ERR_INVALID_PARAM`.
- No wrapper wait, fence, allocation or launch is added. Native submission may
  do inline work; `DEFER` permits deferral but does not guarantee overlap.
- Producer publication, receiver acquire and safe buffer reuse belong to the
  caller. See the example for one explicit protocol.

## Tests

With GTest/GMock installed and Meson configured using
`-Dbuild_tests=true -Dbuildtype=debugoptimized`:

```sh
meson compile -C build unit
meson test -C build unit --test-args='--gtest_filter=CuteDeviceBinding.*'
```

These CPU tests check shim forwarding and progress policy, not GPU transport or
ordering. Two-GPU CUDA IPC passes at THREAD/WARP/BLOCK; a GDA-enabled build and
forced-GDA THREAD/WARP transfers also pass on the tested stack.

GDA BLOCK hangs with UCX `589be408`: its
[reservation helper](https://github.com/openucx/ucx/blob/589be40811c7b53591cbbfe8a3739b5984987cdc/src/uct/ib/mlx5/gdaki/gdaki.cuh#L148)
does not broadcast the leader's queue index. Use THREAD/WARP for GDA until this
dependency is fixed. Inter-node RDMA, performance parity and peer churn remain
unqualified.
