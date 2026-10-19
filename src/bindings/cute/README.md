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

From the repository root, set `UCX_PREFIX` to that UCX installation:

```sh
python src/bindings/cute/build.py --ucx "$UCX_PREFIX" --arch sm_100
export PYTHONPATH="$PWD/src/bindings/cute:${PYTHONPATH:-}"
torchrun --standalone --nproc-per-node=2 examples/python/cute/compute_put.py
```

Select `--level thread|warp|block`, `--defer` and `--elements N` as needed.
Gloo exchanges host metadata and coordinates validation/buffer reuse; NIXL moves
the GPU payload. The example checks three generations and guard regions.
Its host synchronization is for correctness, not a performance benchmark.

## Device contract

`put`, `atomic_add` and `get_xfer_status` forward the
[native API](../../api/device/gpu/nixl_device.cuh), including descriptor indices,
byte offsets, channels, flags and status.

- Use a compile-time THREAD/WARP/BLOCK level; all participants enter uniformly.
- Each participating thread owns a zeroed, 64-byte request aligned to 16 bytes.
  Keep requests, buffers and views alive until completion.
- Poll only after `IN_PROGRESS`, at the same cooperation level; stop on success
  or error. `get_xfer_status` performs one poll, not a completion loop.
- `request=0` forwards a native null request; it does not select blocking mode.
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
ordering. Inter-node RDMA, performance parity and peer churn remain unqualified.
