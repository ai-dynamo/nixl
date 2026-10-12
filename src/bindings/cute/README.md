# NIXL device API for CuTe — review prototype

Use NIXL from an existing CuTe compute kernel. Host registration, peer metadata,
connections and memory-view preparation remain ordinary NIXL Python calls.
The motivating use is **KV-cache transfer initiated from compute**, not a new
communication runtime or an implementation of NIXL EP v2.

## Review map

| File | Responsibility |
| --- | --- |
| [nixl_cute.py](nixl_cute.py) | Three device operations and native scalar arguments |
| [device.cu](device.cu) | Source-matched, inlinable calls to native NIXL templates |
| [build.py](build.py), [compat/](compat/) | LLVM bitcode build; Clang/CUDA/UCX compatibility |
| [compute_put.py](../../../examples/python/cute/compute_put.py) | Application-owned compute, completion loop, notification and bootstrap |
| [device_binding.cpp](../../../test/gtest/unit/cute/device_binding.cpp) | CPU forwarding tests, compiling the actual shim |
| [comparison.html](comparison.html) | Native NIXL vs NCCL GIN vs this binding: coverage and semantics |

There is no stable ABI, extra C++ runtime, CUTLASS fork, `World` abstraction,
MoE implementation, allocator or implicit completion policy. This directory is
source-only: package/wheel integration is a follow-up, not part of this review.

## Device contract

`put` maps to `nixlPut`; `atomic_add` maps to `nixlAtomicAdd`;
`get_xfer_status` maps to one call to `nixlGpuGetXferStatus`.
See the [native contract](../../api/device/gpu/nixl_device.cuh).

- Descriptor indices, byte offsets, channel, flags and status are forwarded.
  `THREAD`, `WARP` and `BLOCK` select the native cooperation template. Use a
  compile-time level; all participants must enter cooperative calls uniformly.
- Each participating thread supplies its own zero-initialized, 64-byte request,
  aligned to 16 bytes. Reuse it only after completion, with the same cooperation
  level. Buffers and views must remain valid while operations are outstanding.
- The binding adds **no completion wait**; native submission may do inline work.
  On `IN_PROGRESS`, progress the request until success/error; do not poll a
  terminal request. The pinned UCX backend reports accepted submissions as
  `IN_PROGRESS`, even when UCX initially returns OK.
- `request=0` passes the native null pointer; it does **not** select blocking
  mode. There is no pollable request in that case. The application must establish
  completion/lifetime by its native protocol; the example always tracks requests.
- `DEFER` forwards the native flag; it permits deferral rather than promising
  overlap. No binding-side loop, fence, allocation or kernel launch is inserted.
- `atomic_add` operates on a remote uint64 counter; native ordering semantics
  apply. Publication of compute stores, receiver acquire, counter ownership and
  buffer reuse remain the application's responsibility.

Conceptually, inside the caller's kernel:

```python
# Compute into registered memory, publish stores and synchronize producers.
status = device.put(local_view, remote_view, nbytes, request,
                    local_index=1, remote_index=1, level=device.BLOCK)
# Independent work may run here if it does not reuse in-flight storage.
while status == device.IN_PROGRESS:
    status = device.get_xfer_status(request, level=device.BLOCK)
# Check errors before consuming/reusing storage or publishing readiness.
```

The binding uses `@cute.extern` to link bitcode into the caller's kernel.
`@cute.kernel` and the host `@cute.jit` launcher in the example belong to that
application; they are not required wrappers around NIXL host setup. No
`@cute.native_struct` is needed for these opaque handles and scalar arguments.

## Build and run

Build/install NIXL with its UCX device backend first. Use the **same NIXL and UCX
sources** to build this bitcode and the host library. The initial toolchain is
LLVM 20, CUDA 13 and `nvidia-cutlass-dsl==4.5.1`; this is not a compatibility promise.
UCX must install its device development headers (`--enable-devel-headers`).

From the repository root, with `UCX_PREFIX` pointing to that installation:

```sh
python src/bindings/cute/build.py --ucx "$UCX_PREFIX" --arch sm_100
export PYTHONPATH="$PWD/src/bindings/cute:${PYTHONPATH:-}"
torchrun --standalone --nproc-per-node=2 examples/python/cute/compute_put.py
# Also exercise --level thread|warp|block, with and without --defer.
```

The example computes a changing payload, submits PUT in that same kernel, then
notifies a GPU receiver. It checks three generations, nonzero descriptor indices,
offsets, channel 2, counter increment 7 and guard regions. Its GPU completion loop
and CPU validation/reuse barriers are deliberately visible. Gloo is only the
two-rank test bootstrap, not a requirement of NIXL's peer model. This is a
correctness example, **not a speed-of-light benchmark or a KV-cache codec**.

With GTest/GMock installed and a Meson build configured using
`-Dbuild_tests=true -Dbuildtype=debugoptimized` (upstream excludes tests in release):

```sh
meson compile -C build cute_device_binding
meson test -C build cute_device_binding --print-errorlogs
```

The CPU tests check argument/status forwarding, null requests, one-step progress
and rejection of unsupported cooperation levels. They do not prove GPU ordering,
transport progress or performance. GPU qualification is recorded in the PR.

## Review decisions and next qualification

1. Confirm API names, request ownership and the native completion/ordering contract.
2. Decide whether `nixlGetPtr` belongs in this first binding. It is intentionally
   absent: it needs a nullable mapped-pointer contract, not a fake remote GET.
3. Agree on installation/build ownership. `GRID` cooperation is also out of scope.
4. Qualify inter-node GPU RDMA, multi-request overlap, peer/view replacement after
   draining old work, and native-CUDA-vs-CuTe overhead before making perf claims.

Elasticity comes from native per-peer metadata/connections/views. This binding
does not impose communicator-wide membership or bake peers into compiled kernels.
It also does not implement discovery, live replacement of in-flight views, failure
recovery or a peer-churn test. Application coordination remains necessary.
