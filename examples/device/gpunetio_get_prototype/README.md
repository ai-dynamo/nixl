# Two-endpoint GPU GET owner/consumer reproducer recipe

**Prototype recipe, not currently runnable against public main.** This isolates the NIXL transfer direction and lifecycle from model layouts, FP8, Engram kernels, and serving. Use two independent endpoint processes and only caller-supplied device/network values.

## Roles and data contract

- **Owner:** allocate a bounded registered VRAM source buffer, fill deterministic bytes plus leading/trailing guards, create the GPUNETIO agent, register the source, export metadata, and keep the buffer/MR/agent alive until the consumer reports validation.
- **Consumer:** allocate/register a local VRAM destination and any SDK-required local scratch, import owner metadata, connect, prepare the remote-source and local-destination views, then submit one GET from a single CUDA thread.
- The caller polls only accepted requests until terminal success/error. It does not reuse the status or submit a second operation while the single-flight lane is occupied. After terminal success and the SDK/NIXL-documented local visibility step, copy the destination to host and verify every byte and both guards. On timeout/error, both processes retain registrations and payload storage until safe QP teardown; timeout is not cancellation.
- Exercise 4 KiB, 64 KiB, and 1 MiB payloads, nonzero descriptor-relative offsets, a busy/repost attempt, rejected invalid ranges, and terminal status repolling. This is correctness only; do not report performance or serving benefits.

The legacy reference signature currently requires an explicit local DUMP scratch element:

```cpp
auto rc = nixl::gpu::experimental::get(
    {local_dst_view, 0, dst_offset}, {remote_src_view, 0, src_offset},
    bytes, {local_scratch_view, 0, scratch_offset}, &status);
```

That symbol and signature exist only in the retained legacy reference file here. They are **not** declared by current public `nixl_device.cuh`; do not compile an application against this snippet as though it were a supported API.

## Generic coordination inputs

The eventual two-process test should reuse the existing `GpunetioDeviceApiTest` coordination inputs, never hard-code addresses or ports. In this GET case the receiver is the owner and the sender is the consumer:

| Input | Variable | Meaning |
|---|---|---|
| Endpoint role | `GPUNETIO_DEVICE_API_ROLE` | `receiver` (owner) or `sender` (consumer) |
| Shared coordination directory | `GPUNETIO_DEVICE_API_COORD_DIR` | fresh directory visible to both processes |
| Run identifier | `GPUNETIO_DEVICE_API_RUN_ID` | unique per owner/consumer pair |
| Device ordinal | `GPUNETIO_DEVICE_API_GPU` | caller-selected GPU on each endpoint |
| RDMA NIC / OOB interface / GID | `GPUNETIO_DEVICE_API_RDMA`, `GPUNETIO_DEVICE_API_OOB_INTERFACE`, `GPUNETIO_DEVICE_API_GID_INDEX` | caller-selected fabric settings |
| OOB and control ports | `GPUNETIO_DEVICE_API_OOB_PORT`, `GPUNETIO_DEVICE_API_CONTROL_PORT` | caller-selected unused ports |
| Owner address | `GPUNETIO_DEVICE_API_TARGET_IPV4` | consumer reaches the owner control listener |

The owner publishes immutable metadata before the consumer connects. A small control channel coordinates `metadata-ready`, `consumer-ready`, `request-done`, and `payload-validated`; it does not establish data visibility. The owner frees resources only after `payload-validated` and after all accepted work is retired. The current #2234 test harness implements this two-process coordination for PUT; a GET test must reverse the payload direction and verify the owner-held source remains alive until the consumer retires the GET.

## Current blocker

The current public branch has no GPU GET call/dispatch/GPUNETIO compat implementation. It also only prepares native GPUNETIO memory views for `VRAM_SEG`; the validated Host-owner/`DRAM_SEG` case needs a separate view-capability patch. The legacy helper requires DOCA 3.1 `DUMP_AUTO` and private backend view/QP internals, while the current `#2234` branch requires the public DOCA 3.5+ profile and does not provide that helper. Therefore this recipe describes the intended owner/consumer test, not a passing public test target.
