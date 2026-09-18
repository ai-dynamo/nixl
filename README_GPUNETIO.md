<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native GPUNETIO Device API GTest harness

## Supported API and lifetime

Build capability and runtime selection are both opt-in. Set the GPUNETIO
backend parameter `native_device_api=true`; the default remains the legacy
host READ/WRITE backend. Select that backend explicitly in both memory-view
preparation calls. An application CUDA thread can then call:

```cpp
nixlGpuXferStatusH status{};
auto rc = nixlPut<nixl_gpu_level_t::THREAD>(
    {local_view, local_index, local_offset},
    {remote_view, remote_index, remote_offset}, bytes, 0, 0, &status);
// Only an accepted operation (rc == NIXL_IN_PROG) may be polled.
// Use a bounded polling loop in the caller.
rc = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(status);
```

The initial implementation supports one remote peer, one unretired PUT per
shared data QP, and ordinary registered VRAM on the engine's execution GPU.
All views of that peer share the same lane. Submission is not completion;
an occupied lane rejects another request without posting. A terminal status
is cached; status objects are single-owner and must not be reused in flight.
Zero-sized transfers, other execution levels, channels/flags, atomics and
host transfers/notifications in native mode are unsupported. No proxy fallback
is supplied. This is not a throughput or production receiver-ready protocol.

Prepare/release are cold operations: no capture, no concurrent application
kernels, and no outstanding work requiring progress in that CUDA context.
In particular, allocator synchronization must not run alongside a persistent
legacy kernel in the same context. Views, registrations, remote metadata,
payload buffers and the engine must outlive all submitted operations.
Finish application kernels and establish terminal NIC completion before
normal release. Native local views pin their registrations. A release on a
non-idle lane retains the backend view and MR pins until QP teardown; it is
not cancellation and does not permit freeing payload memory or running kernels
against released public handles. Errors latch the lane; recovery/error teardown
requires additional validation before production use. The test exits nonzero
without normal cleanup if accepted operations cannot be safely retired.

This branch changes the plugin ABI to version 2 and reserves the last four
bytes of the existing 64-byte device status for a backend tag. Rebuild core,
plugins and application CUDA extensions together; old-header compatibility
is not claimed.

This harness is a bounded correctness test for the public NIXL GPU Device API.
It runs one GTest process per endpoint, with one CUDA GPU per process. The
sender calls only `nixlAgent::prepMemView`, `nixlPut<THREAD>`, and
`nixlGpuGetXferStatus<THREAD>` for the data path. It does not call host
`postXfer`, notifications, a CPU proxy, or a GPUNETIO posting kernel.

The receiver flushes GPUDirect RDMA writes with the documented
`cuFlushGPUDirectRDMAWrites` API before copying and checking the destination.
If that API is unavailable or unsupported, the test fails; a plain CUDA
synchronization is not treated as a visibility proof.

The test covers 4 KiB, 64 KiB, and 1 MiB writes, nonzero descriptor-relative
offsets, payload and guard validation, cached terminal-status repolling, busy
lane rejection, invalid index/offset/flags/channel/null-status/zero/oversize
arguments, and 8,193 sequential 4 KiB writes to distinct destination slots.
The wrap case is about 34 MiB of payload and the process GPU allocation stays
below 128 MiB.

## Configuration

Both processes must use the same fresh coordination directory and run ID.
The following variables are read by the test:

| Variable | Meaning |
| --- | --- |
| `GPUNETIO_DEVICE_API_ROLE` | `sender` or `receiver` |
| `GPUNETIO_DEVICE_API_COORD_DIR` | unique shared file-exchange directory |
| `GPUNETIO_DEVICE_API_RUN_ID` | unique run prefix within that directory |
| `GPUNETIO_DEVICE_API_GPU` | one CUDA ordinal; defaults to `0` |
| `GPUNETIO_DEVICE_API_RDMA` | optional GPUNETIO `network_devices` value |
| `GPUNETIO_DEVICE_API_OOB_INTERFACE` | optional GPUNETIO `oob_interface` value |
| `GPUNETIO_DEVICE_API_GID_INDEX` | optional GPUNETIO `gid_index` value |
| `GPUNETIO_DEVICE_API_OOB_PORT` | optional per-endpoint GPUNETIO OOB port |
| `GPUNETIO_DEVICE_API_TARGET_IPV4` | sender’s receiver IPv4 for the control TCP channel |
| `GPUNETIO_DEVICE_API_CONTROL_PORT` | caller-selected unused control TCP port |

Metadata files are written through a temporary file, `fsync`, atomic `rename`,
and directory `fsync`. A tiny TCP control channel carries `M` (metadata
published), `S` (sender setup complete), and per-case `R`/`D`/`V` tokens for
receiver-ready, sender-data-done, and receiver-validated. The sender opens
metadata directly only after `M`; there is no filesystem existence polling.
The control channel has a 45-second socket/connect bound, and the device
completion poll has a 5-second deadline per request.

Setup is directional. The receiver creates its backend, registers its VRAM,
and publishes metadata plus the descriptor record, then waits only for the
sender’s TCP session-ready token. The sender imports the receiver metadata, makes the OOB
connection, and prepares both public memory views before issuing PUTs. The
receiver does not call `loadRemoteMD`, `makeConnection`, or `prepMemView`.

## Build and run

Run these commands from the checkout. Select the legacy SDK contract with
Meson's `gpunetio_device_api_profile=legacy-doca31` option; Meson emits
`NIXL_ENABLE_GPUNETIO_DEVICE_API=1` and
`NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31=1` for the CUDA target.

```bash
cd /path/to/nixl-checkout
meson setup build-native -Dbuildtype=debugoptimized -Denable_plugins=GPUNETIO \
  -Dbuild_examples=false -Dnixl_cuda_arch_list=90 \
  -Dgpunetio_device_api=enabled -Dgpunetio_device_api_profile=legacy-doca31
ninja -C build-native src/plugins/gpunetio/libplugin_GPUNETIO.so \
  src/utils/device/libnixl_device_allocator_cuda.so \
  test/gtest/native-device-api-test \
  test/gtest/unit/device_memview/device-memview-test
./build-native/test/gtest/unit/device_memview/device-memview-test
```

Use the exact installed CUDA/DOCA and NIXL plugin paths for the build. Do not
run this on a shared GPU unless the owner has explicitly authorized a bounded
normal-path correctness run; do not add profiling, fault injection, or
performance workloads to this harness.

Start the receiver first. `coord` must be a new shared directory visible to
both endpoint processes, and `run_id` must be unique:

```bash
: "${SHARED_TEST_DIR:?Set a directory shared by both endpoints}"
coord=$(mktemp -d "$SHARED_TEST_DIR/nixl-device-api.XXXXXX")
run_id=$(date +%s)-$$
export LD_LIBRARY_PATH="/opt/mellanox/doca/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export NIXL_PLUGIN_DIR="$PWD/build-native/src/plugins/gpunetio"
export LD_LIBRARY_PATH="$NIXL_PLUGIN_DIR:$LD_LIBRARY_PATH"
export CUDA_MODULE_LOADING=EAGER
export GPUNETIO_DEVICE_API_COORD_DIR="$coord"
export GPUNETIO_DEVICE_API_RUN_ID="$run_id"
export GPUNETIO_DEVICE_API_GPU=0
export GPUNETIO_DEVICE_API_RDMA=mlx5_0
export GPUNETIO_DEVICE_API_OOB_INTERFACE=eth0
export GPUNETIO_DEVICE_API_GID_INDEX=3
export GPUNETIO_DEVICE_API_OOB_PORT=6544
export GPUNETIO_DEVICE_API_CONTROL_PORT=6546

GPUNETIO_DEVICE_API_ROLE=receiver \
  ./build-native/test/gtest/native-device-api-test \
  --gtest_filter=GpunetioDeviceApiTest.*
```

Export the same coordination directory, run ID and control port in the sender
shell, with its own build/library/device configuration. Use the receiver's
reachable IPv4 address and an unused local OOB port:

```bash
GPUNETIO_DEVICE_API_ROLE=sender \
  GPUNETIO_DEVICE_API_TARGET_IPV4="$RECEIVER_IPV4" \
  GPUNETIO_DEVICE_API_OOB_PORT=6545 \
  ./build-native/test/gtest/native-device-api-test \
  --gtest_filter=GpunetioDeviceApiTest.*
```

The receiver and sender must use the endpoint-specific OOB interface, RDMA
device, GID, and port values valid in their own environment. The commands
above are an example configuration, not a hardware claim.

## Current boundary

This file intentionally contains the normal-path and bounded argument
harness only. Host wrapper/tag tests are provided separately. Foreign-engine and wrong-role tests,
concurrent submitter tests, stale-registration error teardown, and explicit
native-path tracing remain to be added alongside the corresponding public
implementation and test wiring. See the PR's validation table for tested
revisions; no serving, latency or bandwidth improvement is claimed.
