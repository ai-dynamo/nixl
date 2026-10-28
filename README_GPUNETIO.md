<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Opt-in native GPUNETIO Device API

This increment adds GPU-initiated PUT/completion to current NIXL main while
retaining plugin ABI **3**, main's public device-mode dispatch, and the merged
host GPUNETIO progress implementation. Build with DOCA **3.5 or newer** and
`-Dgpunetio_device_api=enabled`, then set the backend parameter
`native_device_api=true`. Default-off host READ/WRITE behavior is retained.
The legacy DOCA 3.1 profile is not supported by this current-main build.

Prepare local and remote views through `nixlAgent::prepMemView`, selecting the
native GPUNETIO backend explicitly through the backend hints. The backend
returns main's public `memViewWrapper`, allocated using `nixl::deviceOps` and
`deviceMemViewAllocate`. Native execution uses
`nixl::gpu::exec_mode_t::GPUNETIO_DIRECT` and the typed `xferStatusH::mode` field;
it does not write a uint32 tag at an assumed byte offset. Main's UCX/proxy
modes, fixed/dynamic mode templates, and 60-byte request-payload layout remain.

```cpp
nixlGpuXferStatusH status{};
auto rc = nixlPut<nixl_gpu_level_t::THREAD>(
    {local_view, local_index, local_offset},
    {remote_view, remote_index, remote_offset}, bytes, 0, 0, &status);
// Poll only an accepted request (NIXL_IN_PROG), using a bounded caller loop.
rc = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(status);
```

The namespaced `nixl::gpu::put/getXferStatus` interface is also supported. A fixed
`exec_mode_t::GPUNETIO_DIRECT` template mode must use matching native views and
be polled with that same fixed mode, following main's device API contract.

The initial native path supports one peer, one unretired PUT on its shared data
lane, and ordinary registered VRAM on the execution GPU. It requires one data
QP. It does not add owner leases, batched commits, warp preparation, atomics,
channels, flags, READ, notification support, or a receiver-ready protocol.
Unsupported execution levels/arguments reject without posting. Native mode
rejects host transfer operations. Initialization, registration, metadata and
connection setup remain host-managed.

## Lifetime and errors

Status objects belong to one submitter and remain live until terminal
completion; do not reuse an active status. All prepared native views share the
engine lane, and an occupied lane rejects another request. Local views retain
MR ownership. Finish caller kernels and accepted NIC work before releasing
views, registrations, payload storage or the engine. Prepare/release are cold
operations and must not race application kernels or outstanding device work;
allocation synchronization must not coexist with an unrelated persistent
kernel in that context.

A busy/failed view release retains its storage and MR pins until QP teardown;
it is not cancellation or permission to free the payload. CQ errors latch the
native lane. A caller must stop and preserve resources after unresolved work;
error recovery is not supplied by this increment. All NIXL core/plugin/device
libraries and application CUDA extensions must be built from matching ABI-3
headers. Use main's `libdevice_utils.so` / `libnixl_device_ops_cuda.so`, not the
obsolete device-allocator DSOs from historical runs.

## Build and existing correctness harness

The native test is opt-in and needs CUDA, DOCA 3.5+ and GTest. Select the correct
SDK/library paths through the existing build environment before setup:

```bash
meson setup build-native -Dbuildtype=debugoptimized -Dwerror=true \
  -Denable_plugins=GPUNETIO -Dbuild_tests=true -Dbuild_examples=false \
  -Dnixl_cuda_arch_list=90 -Dgpunetio_device_api=enabled
ninja -C build-native src/core/libnixl.so \
  src/plugins/gpunetio/libplugin_GPUNETIO.so \
  src/utils/device/libnixl_device_ops_cuda.so \
  test/gtest/native-device-api-test
```

The test target selects `NIXL_ENABLE_GPUNETIO_DEVICE_API=1` and
`NIXL_GPUNETIO_DEVICE_API_PUBLIC=1`. Application CUDA translation units must
also define both macros and include/link the matching DOCA SDK; enabling the
plugin does not change an external application's compiler flags.
Configure no native capability for the
ordinary default-off build. No additional plugin ABI version bump is required.
The CUDA device-ops DSO is loaded lazily as a sibling of `libdevice_utils.so`;
keep both in the same build/install runtime directory.

The existing two-process `GpunetioDeviceApiTest` covers 4 KiB, 64 KiB, and
1 MiB writes; descriptor-relative offsets, full payload/guards, cached terminal
status, active lane/status rejection, invalid indexes/offsets/channels/flags/
null status/zero/oversize arguments, and 8,193 sequential 4 KiB writes to distinct
destination slots. Only its profile guard, typed-status handling, copyright
header and clock-rate attribute query are adapted to current main/CUDA 13.
No unrelated test suite is added. Receiver visibility is established using
`cuFlushGPUDirectRDMAWrites` before D2H verification.

Use a fresh shared coordination directory and run ID. Configure both processes
with matching runtime/SDK paths; select physical GPUs/fabric for the test host.
Start one receiver and one sender concurrently (the receiver waits for sender):

```bash
export NIXL_PLUGIN_DIR="$PWD/build-native/src/plugins/gpunetio"
export LD_LIBRARY_PATH="$PWD/build-native/src/core:$PWD/build-native/src/infra:$PWD/build-native/src/utils/common:$PWD/build-native/src/utils/serdes:$PWD/build-native/src/utils/stream:$PWD/build-native/src/utils/device${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GPUNETIO_DEVICE_API_COORD_DIR=/path/to/fresh/shared-directory
export GPUNETIO_DEVICE_API_RUN_ID=unique-run
export GPUNETIO_DEVICE_API_RDMA=your_rdma_device
export GPUNETIO_DEVICE_API_OOB_INTERFACE=your_oob_interface
export GPUNETIO_DEVICE_API_GID_INDEX=your_gid
export GPUNETIO_DEVICE_API_CONTROL_PORT=6546
# Receiver, in its own process:
GPUNETIO_DEVICE_API_ROLE=receiver GPUNETIO_DEVICE_API_GPU=1 \
  GPUNETIO_DEVICE_API_OOB_PORT=6544 \
  ./build-native/test/gtest/native-device-api-test --gtest_filter=GpunetioDeviceApiTest.*
# Sender, launched concurrently with the same coordination variables:
GPUNETIO_DEVICE_API_ROLE=sender GPUNETIO_DEVICE_API_GPU=0 \
  GPUNETIO_DEVICE_API_TARGET_IPV4=receiver_ipv4 GPUNETIO_DEVICE_API_OOB_PORT=6545 \
  ./build-native/test/gtest/native-device-api-test --gtest_filter=GpunetioDeviceApiTest.*
```

The test bounds device polling and uses metadata/control handshakes. Retain both
endpoint exits and payload/status results. These commands do not constitute a
new hardware validation receipt; main owns current-main build/GPU qualification.

## Evidence and proposed PR description boundary

The public increment is native GPU PUT/completion on the current main device
API. Its feature diff should contain the native mode/dispatcher arm, native
backend/view implementation, opt-in build wiring, and existing native test.
Already-merged progress, OOB, lifecycle, generic device-ops/wrapper and proxy
implementations belong to main and are not reintroduced as feature changes.

Historical DOCA 3.1 and ABI-2 validation belongs to the old branch/runtime only;
it does not qualify the API-3/DOCA-3.5 adaptation. Likewise the previously reported
H20 dependent compute-chain experiment is historical workload evidence, not a
new latency/bandwidth/serving claim for this merge. Current-main validation must
name its source head and matching runtime. No owner/warp/consumer experiment or
related fork's measurements are attributed to this native-only change.
