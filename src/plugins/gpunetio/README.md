<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NIXL DOCA GPUNetIO Plugin

This plugin uses NVIDIA DOCA GPUNetIO library to enable GPUDirect Async Kernel-Initiated (GDAKI) network communications (a.k.a. GPU-driven communications) for NIXL as a backend.
In order to correctly enable and use this backend, please refer to the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) programming guide to properly configure the system.

## Mode of operations

DOCA GPUNetIO backend enhances NIXL with GPU communications in "stream" mode. It means the backend provides an internal CUDA kernel to execute RDMA Write or RDMA Read to be launched on a CUDA stream.

There are two mode of operation:
- Stream attached: when a transfer request is created with `createXferRequest`, a CUDA stream is specified in the `extra_params.customParam` parameter. When postXfer is called, the backend will launch the CUDA kernel responsible for the RDMA Write or RDMA Read using the CUDA stream specified by the application.
- Stream pool: when a transfer request is created with `createXferRequest`, no CUDA stream is specified in the `extra_params.customParam` parameter. When postXfer is called, the backend will launch the CUDA kernel responsible for the RDMA Write or RDMA Read using a CUDA stream from an internal pool of streams created by the backend at setup time.

Stream attached mode is optimal when the application has a sequence of CUDA tasks enqueued on the GPU to process data so it can attach, on the same stream, a final RDMA Write CUDA kernel. This way, the CPU can asynchronously submit on a single CUDA stream both data processing and network communications.

Stream pool mode instead is when applications mostly wants to process data on the CPU but still wants to take the advange of GPU taking care of network communications in a parallel way (every CUDA communication kernel will be executed in parallel to others on a different stream).

## Input parameters

`data_qp_count` optionally selects 1–4 WRITE QPs per peer (default 1). Both peers
must use the same count. Counts above 1 retain a dedicated READ QP, stripe WRITE
chunks over one block per QP in a single kernel launch and require
`data_qp_count <= cuda_streams <= 4`.
The final chunk waits for all preceding chunks and the previous peer WRITE
before posting, preserving ordering-tail/notification semantics. Each block
posts its non-final chunks and publishes arrival without waiting for other
blocks. The last arriving block posts the final chunk after the existing RDMA
completion checks. Requests share one posting stream, avoiding cross-stream
events while potentially reducing overlap between requests. Attached CUDA
streams are supported for WRITE only with count 1. Single-QP mode preserves the
legacy OOB format; multi-QP mode uses a versioned QPN exchange.

DOCA GPUNetIO backend takes these input parameters:
- network_devices: network device to be used during the execution (e.g. mlx5_0). Current release supports only 1 network device.
- oob_interface: network interface to be used when exchanging control info during initiator/target connection. Optional parameter, not needed if the network device is set in Ethernet mode.
- oob_port: TCP port used to listen for OOB connection setup. It must be an integer in the inclusive range [1, 65535] and defaults to 6544. Set a distinct port for each GPUNETIO backend sharing a host network namespace.
- gid_index: RoCE GID table index, in [0, 255]. Defaults to 0; select the GID for the test fabric when a non-default index is required.
- gpu_devices: GPU CUDA ID to be used during the execution (e.g. 0). Current release supports only 1 GPU device.
- cuda_streams: how many CUDA streams the backend should create at setup time in the internal pool. Relevant only if the application wants to use the "stream pool" mode. It must be an integer in the inclusive range [1, 4]; if this parameter is not specified, default value is `DOCA_POST_STREAM_NUM` (4).

## Example

In directory `test/unit/plugins/doca` there is a sample named `nixl_gpunetio_stream_test.cu` where initiator and target exchange data through DOCA GPUNetIO backend.

NIXL bench also has the option to specify the DOCA GPUNetIO backend. An example command line to launch NIXL bench with GPU 0 andh 512 buffers per transfer can be:
```
 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/gdrcopy/src:/opt/mellanox/doca NIXL_PLUGIN_DIR=/path/to/nixl/lib/x86_64-linux-gnu/plugins CUDA_MODULE_LOADING=EAGER ./nixlbench --etcd-endpoints http://etc_server_ip:2379 --backend=GPUNETIO --initiator_seg_type=VRAM --target_seg_type=DRAM --runtime_type=ETCD --gpunetio_device_list=0 --device_list=mlx5_0 --start_batch_size=512 --max_batch_size=512 --total_buffer_size=34359738368
```

## Caveats

### Descriptor size

One RDMA operation carries at most `DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE` (1 GiB), so a larger
descriptor is posted as several consecutive operations of at most 1 GiB each. A request holds at
most 32 x 512 operations; `prepXfer` rejects larger requests with `NIXL_ERR_INVALID_PARAM`.

### Notifications

Each peer has 4096 notification send buffers. A prepared request with a notification owns one
until the request is released, and `genNotif` holds one while it sends. When all of them are owned,
`prepXfer` and `genNotif` return `NIXL_ERR_BACKEND`.

### Comparing striped WRITE submission designs

For a matched comparison, build both versions with the same release toolchain
and DOCA SDK, then use one common core/NIXLBench executable and switch only
`NIXL_PLUGIN_DIR`. Use two GPUs with MPS off. Set `GPU`, `OOB_PORT`, `NIC`,
`IFACE`, `GID` and `ASIO_PORT` for your system; each worker gets a different GPU
and OOB port, but the same ASIO address/port and remaining options. Start the
first worker before the second:

```bash
CUDA_VISIBLE_DEVICES="$GPU" CUDA_MODULE_LOADING=EAGER CUDA_DEVICE_MAX_CONNECTIONS=8 \
nixlbench --runtime_type=ASIO --asio_address=127.0.0.1 --asio_port="$ASIO_PORT" \
  --backend=GPUNETIO --worker_type=nixl --device_list="$NIC" \
  --gpunetio_device_list=0 --gpunetio_oob_list="$IFACE" \
  --gpunetio_gid_index="$GID" --gpunetio_oob_port="$OOB_PORT" \
  --gpunetio_data_qp_count=4 --initiator_seg_type=VRAM --target_seg_type=VRAM \
  --op_type=WRITE --scheme=pairwise --mode=SG --num_threads=1 \
  --num_initiator_dev=1 --num_target_dev=1 --total_buffer_size=268435456 \
  --start_block_size=4096 --max_block_size=4096 \
  --start_batch_size=4096 --max_batch_size=4096 --pipeline_depth=1 \
  --num_iter=100 --warmup_iter=20 --large_blk_iter_ftr=1 --check_consistency=true
```

Repeat five fresh-worker pairs per version, alternating their order. Repeat with
`--pipeline_depth=4` to check request overlap and count1/depth1 as the default-path
control. Report Post/Prep times, Tx p99 and payload bandwidth separately. These
are component measurements, not serving latency or NIC-only latency.

### Peer-owned GPU memory and two-SGE WRITE

`gpu_devices` selects the communication GPU. In stream-pool mode, a VRAM
descriptor may name another visible GPU as its memory owner. Registration runs
under that owner's CUDA device; posting runs on the communication GPU and
restores the caller's device. Foreign VRAM uses legacy PeerDirect registration
(`ibv_reg_mr`, requiring compatible peer-memory support), rather than exporting
it with the communication GPU's DOCA handle. Synchronize producer writes before
posting, retain allocations and registrations until completion, and establish
receiver GPU visibility before consumption. Attached application streams are
not supported for peer-owned payloads. This is ordinary GPU memory, not VMM or
multicast-memory support.

Eligible consecutive WRITE descriptors with adjacent remote ranges and the same
remote key use one two-SGE WQE. Source addresses and local keys may differ; no
gather buffer is allocated. Non-adjacent or different-key ranges, zero lengths,
and pairs exceeding the SDK transfer limit keep the one-SGE path. The final
descriptor stays separate for arrival-marker ordering. READ, dense CQ signaling,
request completion and notification polling policies are unchanged.

For a component comparison, build main and this PR with the same supported
DOCA SDK and run the existing NIXLBench pairwise WRITE with fixed bytes,
descriptor count, pipeline depth and consistency checking, selecting one GPU
per worker. This measures WRITE pairing with owner-matching memory; it does not
by itself validate peer ownership or serving performance.

To run for performance (e.g. with NIXL bench) it's highly recommended to build NIXL with `buildtype=release`.
For debugging purposes, NIXL can be built with `buildtype=debug`.
