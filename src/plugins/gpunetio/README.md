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

With `write_byte_striping=1` (default: `0`), preparation splits ranges with a
per-chunk byte budget when at least 4 MiB can be spread across QPs and the
descriptor-count chunks would leave some configured QPs unused. Addresses and
memory keys stay with each range. For multiple input descriptors, the final
descriptor remains intact in a separate chunk and is excluded from that byte
budget. Small requests, requests with enough
descriptor chunks, READ and the default single-QP mode keep their existing
mapping. This does not change completion or notification ordering.
For a single large descriptor, only its final eight payload bytes remain in
the final chunk, so the bulk can use all configured QPs before that chunk is
posted. These bytes remain ordinary payload, not a new receiver-ready protocol.
Splitting uses additional positions in the existing 32-position request ring,
so fewer prepared handles may fit. The opt-in is local to the sender and does
not change the QPN exchange or require enabling it on the receiver.
Payload pieces on different QPs may arrive out of order: consume only after
completion or notification. An application arrival marker should be a separate
final descriptor with descriptor merging disabled, not embedded inside a range.

### Reproduce byte-balanced WRITE

Use two distinct GPUs and one common NIXL core/NIXLBench executable, built with
the same release toolchain/SDK. Set `BASELINE_PLUGIN_DIR` to the main plugin and
`CANDIDATE_PLUGIN_DIR` to this branch's plugin. Select the NIC, OOB interface,
GID and GPU/NIC placement for your machine; the loopback address below measures
a same-host component, not physical-network or serving throughput.

```bash
export NIXLBENCH=/path/to/nixlbench
export BASELINE_PLUGIN_DIR=/path/to/main/plugins
export CANDIDATE_PLUGIN_DIR=/path/to/candidate/plugins
export NIC=your_rdma_device IFACE=your_oob_interface GID=your_gid_index
export SOURCE_GPU=0 TARGET_GPU=1
# Set LD_LIBRARY_PATH to the common NIXL and SDK runtime libraries first.
mkdir -p byte-write-results
for repeat in 1 2 3 4 5; do
  arms=(native1 native4 byte2 byte4)
  if ((repeat % 2 == 0)); then arms=(byte4 byte2 native4 native1); fi
  for arm in "${arms[@]}"; do
    qp=4; enabled=false; plugin="$BASELINE_PLUGIN_DIR"
    case "$arm" in
      native1) qp=1;;
      byte2) qp=2; enabled=true; plugin="$CANDIDATE_PLUGIN_DIR";;
      byte4) enabled=true; plugin="$CANDIDATE_PLUGIN_DIR";;
    esac
    args=(--runtime_type=ASIO --asio_address=127.0.0.1 --asio_port=19000
      --backend=GPUNETIO --worker_type=nixl --device_list="$NIC"
      --gpunetio_device_list=0 --gpunetio_oob_list="$IFACE" --gpunetio_gid_index="$GID"
      --gpunetio_data_qp_count="$qp" --gpunetio_write_byte_striping="$enabled"
      --initiator_seg_type=VRAM --target_seg_type=VRAM --op_type=WRITE
      --scheme=pairwise --mode=SG --num_threads=1
      --num_initiator_dev=1 --num_target_dev=1 --total_buffer_size=67108864
      --start_block_size=16777216 --max_block_size=16777216
      --start_batch_size=1 --max_batch_size=1 --pipeline_depth=1
      --num_iter=1000 --warmup_iter=100 --large_blk_iter_ftr=1 --check_consistency=true)
    export NIXL_PLUGIN_DIR="$plugin" CUDA_MODULE_LOADING=EAGER
    CUDA_VISIBLE_DEVICES="$SOURCE_GPU" "$NIXLBENCH" "${args[@]}" \
      --gpunetio_oob_port=19001 > "byte-write-results/$repeat-$arm-source.log" 2>&1 & p0=$!
    sleep 0.2
    CUDA_VISIBLE_DEVICES="$TARGET_GPU" "$NIXLBENCH" "${args[@]}" \
      --gpunetio_oob_port=19002 > "byte-write-results/$repeat-$arm-target.log" 2>&1 & p1=$!
    wait "$p0" || exit 1
    wait "$p1" || exit 1
    sleep 2
  done
done
```

Keep both endpoints' settings matched. Compare bandwidth, mean latency, Prep,
Post and Tx p99 separately; preparation is not included in the transfer window.
Avoid concurrent GPU workloads and pin the same CPUs for both arms when needed.
More QPs are not always faster: measure the 2-QP and 4-QP alternatives.

DOCA GPUNetIO backend takes these input parameters:
- network_devices: network device to be used during the execution (e.g. mlx5_0). Current release supports only 1 network device.
- oob_interface: network interface to be used when exchanging control info during initiator/target connection. Optional parameter, not needed if the network device is set in Ethernet mode.
- oob_port: TCP port used to listen for OOB connection setup. It must be an integer in the inclusive range [1, 65535] and defaults to 6544. Set a distinct port for each GPUNETIO backend sharing a host network namespace.
- gid_index: RoCE GID table index, in [0, 255]. Defaults to 0; select the GID for the test fabric when a non-default index is required.
- gpu_devices: GPU CUDA ID to be used during the execution (e.g. 0). Current release supports only 1 GPU device.
- write_byte_striping: enable byte-balanced large WRITE preparation with `1`; disabled by default (`0`). NIXLBench exposes this as `--gpunetio_write_byte_striping=true`.
- cuda_streams: how many CUDA streams the backend should created at setup time in the internal pool. Relevant only if the application wants to use the "stream pool" mode. If this parameter is not specified, default value is `DOCA_POST_STREAM_NUM`.

## Example

In directory `test/unit/plugins/doca` there is a sample named `nixl_gpunetio_stream_test.cu` where initiator and target exchange data through DOCA GPUNetIO backend.

NIXL bench also has the option to specify the DOCA GPUNetIO backend. An example command line to launch NIXL bench with GPU 0 andh 512 buffers per transfer can be:
```
 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/gdrcopy/src:/opt/mellanox/doca NIXL_PLUGIN_DIR=/path/to/nixl/lib/x86_64-linux-gnu/plugins CUDA_MODULE_LOADING=EAGER ./nixlbench --etcd-endpoints http://etc_server_ip:2379 --backend=GPUNETIO --initiator_seg_type=VRAM --target_seg_type=DRAM --runtime_type=ETCD --gpunetio_device_list=0 --device_list=mlx5_0 --start_batch_size=512 --max_batch_size=512 --total_buffer_size=34359738368
```

## Caveats

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
