<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DOCA GPUNetIO backend takes 3 input parameters:
- network_devices: network device to be used during the execution (e.g. mlx5_0). Current release supports only 1 network device.
- oob_interface: network interface to be used when exchanging control info during initiator/target connection. Optional parameter, not needed if the network device is set in Ethernet mode.
- gpu_devices: GPU CUDA ID to be used during the execution (e.g. 0). Current release supports only 1 GPU device.
- cuda_streams: how many CUDA streams the backend should created at setup time in the internal pool. Relevant only if the application wants to use the "stream pool" mode. If this parameter is not specified, default value is `DOCA_POST_STREAM_NUM`.

## Example

In directory `test/unit/plugins/doca` there is a sample named `nixl_gpunetio_stream_test.cu` where initiator and target exchange data through DOCA GPUNetIO backend.

NIXL bench also has the option to specify the DOCA GPUNetIO backend. An example command line to launch NIXL bench with GPU 0 andh 512 buffers per transfer can be:
```
 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/gdrcopy/src:/opt/mellanox/doca NIXL_PLUGIN_DIR=/path/to/nixl/lib/x86_64-linux-gnu/plugins CUDA_MODULE_LOADING=EAGER ./nixlbench --etcd-endpoints http://etc_server_ip:2379 --backend=GPUNETIO --initiator_seg_type=VRAM --target_seg_type=DRAM --runtime_type=ETCD --gpunetio_device_list=0 --device_list=mlx5_0 --start_batch_size=512 --max_batch_size=512 --total_buffer_size=34359738368
```

## Caveats

To run for performance (e.g. with NIXL bench) it's highly recommended to build NIXL with `buildtype=release`.
For debugging purposes, NIXL can be built with `buildtype=debug`.
