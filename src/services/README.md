<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL Compression Support with nvCOMP (Experimental)

This directory contains an experimental service utility agent built on top of `nixlAgent`.  
It adds pluggable marshal-based services, including compression support, on top of the native NIXL agent API.

[License](https://opensource.org/licenses/Apache-2.0)

## Feature Overview

### What is it ?

A transparent data compression and decompression service integrated into NVIDIA’s Inference Xfer Library (NIXL). It uses nvCOMP, NVIDIA’s GPU-accelerated data (de)compression library, to reduce payload sizes and help reduce transfer times for frequently accessed data in AI inference workloads.

### Problem it solves

Frequent model-weight transfers in reinforcement learning (RL) workflows and growing KV caches in inference workloads move increasingly large volumes of data. Transferring these payloads uncompressed can create bottlenecks in network bandwidth, transfer latency, and storage capacity.

This repository addresses these bottlenecks by integrating nvCOMP-based compression and decompression into the NIXL transfer path, reducing the amount of data that must be transferred and stored.

### How it works

- Pluggable service layer operates like NIXL's backend plugins
- Automatically compresses data before transmission and decompresses it upon reaching the target.
- Zero-touch for applications, handles compression transparently

### Current Status

- Experimental branch feature; API and behavior may change.
- Current focus is GPU compression with **WRITE** and **READ** transfers.

## Available Services (Marshal Modes)

- `nixlMarshalDirectConfig`
  - Pass-through mode (minimal service processing).
- `nixlMarshalCompressConfig`
  - nvCOMP-based compression mode (only available when nvCOMP is found at build time).

## nvCOMP Installation

For this branch, use the **official nvCOMP 5.3 release package**.

Download and install the official package from [https://developer.nvidia.com/nvcomp-downloads](https://developer.nvidia.com/nvcomp-downloads).

## Build / Compilation Flags (Meson)

Enable service builds:

```bash
meson setup build \
  -Dbuild_nixl_service=true \
  -Dbuild_tests=true

# Optional overrides (use only if nvCOMP is not in default search paths):
#   -Dnvcomp_path_inc=/path/to/nvcomp/include
#   -Dnvcomp_path_lib=/path/to/nvcomp/lib

ninja -C build install
```

## Run Python Service Example

```bash
# Marshalled WRITE (default): rank 0 pushes into rank 1's buffer.
python ./examples/python/service_api_example.py

# Marshalled READ: rank 0 pulls from rank 1's buffer instead.
python ./examples/python/service_api_example.py --direction read
```

## Notes

1. **nvCOMP + supported NVIDIA CUDA hardware are required** for compression service scenarios.
2. Tested with UCX RDMA transport (`UCX_TLS=^cuda_ipc`).
3. Tested on H100.
