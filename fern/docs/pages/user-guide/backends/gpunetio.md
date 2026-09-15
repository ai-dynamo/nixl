---
title: DOCA GPUNetIO
description: DOCA GPUNetIO backend for high-performance GPU-to-GPU transfers using GPUDirect Async.
---

## Overview

The DOCA GPUNetIO backend provides high-performance GPU-to-GPU transfers using GPUDirect Async over the DOCA networking stack on supported NVIDIA SmartNICs.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ DRAM |
| **Protocol** | DOCA GPUDirect Async |
| **Best For** | Ultra-low-latency GPU-to-GPU transfers on DOCA-capable systems |

## Installation

### Prerequisites

The DOCA GPUNetIO backend requires:

- **DOCA SDK** -- Version 3.5.0 or later, from the [NVIDIA DOCA SDK](https://developer.nvidia.com/doca-downloads) page. Older DOCA releases use an incompatible GPUNetIO device verbs API; the plugin is skipped when `doca-gpunetio` reports a lower version.
- **GPUDirect Async-capable hardware** -- NVIDIA BlueField SmartNICs (DPUs)
- **CUDA Toolkit** -- Version 12.8 or later

For system configuration and setup details, see the [DOCA GPUNetIO Programming Guide](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html).

## Configuration

The DOCA GPUNetIO backend has no backend-specific environment variables or build options.

## When to Use

- Ultra-low-latency GPU-to-GPU transfers
- Deployments with DOCA-capable NVIDIA SmartNICs (BlueField DPUs)
- Workloads preferring the GPUDirect Async path over UCX RDMA
