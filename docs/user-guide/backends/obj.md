---
title: OBJ
description: Object storage backend supporting S3-compatible stores with multiple executor variants.
---

## Overview

The OBJ backend provides object storage support for S3-compatible stores. It includes three executor variants: standard S3, S3_CRT (AWS CRT-based for higher throughput), and S3/RDMA (Dell accelerated). The executor is selected based on the available libraries and configuration.

| Property | Value |
|----------|-------|
| **Transfer Type** | DRAM ↔ Object |
| **Protocol** | S3, S3_CRT, S3/RDMA |
| **Best For** | Cloud object storage transfers |

## Installation

The OBJ backend requires aws-sdk-cpp version 1.11 with `s3` and `s3-crt` components.

### Build aws-sdk-cpp from Source

```bash
# Install system dependencies (Ubuntu/Debian)
apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev

# Build and install aws-sdk-cpp
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581
mkdir sdk_build && cd sdk_build
cmake ../aws-sdk-cpp/ \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_ONLY="s3;s3-crt" \
    -DENABLE_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local
make -j
make install
```

After installing aws-sdk-cpp, rebuild NIXL to enable the OBJ backend.

### Optional: S3 Accelerated Engines

For GPU-direct and accelerated object storage operations, install `cuobjclient-13.1`. If not available during build, the S3 Accelerated engines are automatically disabled and the plug-in falls back to standard S3 and S3 CRT engines.

## Configuration

### Environment Variables

<Markdown src="/snippets/env-vars-obj.mdx" />

## When to Use

- **DRAM to S3-compatible object stores** -- Transfer host memory to and from any S3-compatible storage service.
- **Cloud checkpoint storage** -- Save and load model checkpoints to cloud object storage.
- **Multiple executor variants** -- Supports standard S3, AWS CRT-accelerated, and Dell-accelerated executors.
