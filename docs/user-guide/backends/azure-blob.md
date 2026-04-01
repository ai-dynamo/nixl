---
title: Azure Blob
description: Azure Blob Storage backend for DRAM-to-object-storage transfers via Azure REST API.
---

## Overview

The Azure Blob backend enables DRAM-to-object-storage transfers using the Azure Blob REST API for workloads in Azure cloud environments.

| Property | Value |
|----------|-------|
| **Transfer Type** | DRAM ↔ Object |
| **Protocol** | Azure Blob REST API |
| **Best For** | Azure cloud deployments with Blob Storage |

## Installation

The Azure Blob backend requires the azure-storage-blobs and azure-identity packages from azure-sdk-for-cpp.

### Build from Source

```bash
git clone --depth 1 https://github.com/Azure/azure-sdk-for-cpp.git \
    --branch azure-storage-blobs_12.15.0
cd azure-sdk-for-cpp/
mkdir build && cd build
AZURE_SDK_DISABLE_AUTO_VCPKG=1 cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DDISABLE_AMQP=ON \
    -DDISABLE_AZURE_CORE_OPENTELEMETRY=ON
cmake --build . --target azure-storage-blobs azure-identity
cmake --install sdk/core
cmake --install sdk/storage/azure-storage-common
cmake --install sdk/storage/azure-storage-blobs
cmake --install sdk/identity
```

After installing the Azure SDK, rebuild NIXL to enable the Azure Blob backend.

## Configuration

### Environment Variables

<Markdown src="/snippets/env-vars-azure-blob.mdx" />

## When to Use

- **Azure cloud storage** -- Deployments where Blob Storage is the target object store.
- **Checkpoint workflows** -- Save and load model checkpoints on Azure infrastructure.
- **Azure-native object storage** -- Use this backend for Azure Blob; use [OBJ](./obj) for S3-compatible storage.
