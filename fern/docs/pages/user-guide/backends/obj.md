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

Backend parameters are string key-value pairs passed when the `OBJ` backend is created. Parameter names are case-sensitive.

### Backend Parameters

| Parameter | Accepted value | Default | Description |
|---|---|---|---|
| `access_key` | String | AWS credential chain | AWS access-key ID. Use together with `secret_key`. |
| `secret_key` | String | AWS credential chain | AWS secret access key. Use together with `access_key`. |
| `session_token` | String | None | Session token for temporary credentials. |
| `bucket` | String | `AWS_DEFAULT_BUCKET` | S3 bucket used for object operations. Backend creation fails if neither source provides a bucket. |
| `endpoint_override` | URL | `AWS_ENDPOINT_OVERRIDE` or SDK default | Overrides the S3 endpoint for S3-compatible services. |
| `scheme` | `http` or `https` | `https` | HTTP scheme used by the S3 client. |
| `region` | AWS region string | `us-east-1` | Region used for request signing and endpoint selection. |
| `use_virtual_addressing` | `true` or `false` | `false` | Enables virtual-hosted-style bucket addressing. |
| `req_checksum` | `required` or `supported` | AWS SDK default | Controls request checksum calculation. |
| `resp_checksum` | `required` or `supported` | AWS SDK default | Controls response checksum validation. |
| `ca_bundle` | File path | System default | CA certificate bundle used for TLS verification. |
| `num_threads` | Unsigned integer | Half the hardware threads, minimum 1 | Worker threads used by the standard S3 client executor. |
| `crtMinLimit` | Size in bytes | Disabled | Enables the S3 CRT client and selects it for objects at least this size. |
| `throughput_target_gbps` | Whole-number Gbps | `10` | CRT throughput target used to size its parallel connection pool. |
| `accelerated` | `true` or `false` | `false` | Selects an accelerated object-storage engine when one is available. |
| `type` | Engine name | None | Accelerated-engine implementation, for example `dell`. |

### CRT Client Tuning

Without `crtMinLimit`, the backend uses the standard S3 client for every transfer. When it is set, smaller objects continue to use the standard client and objects whose size is greater than or equal to the threshold use the CRT client.

The threshold also configures the CRT client's multipart-upload threshold and part size. AWS S3 requires every part except the last to be at least 5 MiB (5,242,880 bytes). If `crtMinLimit` is smaller, the AWS SDK clamps the part size to 5 MiB and logs a warning. Use a value of at least `5242880` to avoid that clamp; `10485760` (10 MiB) is a practical starting point.

`throughput_target_gbps` must be a whole number. Raising it allows the CRT scheduler to open more parallel connections on higher-bandwidth links.

```cpp title="High-throughput S3 CRT configuration"
nixl_b_params_t params = {
    {"bucket", "large-model-storage"},
    {"region", "us-west-2"},
    {"crtMinLimit", "10485760"},
    {"throughput_target_gbps", "25"}
};

agent.createBackend("OBJ", params);
```

<Note>
Accelerated engines are separate from CRT selection. For the Dell ObjectScale engine, set `accelerated` to `true` and `type` to `dell`; the optional acceleration dependency must also be present when NIXL is built.
</Note>

### Environment Variables

<Markdown src="/snippets/env-vars-obj.mdx" />

## When to Use

- **DRAM to S3-compatible object stores** -- Transfer host memory to and from any S3-compatible storage service.
- **Cloud checkpoint storage** -- Save and load model checkpoints to cloud object storage.
- **Multiple executor variants** -- Supports standard S3, AWS CRT-accelerated, and Dell-accelerated executors.
