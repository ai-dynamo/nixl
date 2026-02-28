<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NIXL Dell ObjectScale accelerated S3 engine

This vendor-specific accelerated engine provides S3 over RDMA for Dell ObjectScale.  This engine utilizes the CUDA Toolkit CUObject Client library and the AWS S3 SDK.

If a Dell ObjectScale endpoint is utilized, but RDMA is not enabled, the standard S3 engine should be used instead.

## Dependencies

This engine requires aws-sdk-cpp.  This dependency and steps to install it are provided in the OBJ backend documentation.

This engine requires the CUDA Toolkit version 13.1.1 or later to be installed.

[CUDA GDS Install and Setup](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

The CUDA toolkit provides the CUObject Client library, which is required for this engine to function.  The library is included in the CUDA Toolkit installation.
```

This engine requires the CUDA Toolkit version 13.1.1 or later to be installed.

[CUDA GDS Install and Setup](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

The CUDA toolkit provides the CUObject Client library, which is required for this engine to function.  The library is included in the CUDA Toolkit installation.

## Configuration

The Dell ObjectScale engine supports configuration through two mechanisms: backend parameter maps passed during backend creation, and environment variables for system-wide settings. Backend parameters take precedence over environment variables.  The backend parameters described below are required to enable the Dell ObjectScale engine.

### Backend Parameters

Backend parameters are passed as a key-value map (`nixl_b_params_t`) when creating the backend instance. The Dell ObjectScale Object Storage backend supports AWS S3-compatible storage.  These parameters are described in the OBJ backend documentation.  In addition to the parameters described in the OBJ backend documentation, the following parameters are required to enable the Dell ObjectScale engine:

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `req_checksum` | Request checksum validation (`required`/`supported`) | - | No |
| `accelerated` | Enable accelerated engine (`true`/`false`) | `false` | No |
| `type` | Vendor Type for accelerated engine | - | No |

To enable the Dell ObjectScale engine, the `accelerated` parameter must be set to `true` and the `type` parameter must be set to `dell`.  It is recomended that the `req_checksum` parameter be set to `required` to ensure data integrity.

### Configuration Examples

#### Dell ObjectScale Endpoint

```cpp
nixl_b_params_t params = {
    {"access_key", "test"},
    {"secret_key", "secret"},
    {"bucket", "test-bucket"},
    {"endpoint_override", "http://10.10.10.10:9000"},
    {"scheme", "http"},
    {"use_virtual_addressing", "false"},
    {"req_checksum", "required"},
    {"ca_bundle", "/root/ca-certs/cacert.pem"},
    {"accelerated", "true"},
    {"type", "dell"}
};
agent.createBackend("OBJ", params);
```

#### Environment Variable Configuration

```bash
export AWS_DEFAULT_BUCKET=my-inference-bucket
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=secret
```

```cpp
// Minimal parameter map when using environment variables
nixl_b_params_t params = {
    {"endpoint_override", "http://10.10.10.10:9000"},
    {"req_checksum", "required"},
    {"accelerated", "true"},
    {"type", "dell"}};
agent.createBackend("OBJ", params);
```

### CUObject Client Configuration

The CUObject Client library requires the configuration of the RDMA device address list in the JSON configuration file.  Each client IP address associated with an RDMA device is specified in the "rdma_dev_addr_list" property.  The following is an example JSON file:

```json
{
    "execution": {
        "parallel_io" : false
    },

    "properties": {
        "allow_compat_mode": true,
        "use_pci_p2pdma": true,
        "rdma_peer_type": "dmabuf",
        "rdma_dev_addr_list": ["10.0.1.2", "10.0.2.2"]
    }
}
```

Export the location of the JSON file with the environment variable

```bash
export CUFILE_ENV_PATH_JSON=/path/to/cufile.json
```

## Supported Memory Types

The Dell ObjectScale engine supports the following memory types:

- `OBJ_SEG` - Object Storage segment
- `DRAM_SEG` - DRAM segment
- `VRAM_SEG` - GPU memory segment

## Transfer Operations

The Dell ObjectScale Object Storage backend supports read and write operations between local CPU or GPU memory and S3 objects. Here are the key aspects of transfer operations:

### Device ID to Object Key Mapping

- Each object in S3 is identified by a unique object key
- The backend maintains a mapping between device IDs (`devId`) and object keys
- When registering memory:
  - If `metaInfo` is provided in the blob descriptor, it is used as the object key
  - Otherwise, the device ID is converted to a string and used as the object key
- This mapping is used during transfer operations to locate the correct S3 object

### Read Operations

- Read operations support reading from a specific offset within an object
- The offset is specified in the remote metadata's `addr` field
- The read operation will fetch data starting from this offset
- The amount of data read is determined by the `len` field in the local metadata
- The local memory buffer is written using RDMA by the ObjectScale endpoint using the requested object data

### Write Operations

- Write operations currently do not support offsets
- Attempting to write with a non-zero offset will result in an error
- The entire object is written at once
- The data to write is taken from the local memory buffer specified in the local metadata
- The local memory buffer is read using RDMA by the ObjectScale endpoint and persisted to object storage

### Asynchronous Operations

- All transfer operations are asynchronous
- The backend uses a thread pool executor for handling async operations
- Operation completion is tracked through the request handle
- The `checkXfer` function can be used to poll for operation completion
- The request handle must be released using `releaseReqH` after the operation is complete
