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

This engine requires aws-sdk-cpp version 1.11 to be installed. Example CLI to compile from sources:

```bash
# Ubuntu/Debian
apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581 && mkdir sdk_build && cd sdk_build && cmake ../aws-sdk-cpp/ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3" -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && make -j && make install
```

This engine requires the CUDA Toolkit version 13.1.1 or later to be installed.

[CUDA GDS Install and Setup](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

The CUDA toolkit provides the CUObject Client library, which is required for this engine to function.  The library is included in the CUDA Toolkit installation.

## Configuration

The Dell ObjectScale engine supports configuration through two mechanisms: backend parameter maps passed during backend creation, and environment variables for system-wide settings. Backend parameters take precedence over environment variables.

### Backend Parameters

Backend parameters are passed as a key-value map (`nixl_b_params_t`) when creating the backend instance. The Dell ObjectScale Object Storage backend supports AWS S3-compatible storage and accepts the following parameters:

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `access_key` | AWS access key ID for authentication | - | No* |
| `secret_key` | AWS secret access key for authentication | - | No* |
| `session_token` | AWS session token for temporary credentials | - | No |
| `bucket` | S3 bucket name for operations | - | Yes** |
| `endpoint_override` | Custom S3 endpoint URL | - | No |
| `scheme` | HTTP scheme (`http` or `https`) | `https` | No |
| `region` | AWS region for the S3 service | `us-east-1` | No |
| `use_virtual_addressing` | Use virtual-hosted-style addressing (`true`/`false`) | `false` | No |
| `req_checksum` | Request checksum validation (`required`/`supported`) | - | No |
| `ca_bundle` | path to a custom certificate bundle | - | No |
| `accelerated` | Enable accelerated engine (`true`/`false`) | `false` | No |
| `type` | Vendor Type for accelerated engine | - | No |

\* If `access_key` and `secret_key` are not provided, the AWS SDK will attempt to use default credential providers (IAM roles, environment variables, credential files, etc.)

\** If `bucket` parameter is not provided, the `AWS_DEFAULT_BUCKET` environment variable will be used as fallback.

\*** To utilize the Dell ObjectScale accelerated engine, the `accelerated` parameter must be set to `true` and the `type` parameter must be set to `dell`.

### Environment Variables

The following environment variables are supported for Object Storage configuration:

| Variable | Description | Example |
|----------|-------------|---------|
| `AWS_DEFAULT_BUCKET` | Default S3 bucket name when not specified in parameters | `my-default-bucket` |

Standard AWS SDK environment variables are also supported when credentials are not provided via backend parameters. For a complete list and detailed documentation, see the [AWS CLI Environment Variables](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) documentation.

Common AWS SDK environment variables include:

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key ID |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key |
| `AWS_SESSION_TOKEN` | AWS session token for temporary credentials |
| `AWS_REGION` | Default AWS region |
| `AWS_PROFILE` | AWS credential profile to use |

### Configuration Priority

Configuration values are resolved in the following priority order (highest to lowest):

1. **Backend Parameters**: Values passed directly in the backend parameter map
2. **Environment Variables**: AWS SDK environment variables and `AWS_DEFAULT_BUCKET`
3. **AWS Credential Chain**: Default AWS credential providers (IAM roles, credential files, etc.)
4. **Default Values**: Built-in default values

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

## Transfer Operations

The ObjectScale Object Storage backend supports read and write operations between local CPU or GPU memory and S3 objects. Here are the key aspects of transfer operations:

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
