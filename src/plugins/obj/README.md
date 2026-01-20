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

# NIXL Object Storage Plugin

This backend provides AWS S3 object storage using aws-sdk-cpp version 1.11 with a dual-client architecture for optimal performance across different object sizes.

## Architecture

The Object Storage backend uses two S3 clients to optimize performance based on object size:

- **Standard S3 Client** (`awsS3Client`): Uses the traditional AWS SDK S3 client for smaller objects
- **S3 CRT Client** (`awsS3CrtClient`): Uses AWS Common Runtime (CRT) for high-performance transfers of large objects

The CRT client provides significantly improved throughput and lower CPU utilization for large objects through optimized multipart uploads/downloads and connection pooling. The backend automatically selects the appropriate client based on the object size and the configured `crtMinLimit` threshold.

## Dependencies

This backend requires aws-sdk-cpp version 1.11 to be installed with both `s3` and `s3-crt` components. Example CLI to compile from sources:

```bash
# Ubuntu/Debian
apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581 && mkdir sdk_build && cd sdk_build && cmake ../aws-sdk-cpp/ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3;s3-crt" -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && make -j && make install
```

## Configuration

The Object Storage backend supports configuration through two mechanisms: backend parameter maps passed during backend creation, and environment variables for system-wide settings. Backend parameters take precedence over environment variables.

### Backend Parameters

Backend parameters are passed as a key-value map (`nixl_b_params_t`) when creating the backend instance. The Object Storage backend supports AWS S3-compatible storage and accepts the following parameters:

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `access_key` | AWS access key ID for authentication | - | No* |
| `secret_key` | AWS secret access key for authentication | - | No* |
| `session_token` | AWS session token for temporary credentials | - | No |
| `bucket` | S3 bucket name for operations | - | Yes** |
| `endpoint_override` | Custom S3 endpoint URL | - | No*** |
| `scheme` | HTTP scheme (`http` or `https`) | `https` | No |
| `region` | AWS region for the S3 service | `us-east-1` | No |
| `use_virtual_addressing` | Use virtual-hosted-style addressing (`true`/`false`) | `false` | No |
| `req_checksum` | Request checksum validation (`required`/`supported`) | - | No |
| `ca_bundle` | path to a custom certificate bundle | - | No |
| `crtMinLimit` | Minimum object size (bytes) to use S3 CRT client for high-performance transfers | Disabled**** | No |

\* If `access_key` and `secret_key` are not provided, the AWS SDK will attempt to use default credential providers (IAM roles, environment variables, credential files, etc.)

\** If `bucket` parameter is not provided, the `AWS_DEFAULT_BUCKET` environment variable will be used as fallback.

\*** If `endpoint_override` parameter is not provided, the `AWS_ENDPOINT_OVERRIDE` environment variable will be used as fallback.

\**** If `crtMinLimit` is not provided, the S3 CRT client is disabled and all transfers use the standard S3 client. When set, objects with size >= `crtMinLimit` will use the high-performance CRT client, while smaller objects continue to use the standard client. Recommended value: 10485760 (10 MB) or higher for optimal performance on large objects.

### Environment Variables

The following environment variables are supported for Object Storage configuration:

| Variable | Description | Example |
|----------|-------------|---------|
| `AWS_DEFAULT_BUCKET` | Default S3 bucket name when not specified in parameters | `my-default-bucket` |
| `AWS_ENDPOINT_OVERRIDE` | Custom S3 endpoint URL when not specified in parameters | `http://localhost:9000` |

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

#### Minimal Configuration (using IAM role)

```cpp
nixl_b_params_t params = {{"bucket", "my-bucket"}};
agent.createBackend("obj", params);
```

#### Full Configuration with Explicit Credentials

```cpp
nixl_b_params_t params = {
    {"access_key", "AKIAIOSFODNN7EXAMPLE"},
    {"secret_key", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"},
    {"bucket", "inference-data"},
    {"region", "us-west-2"},
    {"use_virtual_addressing", "true"}
};
agent.createBackend("obj", params);
```

#### Custom S3-Compatible Endpoint

```cpp
nixl_b_params_t params = {
    {"access_key", "minioadmin"},
    {"secret_key", "minioadmin"},
    {"bucket", "test-bucket"},
    {"endpoint_override", "http://localhost:9000"},
    {"scheme", "http"},
    {"region", "us-east-1"},
    {"use_virtual_addressing", "false"},
    {"req_checksum", "supported"},
    {"ca_bundle", "/root/ca-certs/cacert.pem"}
};
agent.createBackend("obj", params);
```

#### Environment Variable Configuration

```bash
export AWS_DEFAULT_BUCKET=my-inference-bucket
export AWS_ACCESS_KEY_ID=EXAMPLE_KEY_ID
export AWS_SECRET_ACCESS_KEY=EXAMPLE_SECRET_ACCESS_KEY
export AWS_REGION=us-west-2
```

```cpp
// Minimal parameter map when using environment variables
nixl_b_params_t params = {{"use_virtual_addressing", "true"}};
agent.createBackend("obj", params);
```

#### Custom S3-Compatible Endpoint via Environment Variable

```bash
export AWS_DEFAULT_BUCKET=test-bucket
export AWS_ENDPOINT_OVERRIDE=http://localhost:9000
export AWS_ACCESS_KEY_ID=EXAMPLE_KEY_ID
export AWS_SECRET_ACCESS_KEY=EXAMPLE_SECRET_ACCESS_KEY
```

```cpp
// Backend will use AWS_ENDPOINT_OVERRIDE for the S3 endpoint
nixl_b_params_t params = {{"use_virtual_addressing", "false"}};
agent.createBackend("obj", params);
```

#### Using AWS Profiles

```bash
export AWS_PROFILE=production
```

```cpp
nixl_b_params_t params = {
    {"bucket", "production-bucket"},
    {"region", "us-east-1"}
};
agent.createBackend("obj", params);
```

#### High-Performance Configuration with S3 CRT Client

```cpp
nixl_b_params_t params = {
    {"bucket", "large-model-storage"},
    {"region", "us-west-2"},
    {"crtMinLimit", "10485760"}  // Use CRT client for objects >= 10 MB
};
agent.createBackend("obj", params);
```

This configuration automatically uses the high-performance S3 CRT client for objects 10 MB and larger, while smaller objects continue to use the standard S3 client. The CRT client provides:

- **Higher Throughput**: Optimized multipart transfers with automatic chunking and parallelization
- **Lower CPU Usage**: Efficient connection pooling and memory management
- **Better for Large Objects**: Particularly beneficial for model weights, checkpoints, and large datasets

## Transfer Operations

The Object Storage backend supports read and write operations between local memory and S3 objects. Here are the key aspects of transfer operations:

### Automatic Client Selection

The backend automatically selects the appropriate S3 client for each transfer operation based on the object size:

- **Standard S3 Client**: Used for objects smaller than `crtMinLimit` (or all objects if `crtMinLimit` is not set)
- **S3 CRT Client**: Used for objects with size >= `crtMinLimit`, providing optimized performance for large transfers

The client selection happens transparently during transfer operations, requiring no changes to application code. The selection is logged at DEBUG level for monitoring and troubleshooting.

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

### Write Operations

- Write operations currently do not support offsets
- Attempting to write with a non-zero offset will result in an error
- The entire object is written at once
- The data to write is taken from the local memory buffer specified in the local metadata

### Asynchronous Operations

- All transfer operations are asynchronous
- The backend uses a thread pool executor for handling async operations
- Operation completion is tracked through the request handle
- The `checkXfer` function can be used to poll for operation completion
- The request handle must be released using `releaseReqH` after the operation is complete
