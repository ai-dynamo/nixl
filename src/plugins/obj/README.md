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

### Required Dependencies

This backend requires aws-sdk-cpp version 1.11 to be installed with both `s3` and `s3-crt` components. Example CLI to compile from sources:

```bash
# Ubuntu/Debian
apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581 && mkdir sdk_build && cd sdk_build && cmake ../aws-sdk-cpp/ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3;s3-crt" -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && make -j && make install
```

### Optional Dependencies

**S3 Accelerated Engines** (`cuobjclient-13.1`): Required for GPU-direct and accelerated object storage operations. When available, enables:
- `S3AccelObjEngineImpl` - Base accelerated S3 engine
- Vendor-specific accelerated implementations under `s3_accel/`

If `cuobjclient-13.1` is not found during build, the S3 Accelerated engines will be automatically disabled, and the plugin will fall back to standard S3 and S3 CRT engines.

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

## Extending with Vendor-Specific Implementations

The object plugin uses a modular, inheritance-based architecture that makes it easy to add vendor-specific backends without modifying core engine logic.

### Architecture Overview

The plugin uses a **pImpl (Pointer to Implementation)** design pattern to provide:
- **ABI Stability**: Interface changes don't require recompiling client code
- **Modularity**: Easy to add vendor-specific engines without modifying core logic
- **Encapsulation**: Implementation details hidden behind abstract interface

The architecture separates concerns into:

- **Clients**: Handle S3-compatible storage operations (inheriting from `awsS3Client` or `awsS3AccelClient`)
- **Engine Implementations**: Manage NIXL backend operations (inheriting from `nixlObjEngineImpl`)

#### Current Hierarchy

```
+----------------------+
|   nixlObjEngine      |  (public interface)
+----------------------+
           |
           | uses (pImpl)
           v
+----------------------+
|  nixlObjEngineImpl   |  (abstract base)
+----------------------+
           |
           v
+------------------------------------+
| DefaultObjEngineImpl               |
| uses: awsS3Client                  |
+------------------------------------+
      |
      |--+------------------------------+
      |  | S3CrtObjEngineImpl           |
      |  | uses: awsS3CrtClient         |
      |  +------------------------------+
      |
      +-->+------------------------------+
          | S3AccelObjEngineImpl         |
          | uses: awsS3AccelClient       |
          +------------------------------+
```

**Key Points:**
- `nixlObjEngine` is the public interface that clients use
- `nixlObjEngineImpl` is the abstract base class for all engine implementations
- Concrete implementations inherit from `nixlObjEngineImpl` (or its subclasses) and override specific methods
- The public interface remains stable while implementations can evolve independently

#### Supported Memory Types

Each engine implementation defines its own supported memory segment types via `getSupportedMems()`:

| Engine | Supported Memory Types | Description |
|--------|----------------------|-------------|
| `DefaultObjEngineImpl` | `OBJ_SEG`, `DRAM_SEG` | Standard S3 client - CPU memory only |
| `S3CrtObjEngineImpl` | `OBJ_SEG`, `DRAM_SEG` | S3 CRT client - CPU memory only |
| `S3AccelObjEngineImpl` | `OBJ_SEG`, `DRAM_SEG` | S3 Accelerated base - CPU memory by default |
| Vendor engines | `OBJ_SEG`, `DRAM_SEG`, `VRAM_SEG` | Vendor-specific - override to add GPU support |

**Important:** Vendor engines that support GPU-direct transfers should override `getSupportedMems()` to include `VRAM_SEG`. The base `S3AccelObjEngineImpl` does not include `VRAM_SEG` by default - each vendor must explicitly expose this capability.

### Adding a Vendor Implementation

> **⚠️ Important: Conditional Compilation for S3 Accelerated Engines**
>
> The S3 Accelerated path (`s3_accel`) and any vendor implementations under it require the `cuobjclient-13.1` library. When adding new extensions to `s3_accel`:
>
> 1. **Protect includes** with `#if defined HAVE_CUOBJ_CLIENT`:
>    ```cpp
>    #if defined HAVE_CUOBJ_CLIENT
>    #include "s3_accel/engine_impl.h"
>    #include "s3_accel/vendor_name/engine_impl.h"
>    #endif
>    ```
>
> 2. **Protect instantiation code** in factory functions:
>    ```cpp
>    #if defined HAVE_CUOBJ_CLIENT
>    if (isAcceleratedRequested(init_params->customParams)) {
>        return std::make_unique<S3AccelObjEngineImpl>(init_params);
>    }
>    #endif
>    ```
>
> 3. **Add sources conditionally** in `meson.build`:
>    ```python
>    if cuobj_dep.found()
>        obj_sources += [
>            's3_accel/vendor_name/client.cpp',
>            's3_accel/vendor_name/engine_impl.cpp',
>        ]
>    endif
>    ```
>
> This ensures the plugin builds correctly both with and without the `cuobjclient` library available.

Follow these steps to add a vendor-specific client and engine:

#### 1. Create Vendor Client

Create a new directory under `s3_accel/` for your vendor (e.g., `s3_accel/vendor_name/`).

**client.h**
```cpp
#pragma once
#include "s3_accel/client.h"

class awsVendorClient : public awsS3AccelClient {
public:
    awsVendorClient(nixl_b_params_t *custom_params,
                    std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);
    virtual ~awsVendorClient() = default;

    // Override methods for vendor-specific behavior if needed
};
```

**client.cpp**
```cpp
#include "client.h"
#include "common/nixl_log.h"

awsVendorClient::awsVendorClient(nixl_b_params_t *custom_params,
                                 std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3AccelClient(custom_params, executor) {
    NIXL_INFO << "Initialized Vendor-specific Object Client";
}
```

#### 2. Create Vendor Engine Implementation

**engine_impl.h**
```cpp
#pragma once
#include "s3_accel/engine_impl.h"

class VendorObjEngineImpl : public S3AccelObjEngineImpl {
public:
    explicit VendorObjEngineImpl(const nixlBackendInitParams *init_params);
    VendorObjEngineImpl(const nixlBackendInitParams *init_params,
                        std::shared_ptr<iS3Client> s3_client);

    // Add VRAM_SEG support for GPU-direct transfers
    nixl_mem_list_t getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    }

    // Override engine methods for vendor-specific behavior if needed
    nixl_status_t registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) override;
};
```

**engine_impl.cpp**
```cpp
#include "engine_impl.h"
#include "s3_accel/vendor_name/client.h"
#include "common/nixl_log.h"

VendorObjEngineImpl::VendorObjEngineImpl(const nixlBackendInitParams *init_params)
    : S3AccelObjEngineImpl(init_params) {
    // Create vendor-specific S3 client and assign to s3Client_
    s3Client_ = std::make_shared<awsVendorClient>(init_params->customParams, executor_);
    NIXL_INFO << "Object storage backend initialized with Vendor client";
}

VendorObjEngineImpl::VendorObjEngineImpl(const nixlBackendInitParams *init_params,
                                         std::shared_ptr<iS3Client> s3_client)
    : S3AccelObjEngineImpl(init_params, s3_client) {
    if (!s3Client_) {
        s3Client_ = std::make_shared<awsVendorClient>(init_params->customParams, executor_);
    }
}

nixl_status_t
VendorObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                const nixl_mem_t &nixl_mem,
                                nixlBackendMD *&out) {
    NIXL_INFO << "Vendor-specific registerMem called";
    return S3AccelObjEngineImpl::registerMem(mem, nixl_mem, out);
}
```

#### 3. Add Selection Logic

Add a helper function in `engine_utils.h` (located at `src/utils/object/engine_utils.h`):

```cpp
inline bool
isVendorRequested(nixl_b_params_t *custom_params) {
    if (!isAcceleratedRequested(custom_params)) return false;
    auto type_it = custom_params->find("type");
    return type_it != custom_params->end() && type_it->second == "vendor_name";
}
```

Update `obj_backend.cpp` to include your engine in the factory function with proper conditional compilation:

```cpp
#if defined HAVE_CUOBJ_CLIENT
#include "s3_accel/engine_impl.h"
#include "s3_accel/vendor_name/engine_impl.h"
#endif

std::unique_ptr<nixlObjEngineImpl>
createObjEngineImpl(const nixlBackendInitParams *init_params) {
#if defined HAVE_CUOBJ_CLIENT
    // Check for vendor-specific engine first
    if (isVendorRequested(init_params->customParams)) {
        return std::make_unique<VendorObjEngineImpl>(init_params);
    }

    // Check for S3 Accelerated engine
    if (isAcceleratedRequested(init_params->customParams)) {
        return std::make_unique<S3AccelObjEngineImpl>(init_params);
    }
#endif

    // Check for S3 CRT engine
    size_t crt_min_limit = getCrtMinLimit(init_params->customParams);
    if (crt_min_limit > 0) {
        return std::make_unique<S3CrtObjEngineImpl>(init_params);
    }

    // Default to standard S3 engine
    return std::make_unique<DefaultObjEngineImpl>(init_params);
}
```

**Note**: Don't forget to also update the second overload of `createObjEngineImpl` that takes client pointers as parameters with the same conditional compilation guards.

#### 4. Update Build Configuration

Add your sources to `meson.build` within the conditional `cuobj_dep` block:

```python
if cuobj_dep.found()
    message('Found CUObjClient Library. Enabling S3 Accelerated engines')
    obj_sources += [
        's3_accel/client.cpp',
        's3_accel/client.h',
        's3_accel/engine_impl.cpp',
        's3_accel/engine_impl.h',
        # Add your vendor sources here:
        's3_accel/vendor_name/client.cpp',
        's3_accel/vendor_name/client.h',
        's3_accel/vendor_name/engine_impl.cpp',
        's3_accel/vendor_name/engine_impl.h',
    ]
    plugin_deps += [ cuobj_dep ]
else
    message('Could not find CUObjClient Library. Skipping S3 Accelerated engines')
endif
```

This ensures your vendor implementation is only compiled when the `cuobjclient` library is available.

#### 5. Usage

```cpp
nixl_b_params_t params = {
    {"bucket", "my-bucket"},
    {"accelerated", "true"},
    {"type", "vendor_name"}
};
agent.createBackend("obj", params);
```

### Engine Method Overrides

Common methods to override for vendor-specific behavior:

- `getSupportedMems()` - Memory segment types supported by this engine (override to add `VRAM_SEG` for GPU-direct transfers)
- `registerMem()` - Custom memory registration logic
- `deregisterMem()` - Custom cleanup logic
- `queryMem()` - Vendor-specific object existence checks
- `prepXfer()` / `postXfer()` - Custom transfer logic
- `getClient()` - Return vendor-specific client

**Note:** Vendor engines must explicitly override `getSupportedMems()` to include `VRAM_SEG` if they support GPU-direct transfers. Example:

```cpp
nixl_mem_list_t
getSupportedMems() const override {
    return {DRAM_SEG, OBJ_SEG, VRAM_SEG};
}
```
