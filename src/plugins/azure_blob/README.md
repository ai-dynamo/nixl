<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-FileCopyrightText: Copyright (c) 2026 Microsoft Corporation.
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

# NIXL Azure Blob Storage Plugin

This backend provides Azure Blob storage using the Azure SDK for C++.

## Dependencies

This backend requires the azure-storage-blobs and azure-identity packages from the
azure-sdk-for-cpp repository. Example CLI to compile from sources:

```bash
git clone --depth 1 https://github.com/Azure/azure-sdk-for-cpp.git --branch azure-storage-blobs_12.15.0 && \
    cd azure-sdk-for-cpp/ && \
    mkdir build && cd build && \
    AZURE_SDK_DISABLE_AUTO_VCPKG=1 cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local -DDISABLE_AMQP=ON -DDISABLE_AZURE_CORE_OPENTELEMETRY=ON && \
    cmake --build . --target azure-storage-blobs azure-identity && \
    cmake --install sdk/core && \
    cmake --install sdk/storage/azure-storage-common && \
    cmake --install sdk/storage/azure-storage-blobs && \
    cmake --install sdk/identity
```

## Configuration

The Azure Blob Storage backend supports configuration through two mechanisms: backend parameter maps passed during backend creation, and environment variables for system-wide settings.
Backend parameters take precedence over environment variables.

### Backend Parameters

Backend parameters are passed as a key-value map (`nixl_b_params_t`) when creating the backend instance. The Azure Blob Storage backend accepts the following parameters:

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `account_url` | URL of Azure Storage account (e.g., `https://<account-name>.blob.core.windows.net`) | - | No* ** |
| `container_name` | Name of Azure Storage container | - | Yes* |
| `connection_string` | Azure Storage connection string (i.e., for testing with Azurite) | - | No* ** |
| `ca_bundle` | Path to a custom certificate bundle | - | No |

\* Each parameter falls back to a corresponding environment variable if not provided (see [Environment Variables](#environment-variables)).

\*\* Either `account_url` or `connection_string` must be provided for the backend to function. If `connection_string` is provided,
`account_url` will be ignored and connection string will be used directly to connect and authenticate to Azure Storage.
`connection_string` is primarily intended for local testing with [Azurite](https://learn.microsoft.com/azure/storage/common/storage-use-azurite). For production workloads, it is recommended to specify an `account_url`.

### Environment Variables

The following environment variables are supported for Azure Blob Storage configuration:

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_STORAGE_ACCOUNT_URL` | URL of Azure Storage account | `https://<account-name>.blob.core.windows.net` |
| `AZURE_STORAGE_CONTAINER_NAME` | Name of Azure Storage container | `my-container` |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Storage connection string | `DefaultEndpointsProtocol=https;AccountName=...` |
| `AZURE_CA_BUNDLE` | Path to a custom certificate bundle | `/path/to/cabundle.pem` |


### Configuration Priority

Configuration values are resolved in the following priority order (highest to lowest):

1. **Backend Parameters**: Values passed directly in the backend parameter map
2. **Environment Variables**: Azure Blob Storage environment variables


### Credentials

By default, the backend uses the Azure default credential chain to authenticate from your current environment using
[Microsoft Entra ID](https://learn.microsoft.com/azure/storage/blobs/authorize-access-azure-active-directory).
Refer to the [documentation](https://learn.microsoft.com/azure/developer/cpp/sdk/authentication/credential-chains#defaultazurecredential-overview)
to learn more about the default credential chain and how to configure it for your environment.

For local development and testing with [Azurite](https://learn.microsoft.com/azure/storage/common/storage-use-azurite), the
backend also supports connecting and authenticating instead using a connection string (see [Connecting to Azurite](#connecting-to-azurite)).

The backend does not currently support shared keys nor shared access signatures (SAS) for authentication.

### Configuration Examples

#### Minimal Configuration

```cpp
nixl_b_params_t params = {{"account_url", "https://myaccount.blob.core.windows.net"}, {"container_name", "my-container"}};
agent.createBackend("AZURE_BLOB", params);
```

#### Environment Variable Configuration

```bash
export AZURE_STORAGE_ACCOUNT_URL=https://myaccount.blob.core.windows.net
export AZURE_STORAGE_CONTAINER_NAME=my-container
```

```cpp
agent.createBackend("AZURE_BLOB", {});
```

#### Connecting to Azurite

To connect to [Azurite](https://learn.microsoft.com/azure/storage/common/storage-use-azurite) for local testing, first start Azurite:
```bash
# Example showing how to run Azurite using Docker
docker run --rm -p 10000:10000 mcr.microsoft.com/azure-storage/azurite azurite-blob --blobHost 0.0.0.0 --skipApiVersionCheck
```

And create an Azure Storage container (if not already created on the currently running Azurite instance):
```bash
# In a separate terminal, create an Azure Storage container in the running azurite instance
az storage container create \
  --name my-container \
  --connection-string 'DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;'
```

You can then connect by providing its connection string as a backend parameter:

```cpp
nixl_b_params_t params = {
    {"connection_string", "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"},
    {"container_name", "my-container"}
};
agent.createBackend("AZURE_BLOB", params);
```

Or as an environment variable:

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
export AZURE_STORAGE_CONTAINER_NAME=my-container
```

```cpp
agent.createBackend("AZURE_BLOB", {});
```

## Transfer Operations

The Azure Blob Storage backend supports read and write operations between local memory and Azure Storage blobs. Here are the key aspects of transfer operations:

### Device ID to Object Key Mapping

- Each blob in Azure Storage is identified by a unique blob name
- The backend maintains a mapping between device IDs (`devId`) and blob names
- When registering memory:
  - If `metaInfo` is provided in the blob descriptor, it is used as the blob name
  - Otherwise, the device ID is converted to a string and used as the blob name
- This mapping is used during transfer operations to locate the correct Azure Storage blob

### Read Operations

- Read operations support reading from a specific offset within a blob
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
