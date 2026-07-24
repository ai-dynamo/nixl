---
title: C++ API Reference
description: Complete C++ API reference for NIXL data transfers.
---

This is the C++ API reference for the NVIDIA Inference Xfer Library (NIXL). C++ is the native API -- all other language bindings are built on top of it. For Python bindings, see the [Python API Reference](./python-api). For Rust bindings, see the [Rust API Reference](./rust-api).

Behavior shared by all bindings is documented in [Northbound API Semantics](./northbound-api).

To use the NIXL C++ API, include a single header:

```cpp
#include "nixl.h"
```

All public types, enumerations, and the `nixlAgent` class are available through this header.

## Types, Enums and Defines

This section documents the enumerations, type aliases, configuration structs, descriptor classes, constants, and handle types used throughout the NIXL C++ API. These are defined in `nixl_types.h`, `nixl_params.h`, and `nixl_descriptors.h`.

### Enumerations

### nixl_mem_t

Memory segment types supported by NIXL. Each type represents a different class of memory or storage that NIXL can register and transfer.

| Value | Description |
|-------|-------------|
| `DRAM_SEG` | Standard host memory (CPU DRAM) |
| `VRAM_SEG` | GPU high-bandwidth memory (HBM/VRAM) |
| `BLK_SEG` | Block-level storage devices |
| `OBJ_SEG` | Distributed object stores (S3, Azure Blob) |
| `FILE_SEG` | Local and remote file systems |

```cpp
nixl_reg_dlist_t descs(VRAM_SEG);  // Create descriptor list for GPU memory
```

### nixl_xfer_op_t

Transfer operation direction.

| Value | Description |
|-------|-------------|
| `NIXL_READ` | Read data from the remote side into local buffers |
| `NIXL_WRITE` | Write data from local buffers to the remote side |

```cpp
agent.createXferReq(NIXL_WRITE, local_descs, remote_descs, "target", req);
```

### nixl_status_t

Status codes and error values returned by NIXL API methods.

| Value | Code | Description |
|-------|------|-------------|
| `NIXL_IN_PROG` | 1 | Transfer is in progress |
| `NIXL_SUCCESS` | 0 | Operation completed successfully |
| `NIXL_ERR_NOT_POSTED` | -1 | Transfer was not posted |
| `NIXL_ERR_INVALID_PARAM` | -2 | Invalid parameter provided |
| `NIXL_ERR_BACKEND` | -3 | Backend-level error |
| `NIXL_ERR_NOT_FOUND` | -4 | Requested resource not found |
| `NIXL_ERR_MISMATCH` | -5 | Mismatched parameters or types |
| `NIXL_ERR_NOT_ALLOWED` | -6 | Operation not allowed in current state |
| `NIXL_ERR_REPOST_ACTIVE` | -7 | Attempting to repost an active transfer |
| `NIXL_ERR_UNKNOWN` | -8 | Unknown error |
| `NIXL_ERR_NOT_SUPPORTED` | -9 | Operation not supported by backend |
| `NIXL_ERR_REMOTE_DISCONNECT` | -10 | Remote agent disconnected |
| `NIXL_ERR_CANCELED` | -11 | Transfer was canceled |
| `NIXL_ERR_NO_TELEMETRY` | -12 | Telemetry data not available |

```cpp
nixl_status_t status = agent.getXferStatus(req);
if (status == NIXL_SUCCESS) { /* transfer complete */ }
```

### nixl_thread_sync_t

Thread synchronization modes for multi-threaded agent usage.

| Value | Description |
|-------|-------------|
| `NIXL_THREAD_SYNC_NONE` | No synchronization (default). Single-threaded usage only. |
| `NIXL_THREAD_SYNC_STRICT` | Full mutual exclusion on all operations. |
| `NIXL_THREAD_SYNC_RW` | Reader-writer lock: concurrent reads, exclusive writes. |

<Note>
`NIXL_THREAD_SYNC_DEFAULT` is an alias for `NIXL_THREAD_SYNC_NONE`. This is an `enum class`, so values must be qualified: `nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT`.
</Note>

```cpp
nixlAgentConfig cfg;
cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
```

### nixl_cost_t

Cost estimation method identifiers.

| Value | Description |
|-------|-------------|
| `ANALYTICAL_BACKEND` | Analytical backend cost estimate (value 0) |

<Note>
This is an `enum class`, so the value must be qualified: `nixl_cost_t::ANALYTICAL_BACKEND`.
</Note>

### Type Aliases

| Alias | Underlying Type | Purpose |
|-------|----------------|---------|
| `nixl_backend_t` | `std::string` | Backend identifier string |
| `nixl_blob_t` | `std::string` | Binary data blob (supports embedded `\0`) |
| `nixl_mem_list_t` | `std::vector<nixl_mem_t>` | List of memory types |
| `nixl_b_params_t` | `std::unordered_map<std::string, std::string>` | Backend key-value parameters |
| `nixl_notifs_t` | `std::unordered_map<std::string, std::vector<nixl_blob_t>>` | Notification map (agent name to messages) |
| `nixl_opt_args_t` | `nixlAgentOptionalArgs` | Optional method arguments struct |
| `nixl_xfer_dlist_t` | `nixlDescList<nixlBasicDesc>` | Transfer descriptor list |
| `nixl_reg_dlist_t` | `nixlDescList<nixlBlobDesc>` | Registration descriptor list |
| `nixl_remote_dlist_t` | `nixlDescList<nixlRemoteDesc>` | Remote descriptor list |
| `nixl_local_dlist_t` | `nixlDescList<nixlBasicDesc>` | Local descriptor list |
| `nixl_query_resp_t` | `std::optional<nixl_b_params_t>` | Query response (empty if no data) |
| `nixl_xfer_telem_t` | `nixlXferTelemetry` | Transfer telemetry data |
| `nixlMemViewH` | `void*` | Memory view handle |

<Warning>
`nixl_blob_t` is a typedef for `std::string` but is intended to hold binary data (including embedded null bytes). Do not call `.c_str()` on it -- use `.data()` and `.size()` instead.
</Warning>

### Configuration Structs

### nixlAgentConfig

Per-agent configuration struct. All fields have defaults, so a default-constructed `nixlAgentConfig` is valid.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `useProgThread` | `bool` | `false` | Enable progress thread for asynchronous transfer advancement |
| `useListenThread` | `bool` | `false` | Enable listener thread for incoming peer connections |
| `listenPort` | `int` | `0` | Port for the listener thread (0 = system-assigned) |
| `syncMode` | `nixl_thread_sync_t` | `NIXL_THREAD_SYNC_NONE` | Thread synchronization mode for multi-threaded usage |
| `captureTelemetry` | `bool` | `false` | Enable telemetry capture regardless of environment variables |
| `pthrDelay` | `uint64_t` | `0` | Progress thread delay between iterations (microseconds) |
| `lthrDelay` | `uint64_t` | `100000` | Listener thread sleep duration (microseconds) |
| `etcdWatchTimeout` | `std::chrono::microseconds` | `5000000` (5 seconds) | Timeout for etcd watch operations |

```cpp
nixlAgentConfig cfg;
cfg.useProgThread = true;
cfg.useListenThread = true;
cfg.listenPort = 5555;
nixlAgent agent("my_agent", cfg);
```

<Note>
`nixlAgentConfig` also provides a parameterized constructor that accepts all fields as arguments. Using the default constructor and setting fields individually is recommended for clarity.
</Note>

### nixlAgentOptionalArgs

Optional arguments struct passed to many agent methods via `const nixl_opt_args_t*`. Pass `nullptr` to use defaults.

**Current fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backends` | `std::vector<nixlBackendH*>` | empty | Limit operation to specified backends |
| `notif` | `std::optional<nixl_blob_t>` | empty | Notification message for transfer operations. If set, enables notification even for empty strings. |
| `ipAddr` | `std::string` | empty | IP address for peer-to-peer metadata operations |
| `port` | `int` | `8888` | Port for metadata operations (requires `ipAddr` to be set) |
| `metadataLabel` | `std::string` | empty | Label for etcd metadata scoping |
| `customParam` | `nixl_blob_t` | empty | Custom backend parameter string |

<Warning title="Deprecated Fields">
The following fields are kept for backward compatibility but should not be used in new code:

- `notifMsg` (`nixl_blob_t`) -- Use `notif` instead. The `notif` optional provides cleaner notification semantics.
- `hasNotif` (`bool`, default `false`) -- Use `notif` instead. Setting `notif` to any value (including empty string) enables notifications.
- `skipDescMerge` (`bool`, default `false`) -- Deprecated. Descriptor merging behavior is handled internally.
- `includeConnInfo` (`bool`, default `false`) -- Deprecated. Connection info inclusion is handled via `backends` and descriptor list contents.
</Warning>

```cpp
nixl_opt_args_t args;
args.notif = "transfer_complete";  // Enable notification with message
args.backends.push_back(backend);  // Limit to specific backend
agent.postXferReq(req, &args);
```

### nixlXferTelemetry

Transfer telemetry data returned by `getXferTelemetry()`.

| Field | Type | Description |
|-------|------|-------------|
| `startTime` | `chrono_point_t` | Time point when the transfer was posted |
| `postDuration` | `chrono_period_us_t` | Duration of the post operation (microseconds) |
| `xferDuration` | `chrono_period_us_t` | Duration of the entire transfer (microseconds) |
| `totalBytes` | `size_t` | Total bytes transferred in the request |
| `descCount` | `size_t` | Number of descriptors in the transfer (reflects merging if any) |

<Note>
`chrono_point_t` is `std::chrono::steady_clock::time_point` and `chrono_period_us_t` is `std::chrono::microseconds`. If `getXferStatus()` is called late after completion, `xferDuration` may overestimate the actual transfer time.
</Note>

### Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `NIXL_INIT_AGENT` | `#define` | `""` (empty string) | Indicates local agent in `prepXferDlist` |
| `default_comm_port` | `constexpr int` | `8888` | Default port for peer-to-peer communication |
| `default_metadata_label` | `const std::string` | -- | Default etcd key label for full metadata |
| `default_partial_metadata_label` | `const std::string` | -- | Default etcd key label for partial metadata |

### Handle Types

The following are opaque handle types returned by NIXL methods. They should not be constructed directly.

| Handle | Description |
|--------|-------------|
| `nixlBackendH*` | Backend handle returned by `createBackend()` |
| `nixlDlistH*` | Prepared descriptor list handle returned by `prepXferDlist()` |
| `nixlXferReqH*` | Transfer request handle returned by `makeXferReq()` / `createXferReq()` |
| `nixlMemViewH` | Memory view handle returned by `prepMemView()` (typedef for `void*`) |

## NIXL Descriptors

NIXL uses a hierarchy of descriptor classes to represent memory regions for registration and transfer.

<div className="diagram-light">
<Frame caption="Descriptor class hierarchy">
<img src="../figures/data-flow/nixl_desc_hierarchy_light.svg" alt="Descriptor class hierarchy" style={{width: '100%'}} />
</Frame>
</div>
<div className="diagram-dark">
<Frame caption="Descriptor class hierarchy">
<img src="../figures/data-flow/nixl_desc_hierarchy_dark.svg" alt="Descriptor class hierarchy" style={{width: '100%'}} />
</Frame>
</div>

### nixlBasicDesc

Base descriptor for a single contiguous memory region.

| Field | Type | Description |
|-------|------|-------------|
| `addr` | `uintptr_t` | Start address of the buffer |
| `len` | `size_t` | Length of the buffer in bytes |
| `devId` | `uint64_t` | Device ID, block ID, or file ID |

```cpp
nixlBasicDesc desc(reinterpret_cast<uintptr_t>(buf), 256, 0);
```

### nixlBlobDesc

Extends `nixlBasicDesc` with metadata blob. Used for memory registration.

| Field | Type | Description |
|-------|------|-------------|
| *(inherits `addr`, `len`, `devId`)* | | |
| `metaInfo` | `nixl_blob_t` | Metadata blob (e.g., file path for FILE_SEG) |

```cpp
nixlBlobDesc desc(reinterpret_cast<uintptr_t>(buf), 256, 0, "metadata");
```

### nixlRemoteDesc

Extends `nixlBasicDesc` with remote agent name. Used for memory view operations.

| Field | Type | Description |
|-------|------|-------------|
| *(inherits `addr`, `len`, `devId`)* | | |
| `remoteAgent` | `std::string` | Name of the remote agent owning this buffer |

```cpp
nixlRemoteDesc desc(addr, 256, 0, "remote_agent");
```

### nixlDescList\<T\>

Template container for descriptor lists. Parameterized by descriptor type.

| Method | Returns | Description |
|--------|---------|-------------|
| `nixlDescList(nixl_mem_t type, int init_size = 0)` | -- | Constructor with memory type and optional initial capacity |
| `addDesc(const T& desc)` | `void` | Add a descriptor to the list |
| `descCount()` | `int` | Number of descriptors in the list |
| `isEmpty()` | `bool` | Check if the list is empty |
| `getType()` | `nixl_mem_t` | Get the memory type of this list |
| `operator[](size_t index)` | `T&` / `const T&` | Access descriptor by index |
| `begin()` / `end()` | iterator | Standard iteration support |
| `remDesc(int index)` | `void` | Remove descriptor at index (may throw `std::out_of_range`) |
| `trim()` | `nixlDescList<nixlBasicDesc>` | Convert to basic descriptor list (strips metadata) |
| `getIndex(const nixlBasicDesc& query)` | `int` | Find index of matching descriptor (negative on error) |
| `clear()` | `void` | Remove all descriptors |
| `resize(size_t count)` | `void` | Resize the descriptor list |

```cpp
nixl_reg_dlist_t descs(DRAM_SEG);
descs.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buf), 256, 0, ""));
// descs.descCount() == 1
```


## Initialization and Configuration

<Tip>
For a complete workflow example, see [Quick Start -- Agent Initialization](../getting-started/quick-start#agent-initialization).
</Tip>

### nixlAgent

Constructor for the Transfer Agent. Each agent represents one endpoint in a data transfer and manages backends, memory registrations, metadata, and transfer operations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `const std::string&` | Unique name for this agent. Used to identify the agent in metadata exchange and transfers. |
| `cfg` | `const nixlAgentConfig&` | Agent configuration (threading, telemetry, etc.) |
| **Returns** | `nixlAgent` | Constructed agent object |

<Note>
`nixlAgent` is non-movable and non-copyable. It must be created in place or managed via a pointer. The destructor handles all resource cleanup.
</Note>

```cpp
nixlAgentConfig cfg;
cfg.useProgThread = true;
nixlAgent agent("my_agent", cfg);
```

### ~nixlAgent

Destructor. Cleans up all backends, registered memory, metadata, and internal resources.

```cpp
{
    nixlAgent agent("my_agent", cfg);
    // ... use agent ...
}  // destructor called here
```

### getAvailPlugins

Discover the available backend plug-ins found in the plug-in search paths.

| Parameter | Type | Description |
|-----------|------|-------------|
| `plugins` | `std::vector<nixl_backend_t>&` | [out] Populated with the names of available backend plug-ins |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
std::vector<nixl_backend_t> plugins;
agent.getAvailPlugins(plugins);
// plugins might contain: {"UCX", "GDS", "POSIX", ...}
```

## Backend Management

<Tip>
For a complete workflow example, see [Quick Start -- Backend Creation](../getting-started/quick-start#backend-creation).
</Tip>

### getPluginParams

Get the supported memory types and initialization parameters (with defaults) for a backend plug-in, before creating an instance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | `const nixl_backend_t&` | Plugin backend type name (e.g., `"UCX"`) |
| `mems` | `nixl_mem_list_t&` | [out] Memory types supported by this plug-in |
| `params` | `nixl_b_params_t&` | [out] Init parameters and their default values |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_mem_list_t mems;
nixl_b_params_t params;
agent.getPluginParams("UCX", mems, params);
```

### getBackendParams

Get the parameters of an already-instantiated backend. This returns a comprehensive list including defaults that were not explicitly specified during creation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | `const nixlBackendH*` | Backend handle obtained from `createBackend()` |
| `mems` | `nixl_mem_list_t&` | [out] Memory types supported by this backend |
| `params` | `nixl_b_params_t&` | [out] Parameters and their current values |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_mem_list_t mems;
nixl_b_params_t params;
agent.getBackendParams(backend, mems, params);
```

### createBackend

Instantiate a backend engine with the given parameters.

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | `const nixl_backend_t&` | Backend type name (e.g., `"UCX"`, `"GDS"`) |
| `params` | `const nixl_b_params_t&` | Backend-specific initialization parameters |
| `backend` | `nixlBackendH*&` | [out] Backend handle for subsequent operations |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
Multiple backends can be created on the same agent. NIXL automatically selects the best backend for each transfer based on the source and destination memory types and the backends available on both agents.
</Note>

```cpp
nixlBackendH* backend;
nixl_b_params_t params;
agent.createBackend("UCX", params, backend);
```

## Memory Registration

<Tip>
For a complete workflow example, see [Quick Start -- Memory Registration](../getting-started/quick-start#memory-registration).
</Tip>

### registerMem

Register memory or storage segments with NIXL. Registration creates internal data structures that backends use to track memory regions and generate metadata for remote access.

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `const nixl_reg_dlist_t&` | Descriptor list of buffers to register |
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `backends` is set, registration is limited to those backends. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
If no backend hints are provided, NIXL auto-selects all compatible backends for the given memory type. Memory must be registered before metadata exchange -- metadata exchanged before registration will not include the segment information needed for transfers.
</Note>

```cpp
nixl_reg_dlist_t descs(DRAM_SEG);
descs.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buf), 256, 0, ""));
agent.registerMem(descs);
```

### deregisterMem

Deregister previously registered memory or storage segments from NIXL.

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `const nixl_reg_dlist_t&` | Descriptor list of buffers to deregister (must match registration descriptors) |
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `backends` is set, deregistration is limited to those backends. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
agent.deregisterMem(descs);
```

### queryMem

Query information about registered memory or storage segments from a specific backend.

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `const nixl_reg_dlist_t&` | Descriptor list of buffers to query |
| `resp` | `std::vector<nixl_query_resp_t>&` | [out] Query response for each descriptor. Use `.has_value()` to check validity and `.value()` to access the key-value dictionary. |
| `extra_params` | `const nixl_opt_args_t*` | Required. The target backend should be specified via `extra_params->backends`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
std::vector<nixl_query_resp_t> resp;
nixl_opt_args_t args;
args.backends.push_back(backend);
agent.queryMem(descs, resp, &args);
```

### makeConnection

Proactively establish a connection to a remote agent, instead of deferring it to the first transfer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `const std::string&` | Name of the remote agent to connect to |
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `backends` is set, connection is limited to those backends. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
Connections are normally established lazily on the first transfer. Use `makeConnection()` to pre-establish connections and avoid first-transfer latency.
</Note>

```cpp
agent.makeConnection("remote_agent");
```

## Transfer Preparation

<Tip>
For a complete workflow example, see [Quick Start -- Creating and Executing Transfers](../getting-started/quick-start#creating-and-executing-transfers).
</Tip>

### prepXferDlist (4-parameter)

Prepare a descriptor list for use in transfer requests. Elements from the prepared list can later be selected by index in `makeXferReq()`. This overload accepts an explicit agent name for the descriptor side.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_name` | `const std::string&` | Agent name for the prepared list. Use `NIXL_INIT_AGENT` (empty string) for local descriptors, the remote agent's name for remote descriptors, or the local agent's own name for loopback transfers. |
| `descs` | `const nixl_xfer_dlist_t&` | Descriptor list to prepare |
| `dlist_hndl` | `nixlDlistH*&` | [out] Prepared descriptor list handle |
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `backends` is set, preparation is limited to those backends. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
Preparation succeeds if at least one backend can handle all elements in the descriptor list. Use this method when you need fine-grained control over which descriptors participate in each transfer via index selection.
</Note>

```cpp
nixlDlistH* local_hndl;
agent.prepXferDlist(NIXL_INIT_AGENT, local_descs, local_hndl);

nixlDlistH* remote_hndl;
agent.prepXferDlist("remote_agent", remote_descs, remote_hndl);
```

### prepXferDlist (3-parameter)

Convenience overload that prepares a local descriptor list. Equivalent to calling the 4-parameter overload with `NIXL_INIT_AGENT` as the agent name.

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `const nixl_xfer_dlist_t&` | Descriptor list to prepare |
| `dlist_hndl` | `nixlDlistH*&` | [out] Prepared descriptor list handle |
| `extra_params` | `const nixl_opt_args_t*` | Optional. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixlDlistH* local_hndl;
agent.prepXferDlist(local_descs, local_hndl);
```

### makeXferReq

Create a transfer request by selecting indices from already-prepared descriptor list handles. NIXL automatically determines the backend for the transfer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `operation` | `const nixl_xfer_op_t&` | Transfer direction (`NIXL_READ` or `NIXL_WRITE`) |
| `local_side` | `const nixlDlistH*` | Local prepared descriptor list handle |
| `local_indices` | `const std::vector<int>&` | Indices into the local descriptor list |
| `remote_side` | `const nixlDlistH*` | Remote (or loopback) prepared descriptor list handle |
| `remote_indices` | `const std::vector<int>&` | Indices into the remote descriptor list |
| `req_hndl` | `nixlXferReqH*&` | [out] Transfer request handle |
| `extra_params` | `const nixl_opt_args_t*` | Optional. Use `backends` to limit backend selection; use `notif` to attach a notification message. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixlXferReqH* req;
std::vector<int> indices = {0, 1, 2};
agent.makeXferReq(NIXL_WRITE, local_hndl, indices,
                  remote_hndl, indices, req);
```

### createXferReq

Combined API that creates a transfer request directly from two descriptor lists. Internally prepares both sides and creates the transfer handle. Equivalent to calling `prepXferDlist()` for each side followed by `makeXferReq()` with all indices.

| Parameter | Type | Description |
|-----------|------|-------------|
| `operation` | `const nixl_xfer_op_t&` | Transfer direction (`NIXL_READ` or `NIXL_WRITE`) |
| `local_descs` | `const nixl_xfer_dlist_t&` | Local descriptor list |
| `remote_descs` | `const nixl_xfer_dlist_t&` | Remote (or loopback) descriptor list |
| `remote_agent` | `const std::string&` | Remote (or self) agent name |
| `req_hndl` | `nixlXferReqH*&` | [out] Transfer request handle |
| `extra_params` | `const nixl_opt_args_t*` | Optional. Use `backends` to limit backend selection; use `notif` to attach a notification message. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
If the same descriptors are reused across multiple transfers, prefer `prepXferDlist()` + `makeXferReq()` to avoid repeated preparation overhead. `createXferReq()` is simpler for one-off transfers.
</Note>

```cpp
nixlXferReqH* req;
nixl_opt_args_t args;
args.notif = "done";
agent.createXferReq(NIXL_WRITE, local_descs, remote_descs,
                    "remote_agent", req, &args);
```

## Transfer Operations

<Tip>
For a complete workflow example, see [Quick Start -- Creating and Executing Transfers](../getting-started/quick-start#creating-and-executing-transfers).
</Tip>

### estimateXferCost

Estimate the cost (duration) of executing a transfer request before posting it.

| Parameter | Type | Description |
|-----------|------|-------------|
| `req_hndl` | `const nixlXferReqH*` | Transfer request handle |
| `duration` | `std::chrono::microseconds&` | [out] Estimated transfer duration |
| `err_margin` | `std::chrono::microseconds&` | [out] Estimated error margin |
| `method` | `nixl_cost_t&` | [out] Method used to compute the estimate |
| `extra_params` | `const nixl_opt_args_t*` | Optional. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
std::chrono::microseconds duration, margin;
nixl_cost_t method;
agent.estimateXferCost(req, duration, margin, method);
```

### postXferReq

Submit a transfer request, initiating the data transfer. The operation is non-blocking -- after posting, use `getXferStatus()` to poll for completion.

| Parameter | Type | Description |
|-----------|------|-------------|
| `req_hndl` | `nixlXferReqH*` | Transfer request handle from `makeXferReq()` or `createXferReq()` |
| `extra_params` | `const nixl_opt_args_t*` | Optional. Notification message can be provided or updated per re-post. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` if transfer completed inline, `NIXL_IN_PROG` if transfer is asynchronous, or error code |

<Note>
Small transfers may complete within the `postXferReq()` call itself, returning `NIXL_SUCCESS` immediately. For larger transfers, expect `NIXL_IN_PROG` and poll with `getXferStatus()`.
</Note>

```cpp
nixl_status_t status = agent.postXferReq(req);
if (status == NIXL_IN_PROG) {
    // Poll for completion
}
```

### getXferStatus

Check the status of a posted transfer request.

| Parameter | Type | Description |
|-----------|------|-------------|
| `req_hndl` | `nixlXferReqH*` | Transfer request handle after `postXferReq()` |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` when complete, `NIXL_IN_PROG` while running, or negative error code |

```cpp
nixl_status_t status;
do {
    status = agent.getXferStatus(req);
} while (status == NIXL_IN_PROG);
```

### getXferTelemetry

Retrieve telemetry data for a transfer request. Telemetry capture must be enabled via `nixlAgentConfig::captureTelemetry`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `req_hndl` | `const nixlXferReqH*` | Transfer request handle |
| `telemetry` | `nixl_xfer_telem_t&` | [out] Populated telemetry data |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_NO_TELEMETRY` if capture is not enabled |

```cpp
nixl_xfer_telem_t telem;
if (agent.getXferTelemetry(req, telem) == NIXL_SUCCESS) {
    auto us = telem.xferDuration.count();
}
```

### queryXferBackend

Query the backend chosen for a transfer request. Useful when you need to target the same backend for a notification via `genNotif()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `req_hndl` | `const nixlXferReqH*` | Transfer request handle |
| `backend` | `nixlBackendH*&` | [out] Backend handle chosen for this transfer |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixlBackendH* xfer_backend;
agent.queryXferBackend(req, xfer_backend);
```

### releaseXferReq

Release a transfer request handle. If the transfer is still active, it will be canceled (or an error returned if the transfer cannot be aborted).

| Parameter | Type | Description |
|-----------|------|-------------|
| `req_hndl` | `nixlXferReqH*` | Transfer request handle to release |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
agent.releaseXferReq(req);
```

### releasedDlistH

Release a prepared descriptor list handle.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dlist_hndl` | `nixlDlistH*` | Prepared descriptor list handle to release |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
The method name is `releasedDlistH` (with a trailing 'd') -- this is the canonical name from the source header.
</Note>

```cpp
agent.releasedDlistH(local_hndl);
agent.releasedDlistH(remote_hndl);
```

## Memory View

### prepMemView (remote)

Prepare a memory view handle for remote buffers. The handle can later be used for memory transfer operations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dlist` | `const nixl_remote_dlist_t&` | Descriptor list for the remote buffers |
| `mvh` | `nixlMemViewH&` | [out] Memory view handle |
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `backends` is set, limits backend selection. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_remote_dlist_t remote_descs(VRAM_SEG);
remote_descs.addDesc(nixlRemoteDesc(addr, size, 0, "remote_agent"));
nixlMemViewH mvh;
agent.prepMemView(remote_descs, mvh);
```

### prepMemView (local)

Prepare a memory view handle for local buffers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dlist` | `const nixl_local_dlist_t&` | Descriptor list for the local buffers |
| `mvh` | `nixlMemViewH&` | [out] Memory view handle |
| `extra_params` | `const nixl_opt_args_t*` | Optional. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_local_dlist_t local_descs(DRAM_SEG);
local_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(buf), 256, 0));
nixlMemViewH mvh;
agent.prepMemView(local_descs, mvh);
```

### releaseMemView

Release a memory view handle.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mvh` | `nixlMemViewH` | Memory view handle to release |
| **Returns** | `void` | |

```cpp
agent.releaseMemView(mvh);
```

## Notification Handling

### getNotifs

Retrieve pending notifications from remote agents. Entries are added to the input map (which may already contain entries) and released from the agent's internal queue.

| Parameter | Type | Description |
|-----------|------|-------------|
| `notif_map` | `nixl_notifs_t&` | Map from agent name to list of notification blobs. New entries are appended. |
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `backends` is set, only retrieves notifications from those backends. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
Notifications are consumed by this call -- they are removed from the agent's internal queue after being added to `notif_map`.
</Note>

```cpp
nixl_notifs_t notifs;
agent.getNotifs(notifs);
for (auto& [agent_name, msgs] : notifs) {
    for (auto& msg : msgs) {
        // Process notification from agent_name
    }
}
```

### genNotif

Generate and send a standalone notification to a remote agent. The remote agent's metadata must be available before calling this method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `const std::string&` | Name of the remote agent |
| `msg` | `const nixl_blob_t&` | Notification message to send |
| `extra_params` | `const nixl_opt_args_t*` | Optional. Use `backends` to specify which backend to use for sending. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
Standalone notifications sent via `genNotif()` are received by the remote agent through `getNotifs()`, alongside transfer-bound notifications. Use `queryXferBackend()` if you want to send the notification over the same backend as a specific transfer.
</Note>

```cpp
agent.genNotif("remote_agent", "control_message");
```

## Metadata -- Side Channel

<Tip>
For a complete workflow example, see [Quick Start -- Metadata Exchange](../getting-started/quick-start#metadata-exchange).
</Tip>

Side-channel metadata methods use a serialized blob that the application transports between agents using its own mechanism (shared memory, message queue, custom RPC, etc.).

### getLocalMD

Get this agent's full metadata as a serialized blob. The blob can be transported to other agents and loaded with `loadRemoteMD()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `str` | `nixl_blob_t&` | [out] Serialized metadata blob |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_blob_t metadata;
agent.getLocalMD(metadata);
// Transport metadata to remote agent via your own mechanism
```

### getLocalPartialMD

Get a partial metadata blob containing only the specified descriptors and optionally backend connection information.

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `const nixl_reg_dlist_t&` | Descriptor list to include. If empty, only backend connection info is included. |
| `str` | `nixl_blob_t&` | [out] Serialized partial metadata blob |
| `extra_params` | `const nixl_opt_args_t*` | Optional. Use `includeConnInfo` to add connection info for backends supporting the memory type. Use `backends` to filter which backends are included. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
When `descs` is empty, only backends' connection info is included in the metadata, regardless of the value of `extra_params->includeConnInfo`.
</Note>

```cpp
nixl_blob_t partial_md;
agent.getLocalPartialMD(descs, partial_md);
```

### loadRemoteMD

Load and unpack a remote agent's metadata blob. After loading, the local agent can initiate transfers toward the remote agent.

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_metadata` | `const nixl_blob_t&` | Serialized metadata blob received from the remote agent |
| `agent_name` | `std::string&` | [out] Agent name extracted from the loaded metadata |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
std::string remote_name;
agent.loadRemoteMD(received_metadata, remote_name);
// Now can transfer to/from remote_name
```

### invalidateRemoteMD

Invalidate the cached metadata for a remote agent. This disconnects from the remote agent and prevents further transfers to it.

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `const std::string&` | Name of the remote agent to invalidate |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
agent.invalidateRemoteMD("remote_agent");
```

## Metadata -- Direct Channel

<Tip>
For a complete workflow example, see [Quick Start -- Metadata Exchange](../getting-started/quick-start#metadata-exchange).
</Tip>

Direct-channel methods send and fetch metadata over peer-to-peer TCP sockets or an etcd metadata server, without the application needing to transport blobs manually.

### sendLocalMD

Send this agent's full metadata to a remote peer or the etcd metadata server.

| Parameter | Type | Description |
|-----------|------|-------------|
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `ipAddr` is set, sends to the specified peer (peer-to-peer mode). If `ipAddr` is not set, sends to the etcd metadata server. `port` defaults to `default_comm_port` (8888). Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
// Send to ETCD (no IP specified)
agent.sendLocalMD();

// Send to a specific peer
nixl_opt_args_t args;
args.ipAddr = "10.0.0.2";
args.port = 5555;
agent.sendLocalMD(&args);
```

### sendLocalPartialMD

Send partial metadata to a remote peer or the etcd metadata server.

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `const nixl_reg_dlist_t&` | Descriptor list to include. If empty, only backend connection info is sent. |
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `ipAddr` is set, sends to a single peer. If `ipAddr` is not set, sends to the etcd metadata server. `metadataLabel` is required when sending to etcd and ignored for peer-to-peer. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

<Note>
When sending partial metadata to etcd, `extra_params->metadataLabel` is required to scope the metadata under a specific key label. When sending to a peer, the label is ignored.
</Note>

```cpp
nixl_opt_args_t args;
args.metadataLabel = "gpu_buffers";
agent.sendLocalPartialMD(descs, &args);
```

### fetchRemoteMD

Fetch a remote agent's metadata from a peer or the etcd metadata server, then unpack it internally.

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_name` | `const std::string` | Name of the remote agent to fetch metadata for |
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `ipAddr` is set, fetches from the specified peer (only full metadata). If `ipAddr` is not set, fetches from etcd. `metadataLabel` can specify partial metadata to fetch from etcd. `port` defaults to `default_comm_port`. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
// Fetch from ETCD
agent.fetchRemoteMD("remote_agent");

// Fetch from a specific peer
nixl_opt_args_t args;
args.ipAddr = "10.0.0.2";
agent.fetchRemoteMD("remote_agent", &args);
```

### invalidateLocalMD

Invalidate this agent's metadata on a remote peer or the etcd metadata server.

| Parameter | Type | Description |
|-----------|------|-------------|
| `extra_params` | `const nixl_opt_args_t*` | Optional. If `ipAddr` is set, invalidates metadata on the specified peer. If `ipAddr` is not set, invalidates all of this agent's labels from the etcd server. Default: `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
// Invalidate from ETCD (removes all labels)
agent.invalidateLocalMD();
```

### checkRemoteMD

Check if metadata is available for a remote agent. Optionally verify that specific descriptors are included.

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_name` | `const std::string` | Name of the remote agent to check |
| `descs` | `const nixl_xfer_dlist_t&` | Descriptor list to check for. Pass an empty list to check for any metadata from the agent. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` if metadata is available, `NIXL_ERR_NOT_FOUND` if not found |

```cpp
nixl_xfer_dlist_t empty(DRAM_SEG);
if (agent.checkRemoteMD("remote_agent", empty) == NIXL_SUCCESS) {
    // Remote metadata is available
}
```
