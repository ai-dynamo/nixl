---
title: Southbound API Reference
description: Complete reference for the NIXL Southbound API backend engine interface.
---

The Southbound API (SB API) is the standardized interface between NIXL's Transfer Agent and backend plug-ins. Every backend plug-in must implement this interface by inheriting from the `nixlBackendEngine` base class and overriding the required virtual methods.

This reference documents all 25 methods in the SB API: 13 pure virtual methods that every backend must implement, 7 conditionally required methods based on capability flags, and 5 optional methods with default implementations. For a step-by-step tutorial on building a plug-in, see [Building a Backend Plugin](./building-a-backend-plugin). For the user-facing API, see the [C++ API Reference](../api-reference/cpp-api).

## Base Class Hierarchy

The SB API is defined through three base classes and one parameter struct, all declared in the `src/api/cpp/backend/` headers.

### nixlBackendEngine

The primary class that every backend plug-in inherits from. Defined in `backend_engine.h`.

**Protected members** (accessible to child backends):

| Member | Type | Description |
|--------|------|-------------|
| `initErr` | `bool` | Set to `true` in the constructor if initialization fails. Defaults to `false`. |
| `localAgent` | `const std::string` | Name of the local agent that owns this backend instance. Read-only. |
| `enableTelemetry_` | `const bool` | Whether telemetry is enabled for this backend. Read-only. |

**Protected helper methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `setInitParam` | `nixl_status_t setInitParam(const std::string &key, const std::string &value)` | Store a custom parameter. Returns `NIXL_ERR_NOT_ALLOWED` if the key already exists. |
| `getInitParam` | `nixl_status_t getInitParam(const std::string &key, std::string &value) const` | Retrieve a custom parameter. Returns `NIXL_ERR_INVALID_PARAM` if the key is not found. |
| `addTelemetryEvent` | `void addTelemetryEvent(const std::string &event_name, uint64_t value)` | Record a telemetry event. No-op if telemetry is disabled or queue is full (max 1000 events). |

**Public helper methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `getInitErr` | `bool getInitErr() const noexcept` | Check whether construction failed. |
| `getType` | `const nixl_backend_t& getType() const noexcept` | Get the backend type string (e.g., `"UCX"`, `"POSIX"`). |
| `getCustomParams` | `const nixl_b_params_t& getCustomParams() const noexcept` | Get the key-value parameters map. |
| `getTelemetryEvents` | `std::vector<nixlTelemetryEvent> getTelemetryEvents()` | Move-return accumulated telemetry events (clears internal buffer). |

**Constructor:**

```cpp
explicit nixlBackendEngine(const nixlBackendInitParams *init_params);
```

The constructor initializes `backendType`, `customParams`, `localAgent`, and `enableTelemetry_` from the provided init params. The backend is **non-copyable and non-movable**.

**Destructor:**

```cpp
virtual ~nixlBackendEngine() = default;
```

Override the destructor to clean up any backend-specific resources (connections, registered memory, I/O queues).

### nixlBackendMD

Base class for backend metadata objects. Both registered memory metadata and remote identifier metadata inherit from this class. Defined in `backend_aux.h`.

**Protected field:**

| Field | Type | Description |
|-------|------|-------------|
| `isPrivateMD` | `bool` | Distinguishes private (registered memory) from public (remote identifier) metadata. |

**Constructor:**

```cpp
nixlBackendMD(bool isPrivate);
```

**Destructor:**

```cpp
virtual ~nixlBackendMD();
```

<Note>
The `nixlBackendMD` pointer is opaque to the NIXL agent. The agent stores it during `registerMem()` and passes it back during transfer operations. Your backend is responsible for casting it to your concrete metadata subclass.
</Note>

### nixlBackendReqH

Base class for transfer request handles. Backend-specific request state inherits from this class. Defined in `backend_aux.h`.

**Constructor:**

```cpp
nixlBackendReqH();
```

**Destructor:**

```cpp
virtual ~nixlBackendReqH();
```

The agent creates request handles via `prepXfer()` and destroys them via `releaseReqH()`. Your backend stores whatever state it needs for tracking in-progress transfers in your subclass.

### nixlBackendInitParams

Parameter container passed to the backend constructor. Not inherited -- this is a plain data class. Defined in `backend_aux.h`.

| Field | Type | Description |
|-------|------|-------------|
| `localAgent` | `std::string` | Name of the agent creating this backend. |
| `type` | `nixl_backend_t` | Backend type identifier (e.g., `"UCX"`, `"POSIX"`). |
| `customParams` | `nixl_b_params_t*` | Pointer to the key-value parameter map. |
| `enableProgTh` | `bool` | Whether the progress thread is enabled. |
| `pthrDelay` | `nixlTime::us_t` | Progress thread delay between iterations (microseconds). |
| `syncMode` | `nixl_thread_sync_t` | Thread synchronization mode. |
| `enableTelemetry_` | `bool` | Whether telemetry collection is enabled. |

## Capability Matrix

The following table shows which methods are required based on the capability flags your backend reports. Methods marked under "Always Required" must be implemented by every backend. Methods under capability columns are required only when that capability returns `true`.

| Method | Always Required | supportsLocal | supportsRemote | supportsNotif |
|--------|:-:|:-:|:-:|:-:|
| Constructor / Destructor | x | | | |
| `supportsRemote()` | x | | | |
| `supportsLocal()` | x | | | |
| `supportsNotif()` | x | | | |
| `getSupportedMems()` | x | | | |
| `registerMem()` | x | | | |
| `deregisterMem()` | x | | | |
| `connect()` | x | | | |
| `disconnect()` | x | | | |
| `unloadMD()` | x | | | |
| `prepXfer()` | x | | | |
| `postXfer()` | x | | | |
| `checkXfer()` | x | | | |
| `releaseReqH()` | x | | | |
| `loadLocalMD()` | | x | | |
| `getPublicData()` | | | x | |
| `getConnInfo()` | | | x | |
| `loadRemoteConnInfo()` | | | x | |
| `loadRemoteMD()` | | | x | |
| `getNotifs()` | | | | x |
| `genNotif()` | | | | x |
| `prepMemView()` (2 overloads) | | | | |
| `releaseMemView()` | | | | |
| `queryMem()` | | | | |
| `estimateXferCost()` | | | | |

The last 5 methods are truly optional and have default implementations that return `NIXL_ERR_NOT_SUPPORTED` or no-op.

<Note>
A network backend (e.g., UCX) should set `supportsRemote` and `supportsNotif` to `true`, and preferably `supportsLocal` as well so that another backend is not needed for local transfers. A storage backend (e.g., GDS, POSIX) should set `supportsLocal` to `true`; `supportsNotif` is optional.
</Note>

## Descriptor Types

The SB API uses several descriptor types for memory registration and transfer operations. These are defined across `nixl_descriptors.h` and `backend_aux.h`.

### nixlBlobDesc

Registration descriptor used by `registerMem()` and `loadRemoteMD()`. Extends `nixlBasicDesc` with a metadata blob.

| Field | Type | Description |
|-------|------|-------------|
| `addr` | `uintptr_t` | Start address of the buffer (or offset for file/object) |
| `len` | `size_t` | Length of the buffer in bytes |
| `devId` | `uint64_t` | Device ID, block ID, or file descriptor |
| `metaInfo` | `nixl_blob_t` | Optional metadata string (e.g., file path, bucket ID) |

### nixlMetaDesc

Transfer descriptor used within `nixl_meta_dlist_t`. Extends `nixlBasicDesc` with a backend metadata pointer.

| Field | Type | Description |
|-------|------|-------------|
| `addr` | `uintptr_t` | Start address of the buffer |
| `len` | `size_t` | Length of the buffer in bytes |
| `devId` | `uint64_t` | Device ID |
| `metadataP` | `nixlBackendMD*` | Pointer to backend metadata for this descriptor |

### nixlRemoteMetaDesc

Remote transfer descriptor. Extends `nixlMetaDesc` with a remote agent name.

| Field | Type | Description |
|-------|------|-------------|
| (inherits all from `nixlMetaDesc`) | | |
| `remoteAgent` | `std::string` | Name of the remote agent that owns this memory |

### Descriptor List Types

| Type Alias | Underlying Type | Used In |
|------------|----------------|---------|
| `nixl_reg_dlist_t` | `nixlDescList<nixlBlobDesc>` | `queryMem()` |
| `nixl_meta_dlist_t` | `nixlDescList<nixlMetaDesc>` | `prepXfer()`, `postXfer()`, `prepMemView()` (local) |
| `nixl_remote_meta_dlist_t` | `nixlDescList<nixlRemoteMetaDesc>` | `prepMemView()` (remote) |
| `notif_list_t` | `std::vector<std::pair<std::string, std::string>>` | `getNotifs()` -- pairs of (agent name, notification) |
| `nixl_opt_b_args_t` | `nixlBackendOptionalArgs` | `prepXfer()`, `postXfer()` |

### nixl_opt_b_args_t Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `notifMsg` | `nixl_blob_t` | empty | Notification message to send after transfer completes |
| `hasNotif` | `bool` | `false` | Whether a notification should be generated |
| `customParam` | `nixl_blob_t` | empty | Custom backend parameter string |

### Descriptor Field Meanings by Memory Type

| mem type | addr | len | devID | str (metaInfo) |
|----------|------|-----|-------|-----------------|
| DRAM | buffer address | buffer length | 0 (or region) | -- |
| VRAM | buffer address | buffer length | GPU ID | -- |
| BLK | block offset | block length | Volume ID | -- |
| FILE | offset in file | length (or 0) | file descriptor | Path (+ access mode) |
| OBJ | offset in object | length (or 0) | key | Extended key (+ bucket ID) |

## Capability Indicators

<Tip>
These four methods are pure virtual and must be implemented by every backend. They determine which additional methods the NIXL agent will call on your backend.
</Tip>

### supportsRemote

Indicates whether this backend supports data transfers across nodes (between different agents on different machines).

```cpp
virtual bool supportsRemote() const = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| **Returns** | `bool` | `true` if the backend supports remote (cross-node) operations |

```cpp
bool MyBackend::supportsRemote() const {
    return true;  // This is a network backend
}
```

### supportsLocal

Indicates whether this backend supports data transfers within the same node (local or loopback operations).

```cpp
virtual bool supportsLocal() const = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| **Returns** | `bool` | `true` if the backend supports local (within-node) operations |

```cpp
bool MyBackend::supportsLocal() const {
    return true;  // This backend handles local file I/O
}
```

### supportsNotif

Indicates whether this backend supports sending and receiving notifications. Related notification methods (`getNotifs`, `genNotif`) are not pure virtual and return errors if called on a backend that does not support notifications.

```cpp
virtual bool supportsNotif() const = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| **Returns** | `bool` | `true` if the backend supports notifications |

```cpp
bool MyBackend::supportsNotif() const {
    return false;  // Storage backends typically don't need notifications
}
```

### getSupportedMems

Returns the list of memory types this backend can handle. The agent uses this to route transfer requests to the appropriate backend.

```cpp
virtual nixl_mem_list_t getSupportedMems() const = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| **Returns** | `nixl_mem_list_t` | Vector of supported `nixl_mem_t` values (e.g., `DRAM_SEG`, `VRAM_SEG`, `FILE_SEG`) |

```cpp
nixl_mem_list_t MyBackend::getSupportedMems() const {
    return {DRAM_SEG, FILE_SEG};
}
```

<Note>
Based on these flags, the required methods change. A network backend should have `supportsRemote` and `supportsNotif` return `true` (and preferably `supportsLocal` as well). A storage backend should have `supportsLocal` return `true`.
</Note>

## Memory Management

<Tip>
For the user-facing memory registration workflow, see [C++ API Reference -- Memory Registration](../api-reference/cpp-api#registermem).
</Tip>

### registerMem

Registers a single contiguous memory region with the backend. The backend creates a metadata object for this region and returns it through the output parameter. The agent stores this metadata and passes it back during transfers.

```cpp
virtual nixl_status_t registerMem(const nixlBlobDesc &mem,
                                  const nixl_mem_t &nixl_mem,
                                  nixlBackendMD* &out) = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `mem` | `const nixlBlobDesc&` | Memory descriptor (address, length, device ID, optional metadata string) |
| `nixl_mem` | `const nixl_mem_t&` | Memory type (e.g., `DRAM_SEG`, `VRAM_SEG`, `FILE_SEG`) |
| `out` | `nixlBackendMD*&` | [out] Pointer to backend-created metadata object |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_status_t MyBackend::registerMem(const nixlBlobDesc &mem,
                                     const nixl_mem_t &nixl_mem,
                                     nixlBackendMD* &out) {
    auto *md = new MyBackendMD(mem, nixl_mem);
    out = md;
    return NIXL_SUCCESS;
}
```

### deregisterMem

Deregisters a previously registered memory region and frees the associated metadata object.

```cpp
virtual nixl_status_t deregisterMem(nixlBackendMD* meta) = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `meta` | `nixlBackendMD*` | Metadata object returned by `registerMem()` |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_status_t MyBackend::deregisterMem(nixlBackendMD* meta) {
    delete static_cast<MyBackendMD*>(meta);
    return NIXL_SUCCESS;
}
```

<Note>
The agent passes one contiguous memory descriptor at a time. The output pointer is stored by the agent and passed back during transfers. Your backend does not need to do bookkeeping for these pointers.
</Note>

## Connection Management

### connect

Initiates a connection to a remote agent (or to itself for loopback). Some backends require a self-connection for local operations. The agent may call `connect` proactively or defer it until the first transfer.

```cpp
virtual nixl_status_t connect(const std::string &remote_agent) = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `const std::string&` | Name of the remote agent to connect to |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_status_t MyBackend::connect(const std::string &remote_agent) {
    // Establish connection using loaded remote connection info
    return NIXL_SUCCESS;
}
```

### disconnect

Terminates a connection with a remote agent (or self for loopback). Called during metadata invalidation or agent destruction.

```cpp
virtual nixl_status_t disconnect(const std::string &remote_agent) = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `const std::string&` | Name of the remote agent to disconnect from |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_status_t MyBackend::disconnect(const std::string &remote_agent) {
    // Release connection resources
    return NIXL_SUCCESS;
}
```

### getConnInfo

Provides the serialized connection information that remote agents need to communicate with this backend. Called once after backend creation.

```cpp
virtual nixl_status_t getConnInfo(std::string &str) const;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `str` | `std::string&` | [out] Serialized connection data |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_BACKEND` (default) |

```cpp
nixl_status_t MyBackend::getConnInfo(std::string &str) const {
    str = serializeMyConnectionInfo();
    return NIXL_SUCCESS;
}
```

<Note>
Required only if `supportsRemote()` returns `true`. The default implementation returns `NIXL_ERR_BACKEND`.
</Note>

### loadRemoteConnInfo

Loads connection information received from a remote agent. This does not establish the connection -- it only stores the information for later use by `connect()`.

```cpp
virtual nixl_status_t loadRemoteConnInfo(const std::string &remote_agent,
                                         const std::string &remote_conn_info);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `const std::string&` | Name of the remote agent |
| `remote_conn_info` | `const std::string&` | Serialized connection data from the remote agent |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_BACKEND` (default) |

```cpp
nixl_status_t MyBackend::loadRemoteConnInfo(const std::string &remote_agent,
                                            const std::string &remote_conn_info) {
    connInfoMap_[remote_agent] = deserialize(remote_conn_info);
    return NIXL_SUCCESS;
}
```

<Note>
Required only if `supportsRemote()` returns `true`. The default implementation returns `NIXL_ERR_BACKEND`.
</Note>

## Metadata Management

### getPublicData

Serializes the remote identifier for a registered memory region. Remote agents use this serialized data to access this memory via `loadRemoteMD()`.

```cpp
virtual nixl_status_t getPublicData(const nixlBackendMD* meta,
                                    std::string &str) const;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `meta` | `const nixlBackendMD*` | Metadata object from `registerMem()` |
| `str` | `std::string&` | [out] Serialized remote identifier |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_BACKEND` (default) |

```cpp
nixl_status_t MyBackend::getPublicData(const nixlBackendMD* meta,
                                       std::string &str) const {
    auto *md = static_cast<const MyBackendMD*>(meta);
    str = md->serializeForRemote();
    return NIXL_SUCCESS;
}
```

<Note>
Required only if `supportsRemote()` returns `true`. The default implementation returns `NIXL_ERR_BACKEND`.
</Note>

### loadRemoteMD

Deserializes a remote memory identifier received from a remote agent. Creates a metadata object that the agent uses for remote transfers.

```cpp
virtual nixl_status_t loadRemoteMD(const nixlBlobDesc &input,
                                   const nixl_mem_t &nixl_mem,
                                   const std::string &remote_agent,
                                   nixlBackendMD* &output);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `const nixlBlobDesc&` | Descriptor with the remote memory info (address, length, device, serialized metadata in `metaInfo`) |
| `nixl_mem` | `const nixl_mem_t&` | Memory type of the remote region |
| `remote_agent` | `const std::string&` | Name of the remote agent |
| `output` | `nixlBackendMD*&` | [out] Pointer to the deserialized remote metadata object |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_BACKEND` (default) |

```cpp
nixl_status_t MyBackend::loadRemoteMD(const nixlBlobDesc &input,
                                      const nixl_mem_t &nixl_mem,
                                      const std::string &remote_agent,
                                      nixlBackendMD* &output) {
    auto *md = new MyRemoteMD(input, remote_agent);
    output = md;
    return NIXL_SUCCESS;
}
```

<Note>
Required only if `supportsRemote()` returns `true`. The default implementation returns `NIXL_ERR_BACKEND`.
</Note>

### loadLocalMD

Loads metadata for local (within-node) operations. For simple backends, this can return the input pointer directly. For backends that need different metadata for initiator vs. target descriptors, this creates a separate target metadata object.

```cpp
virtual nixl_status_t loadLocalMD(nixlBackendMD* input,
                                  nixlBackendMD* &output);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `nixlBackendMD*` | Metadata from `registerMem()` |
| `output` | `nixlBackendMD*&` | [out] Metadata for target-side operations (can be same as input) |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_BACKEND` (default) |

```cpp
nixl_status_t MyBackend::loadLocalMD(nixlBackendMD* input,
                                     nixlBackendMD* &output) {
    output = input;  // Simple case: same metadata for initiator and target
    return NIXL_SUCCESS;
}
```

<Note>
Required only if `supportsLocal()` returns `true`. The default implementation returns `NIXL_ERR_BACKEND`.
</Note>

### unloadMD

Releases resources associated with a metadata object that was created by `loadRemoteMD()` or `loadLocalMD()`. Always required.

```cpp
virtual nixl_status_t unloadMD(nixlBackendMD* input) = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `nixlBackendMD*` | Metadata object to release |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_status_t MyBackend::unloadMD(nixlBackendMD* input) {
    delete static_cast<MyRemoteMD*>(input);
    return NIXL_SUCCESS;
}
```

<Warning>
If `loadLocalMD()` returns the same pointer as input (identity return), `unloadMD()` must handle this case without double-freeing. Check whether the pointer is the same as the registered memory metadata before deleting.
</Warning>

## Transfer Operations

<Tip>
For the user-facing transfer workflow, see [C++ API Reference -- Transfer Operations](../api-reference/cpp-api#postxferreq).
</Tip>

### prepXfer

Prepares a transfer request. Creates a backend-specific request handle (`nixlBackendReqH` subclass) that stores the state needed for the transfer. This method does not start the transfer.

```cpp
virtual nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               nixlBackendReqH* &handle,
                               const nixl_opt_b_args_t* opt_args=nullptr) const = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `operation` | `const nixl_xfer_op_t&` | Transfer operation (`NIXL_READ` or `NIXL_WRITE`) |
| `local` | `const nixl_meta_dlist_t&` | Local descriptor list with metadata pointers |
| `remote` | `const nixl_meta_dlist_t&` | Remote descriptor list with metadata pointers |
| `remote_agent` | `const std::string&` | Name of the remote agent (or self for loopback) |
| `handle` | `nixlBackendReqH*&` | [out] Backend-created request handle |
| `opt_args` | `const nixl_opt_b_args_t*` | Optional arguments (notification, custom params). Defaults to `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code |

```cpp
nixl_status_t MyBackend::prepXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH* &handle,
                                  const nixl_opt_b_args_t* opt_args) const {
    handle = new MyReqHandle(operation, local, remote);
    return NIXL_SUCCESS;
}
```

### postXfer

Posts (starts) an asynchronous transfer. This method should initiate the data movement and return immediately without waiting for completion. If the transfer is very small, it may complete synchronously and return `NIXL_SUCCESS` directly.

```cpp
virtual nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               nixlBackendReqH* &handle,
                               const nixl_opt_b_args_t* opt_args=nullptr) const = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `operation` | `const nixl_xfer_op_t&` | Transfer operation (`NIXL_READ` or `NIXL_WRITE`) |
| `local` | `const nixl_meta_dlist_t&` | Local descriptor list with metadata pointers |
| `remote` | `const nixl_meta_dlist_t&` | Remote descriptor list with metadata pointers |
| `remote_agent` | `const std::string&` | Name of the remote agent (or self for loopback) |
| `handle` | `nixlBackendReqH*&` | Request handle from `prepXfer()` (may be updated) |
| `opt_args` | `const nixl_opt_b_args_t*` | Optional arguments (notification, custom params). Defaults to `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` (complete), `NIXL_IN_PROG` (started), or error code |

```cpp
nixl_status_t MyBackend::postXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH* &handle,
                                  const nixl_opt_b_args_t* opt_args) const {
    auto *req = static_cast<MyReqHandle*>(handle);
    req->submit();  // Start async I/O
    return NIXL_IN_PROG;
}
```

### checkXfer

Checks the status of an in-progress transfer. The backend may use this call to internally progress its I/O engine.

```cpp
virtual nixl_status_t checkXfer(nixlBackendReqH* handle) const = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `nixlBackendReqH*` | Request handle from `prepXfer()` |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` (complete), `NIXL_IN_PROG` (still running), or error code |

```cpp
nixl_status_t MyBackend::checkXfer(nixlBackendReqH* handle) const {
    auto *req = static_cast<MyReqHandle*>(handle);
    return req->isComplete() ? NIXL_SUCCESS : NIXL_IN_PROG;
}
```

### releaseReqH

Releases a transfer request handle. If the transfer is still in progress, the backend should attempt to abort it. This method should be non-blocking.

```cpp
virtual nixl_status_t releaseReqH(nixlBackendReqH* handle) const = 0;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `nixlBackendReqH*` | Request handle to release |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or error code if abort failed |

```cpp
nixl_status_t MyBackend::releaseReqH(nixlBackendReqH* handle) const {
    delete static_cast<MyReqHandle*>(handle);
    return NIXL_SUCCESS;
}
```

<Warning>
The method is named `releaseReqH`, not `releaseXferReq` as mentioned in some older documentation. Always use the name from `backend_engine.h`.
</Warning>

<Note>
A transfer request is prepped once but can be posted multiple times (after reaching `NIXL_SUCCESS` state each time). There is no ordering guarantee across transfer requests. The user is responsible for avoiding concurrent writes to the same memory.
</Note>

### estimateXferCost

Estimates the duration and cost of a transfer operation. This method is optional and allows the agent to make informed decisions about backend selection.

```cpp
virtual nixl_status_t
estimateXferCost(const nixl_xfer_op_t &operation,
                 const nixl_meta_dlist_t &local,
                 const nixl_meta_dlist_t &remote,
                 const std::string &remote_agent,
                 nixlBackendReqH *const &handle,
                 std::chrono::microseconds &duration,
                 std::chrono::microseconds &err_margin,
                 nixl_cost_t &method,
                 const nixl_opt_args_t *extra_params = nullptr) const;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `operation` | `const nixl_xfer_op_t&` | Transfer operation (`NIXL_READ` or `NIXL_WRITE`) |
| `local` | `const nixl_meta_dlist_t&` | Local descriptor list |
| `remote` | `const nixl_meta_dlist_t&` | Remote descriptor list |
| `remote_agent` | `const std::string&` | Name of the remote agent |
| `handle` | `nixlBackendReqH *const&` | Request handle from `prepXfer()` |
| `duration` | `std::chrono::microseconds&` | [out] Estimated transfer duration |
| `err_margin` | `std::chrono::microseconds&` | [out] Error margin on the estimate |
| `method` | `nixl_cost_t&` | [out] Estimation method used |
| `extra_params` | `const nixl_opt_args_t*` | Optional extra parameters. Defaults to `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_NOT_SUPPORTED` (default) |

```cpp
nixl_status_t MyBackend::estimateXferCost(
    const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
    const nixl_meta_dlist_t &remote, const std::string &remote_agent,
    nixlBackendReqH *const &handle, std::chrono::microseconds &duration,
    std::chrono::microseconds &err_margin, nixl_cost_t &method,
    const nixl_opt_args_t *extra_params) const {
    duration = std::chrono::microseconds(100);  // 100us estimate
    err_margin = std::chrono::microseconds(50);
    method = nixl_cost_t::ANALYTICAL_BACKEND;
    return NIXL_SUCCESS;
}
```

## Notification Handling

### getNotifs

Retrieves notifications received from remote agents (or local via loopback). The agent iterates over all notification-capable backends and merges results.

```cpp
virtual nixl_status_t getNotifs(notif_list_t &notif_list);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `notif_list` | `notif_list_t&` | [out] Vector of (agent name, notification message) pairs |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_BACKEND` (default) |

```cpp
nixl_status_t MyBackend::getNotifs(notif_list_t &notif_list) {
    // Drain received notifications into the list
    for (auto &notif : pendingNotifs_) {
        notif_list.push_back(notif);
    }
    pendingNotifs_.clear();
    return NIXL_SUCCESS;
}
```

<Note>
Required only if `supportsNotif()` returns `true`. The default implementation returns `NIXL_ERR_BACKEND`. The backend must extract the source agent name from each received notification.
</Note>

### genNotif

Generates a standalone notification to a remote agent. This notification is not bound to any transfer and does not provide ordering guarantees.

```cpp
virtual nixl_status_t genNotif(const std::string &remote_agent,
                               const std::string &msg) const;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `const std::string&` | Name of the target agent |
| `msg` | `const std::string&` | Notification message payload |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_BACKEND` (default) |

```cpp
nixl_status_t MyBackend::genNotif(const std::string &remote_agent,
                                  const std::string &msg) const {
    sendNotification(remote_agent, msg);
    return NIXL_SUCCESS;
}
```

<Note>
Required only if `supportsNotif()` returns `true`. The default implementation returns `NIXL_ERR_BACKEND`.
</Note>

## Optional Methods

These methods have default implementations and can be overridden to add functionality. They all return `NIXL_ERR_NOT_SUPPORTED` or perform a no-op by default.

### prepMemView (remote)

Prepares a memory view for remote buffers. Memory views provide an alternative access pattern for reading remote memory.

```cpp
virtual nixl_status_t
prepMemView(const nixl_remote_meta_dlist_t &descs,
            nixlMemViewH &handle,
            const nixl_opt_b_args_t *opt_args = nullptr) const;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `const nixl_remote_meta_dlist_t&` | Remote memory descriptor list |
| `handle` | `nixlMemViewH&` | [out] Memory view handle (typedef for `void*`) |
| `opt_args` | `const nixl_opt_b_args_t*` | Optional arguments. Defaults to `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_NOT_SUPPORTED` (default) |

```cpp
nixl_status_t MyBackend::prepMemView(const nixl_remote_meta_dlist_t &descs,
                                     nixlMemViewH &handle,
                                     const nixl_opt_b_args_t *opt_args) const {
    handle = createRemoteView(descs);
    return NIXL_SUCCESS;
}
```

### prepMemView (local)

Prepares a memory view for local buffers.

```cpp
virtual nixl_status_t
prepMemView(const nixl_meta_dlist_t &descs,
            nixlMemViewH &handle,
            const nixl_opt_b_args_t *opt_args = nullptr) const;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `const nixl_meta_dlist_t&` | Local memory descriptor list |
| `handle` | `nixlMemViewH&` | [out] Memory view handle (typedef for `void*`) |
| `opt_args` | `const nixl_opt_b_args_t*` | Optional arguments. Defaults to `nullptr`. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_NOT_SUPPORTED` (default) |

```cpp
nixl_status_t MyBackend::prepMemView(const nixl_meta_dlist_t &descs,
                                     nixlMemViewH &handle,
                                     const nixl_opt_b_args_t *opt_args) const {
    handle = createLocalView(descs);
    return NIXL_SUCCESS;
}
```

### releaseMemView

Releases a memory view handle previously created by `prepMemView()`.

```cpp
virtual void releaseMemView(nixlMemViewH handle) const;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `nixlMemViewH` | Memory view handle to release |
| **Returns** | `void` | -- |

```cpp
void MyBackend::releaseMemView(nixlMemViewH handle) const {
    delete static_cast<MyMemView*>(handle);
}
```

### queryMem

Queries information about a list of memory or storage descriptors. File and object backends can override this to provide metadata such as file size, existence, or permissions.

```cpp
virtual nixl_status_t
queryMem(const nixl_reg_dlist_t &descs,
         std::vector<nixl_query_resp_t> &resp) const;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `const nixl_reg_dlist_t&` | Registration descriptor list to query |
| `resp` | `std::vector<nixl_query_resp_t>&` | [out] Response vector (one entry per descriptor). Empty optional if no data available. |
| **Returns** | `nixl_status_t` | `NIXL_SUCCESS` or `NIXL_ERR_NOT_SUPPORTED` (default) |

```cpp
nixl_status_t MyBackend::queryMem(const nixl_reg_dlist_t &descs,
                                  std::vector<nixl_query_resp_t> &resp) const {
    for (size_t i = 0; i < descs.descCount(); i++) {
        nixl_b_params_t info;
        info["size"] = std::to_string(getFileSize(descs[i]));
        resp.push_back(info);
    }
    return NIXL_SUCCESS;
}
```

<Note>
All optional methods default to `NIXL_ERR_NOT_SUPPORTED` or no-op. Override them only if your backend has the functionality to support them.
</Note>
