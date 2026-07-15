---
title: Rust API Reference
description: Complete Rust API reference for NIXL data transfers.
---

This is the Rust API reference for the NVIDIA Inference Xfer Library (NIXL). The Rust bindings provide a safe, idiomatic interface through the `Agent` struct. For the C++ native API, see the [C++ API Reference](./cpp-api). For Python bindings, see the [Python API Reference](./python-api).

Behavior shared by all bindings is documented in [Northbound API Semantics](./northbound-api).

Key Rust-specific features of the NIXL bindings:

- **Result-based error handling**: All fallible operations return `Result<T, NixlError>`
- **RAII resource management**: `Agent`, `XferRequest`, `XferDlistHandle`, `RegistrationHandle`, `Backend`, `OptArgs`, `NotificationMap`, `RegDescList`, `XferDescList`, `QueryResponseList`, `Params`, and `MemList` all implement `Drop` for automatic cleanup
- **Thread safety**: `Agent` wraps inner state in `Arc<RwLock<AgentInner>>`, making it `Clone`, `Send`, and `Sync`
- **Trait-based extensibility**: `MemoryRegion`, `NixlDescriptor`, `NixlRegistration` traits for custom memory types

<Warning>
`Agent::clone()` creates a handle to the **same** underlying agent via `Arc`, not an independent copy. All clones share the same backends, registrations, and metadata. This is intentional for multi-threaded usage -- clone the `Agent` and move it into threads or tasks.
</Warning>

To use the NIXL Rust API, add `nixl_sys` as a dependency and import:

```rust
use nixl_sys::Agent;
```

## Types, Enums and Defines

This section documents the structs, enums, traits, and handle types that make up the Rust NIXL API. Subsections are organized to mirror the C++ API structure.

### Structs and Configuration

### Agent

The main Transfer Agent struct. Each agent represents one endpoint in a data transfer and manages backends, memory registrations, metadata, and transfer operations.

`Agent` implements `Clone` (shared state via `Arc<RwLock<AgentInner>>`), `Send`, `Sync`, `Debug`, and `Drop`. The `Drop` implementation invalidates all remote metadata, destroys all backends, and destroys the underlying C agent handle.

```rust
use nixl_sys::Agent;
let agent = Agent::new("my_agent")?;
```

### AgentConfig

Per-agent configuration struct with a `Default` implementation. All fields have defaults, so `AgentConfig::default()` is valid for most use cases.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_prog_thread` | `bool` | `true` | Enable progress thread for asynchronous transfer advancement |
| `enable_listen_thread` | `bool` | `false` | Enable listener thread for incoming peer connections |
| `listen_port` | `i32` | `0` | Port for the listener thread (0 = system-assigned) |
| `thread_sync` | `ThreadSync` | `ThreadSync::None` | Thread synchronization mode for multi-threaded usage |
| `num_workers` | `u32` | `1` | Number of worker threads |
| `pthr_delay_us` | `u64` | `0` | Progress thread delay between iterations (microseconds) |
| `lthr_delay_us` | `u64` | `100_000` | Listener thread sleep duration (microseconds) |
| `capture_telemetry` | `bool` | `false` | Enable telemetry capture for transfer performance metrics |

```rust
use nixl_sys::{Agent, AgentConfig, ThreadSync};

let cfg = AgentConfig {
    enable_prog_thread: true,
    enable_listen_thread: true,
    listen_port: 5555,
    ..Default::default()
};
let agent = Agent::new_configured("my_agent", &cfg)?;
```

### Enumerations

### ThreadSync

Thread synchronization mode enum for multi-threaded agent usage.

| Variant | Description |
|---------|-------------|
| `None` | No synchronization (default). Single-threaded usage only. |
| `Strict` | Full mutual exclusion on all operations. |
| `Rw` | Reader-writer lock: concurrent reads, exclusive writes. |
| `Default` | Alias for `None`. |

**C++ equivalent:** [`nixl_thread_sync_t`](./cpp-api#nixl_thread_sync_t)

### XferStatus

Transfer status enum returned by `get_xfer_status()`.

| Variant | Description |
|---------|-------------|
| `Success` | Transfer completed successfully |
| `InProgress` | Transfer is still running |

The `is_success()` helper method returns `true` for `XferStatus::Success`.

```rust
let status = agent.get_xfer_status(&req)?;
if status.is_success() {
    println!("Transfer complete");
}
```

### XferOp

Transfer operation direction.

| Variant | Description |
|---------|-------------|
| `Read` | Read data from the remote side into local buffers |
| `Write` | Write data from local buffers to the remote side |

**C++ equivalent:** [`nixl_xfer_op_t`](./cpp-api#nixl_xfer_op_t)

### CostMethod

Cost estimation method identifier returned by `estimate_xfer_cost()`.

| Variant | Description |
|---------|-------------|
| `AnalyticalBackend` | Analytical backend cost estimate |
| `Unknown` | Unknown estimation method |

**C++ equivalent:** [`nixl_cost_t`](./cpp-api#nixl_cost_t)

### NixlError

Error enum for all NIXL operations. All fallible methods return `Result<T, NixlError>`.

| Variant | Description |
|---------|-------------|
| `InvalidParam` | Invalid parameter provided to NIXL |
| `BackendError` | Backend-level error occurred |
| `StringConversionError(NulError)` | Failed to convert string (contains interior nul byte) |
| `IndexOutOfBounds` | Index out of bounds |
| `InvalidDataPointer` | Invalid data pointer returned from C API |
| `FailedToCreateXferRequest` | Failed to create a transfer request |
| `RegDescListCreationFailed` | Failed to create a registration descriptor list |
| `RegDescAddFailed` | Failed to add a registration descriptor |
| `FailedToCreateXferDlistHandle` | Failed to create a transfer descriptor list handle |
| `FailedToCreateBackend` | Failed to create a backend |
| `NoTelemetry` | Telemetry is not enabled or transfer is not complete |

`NixlError` implements `std::error::Error` (via `thiserror`) and `Debug`, making it compatible with the standard Rust error handling ecosystem.

```rust
match agent.create_backend("UCX", &params) {
    Ok(backend) => { /* success */ }
    Err(NixlError::InvalidParam) => { /* handle invalid parameter */ }
    Err(NixlError::FailedToCreateBackend) => { /* handle backend creation failure */ }
    Err(e) => { /* other error */ }
}
```

### Backend

Opaque handle to a NIXL backend engine. Implements `Send + Sync` (safe to share across threads) and `Debug`. The backend is destroyed when the owning `Agent` is dropped.

**C++ equivalent:** [`nixlBackendH*`](./cpp-api#handle-types)

### OptArgs

Optional arguments struct passed to many agent methods. Uses setter methods to configure fields.

| Method | Description |
|--------|-------------|
| `OptArgs::new()` | Create a new empty optional arguments struct |
| `add_backend(&mut self, backend: &Backend)` | Add a backend to limit the operation scope |
| `set_notification_message(&mut self, message: &[u8])` | Set the notification message as raw bytes |
| `get_notification_message(&self)` | Get the notification message |
| `set_has_notification(&mut self, has: bool)` | Enable or disable notification |
| `has_notification(&self)` | Check if notification is enabled |
| `set_skip_descriptor_merge(&mut self, skip: bool)` | Set whether to skip descriptor merging |
| `skip_descriptor_merge(&self)` | Check if descriptor merging is skipped |
| `set_ip_addr(&mut self, ip: &str)` | Set IP address for peer-to-peer metadata operations |
| `set_port(&mut self, port: u16)` | Set port for metadata operations |

All setter methods return `Result<(), NixlError>`. `OptArgs` implements `Drop` for automatic cleanup.

**C++ equivalent:** [`nixlAgentOptionalArgs`](./cpp-api#nixlagentoptionalargs)

```rust
use nixl_sys::OptArgs;

let mut opts = OptArgs::new()?;
opts.set_notification_message(b"transfer_done")?;
opts.add_backend(&backend)?;
```

### MemType

Memory segment types supported by NIXL.

| Variant | Description |
|---------|-------------|
| `Dram` | Standard host memory (CPU DRAM) |
| `Vram` | GPU high-bandwidth memory (HBM/VRAM) |
| `Block` | Block-level storage devices |
| `Object` | Distributed object stores (S3, Azure Blob) |
| `File` | Local and remote file systems |
| `Unknown` | Unknown memory type |

`MemType` derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Serialize`, `Deserialize`, and implements `Display`.

**C++ equivalent:** [`nixl_mem_t`](./cpp-api#nixl_mem_t)

### Descriptor Types

NIXL uses descriptor lists to represent sets of memory regions for registration and transfer.

**`RegDescList`** -- Registration descriptor list for memory segments to register.

| Method | Returns | Description |
|--------|---------|-------------|
| `RegDescList::new(mem_type: MemType)` | `Result<Self, NixlError>` | Create a new list for the given memory type |
| `add_desc(&mut self, addr, len, dev_id)` | `()` | Add a descriptor (address, length, device ID) |
| `add_desc_with_meta(&mut self, addr, len, dev_id, metadata)` | `()` | Add a descriptor with metadata bytes |
| `add_storage_desc(&mut self, desc: &dyn NixlDescriptor)` | `Result<(), NixlError>` | Add from a type implementing `NixlDescriptor` |
| `len()` | `Result<usize, NixlError>` | Number of descriptors |
| `is_empty()` | `Result<bool, NixlError>` | Check if list is empty |
| `get(index)` | `Result<&RegDescriptor, NixlError>` | Access descriptor by index |
| `rem_desc(index)` | `Result<(), NixlError>` | Remove descriptor at index |
| `clear()` | `()` | Remove all descriptors |
| `serialize()` / `deserialize(bytes)` | `Result<Vec<u8>, NixlError>` / `Result<Self, NixlError>` | Bincode serialization |

Also supports `Index<usize>`, `IndexMut<usize>`, `PartialEq`, `Debug`, and `Drop`.

**C++ equivalent:** [`nixl_reg_dlist_t`](./cpp-api#nixl-descriptors)

**`XferDescList`** -- Transfer descriptor list for memory regions in transfer operations.

| Method | Returns | Description |
|--------|---------|-------------|
| `XferDescList::new(mem_type: MemType)` | `Result<Self, NixlError>` | Create a new list for the given memory type |
| `add_desc(&mut self, addr, len, dev_id)` | `()` | Add a descriptor (address, length, device ID) |
| `add_storage_desc(&mut self, desc: &D)` | `Result<(), NixlError>` | Add from a type implementing `NixlDescriptor` |
| `len()` | `Result<usize, NixlError>` | Number of descriptors |
| `is_empty()` | `Result<bool, NixlError>` | Check if list is empty |
| `get(index)` | `Result<&XferDescriptor, NixlError>` | Access descriptor by index |
| `rem_desc(index)` | `Result<(), NixlError>` | Remove descriptor at index |
| `clear()` | `()` | Remove all descriptors |
| `serialize()` / `deserialize(bytes)` | `Result<Vec<u8>, NixlError>` / `Result<Self, NixlError>` | Bincode serialization |

Also supports `Index<usize>`, `IndexMut<usize>`, `PartialEq`, `Debug`, and `Drop`.

**C++ equivalent:** [`nixl_xfer_dlist_t`](./cpp-api#nixl-descriptors)

**`XferDlistHandle`** -- Opaque handle to a prepared descriptor list. Returned by `prepare_xfer_dlist()` and consumed by `make_xfer_req()`. Implements `Drop`, which releases the prepared list on the agent side.

**C++ equivalent:** [`nixlDlistH*`](./cpp-api#handle-types)

**`RegistrationHandle`** -- Handle to a registered memory region. Implements `Drop`, which automatically deregisters the memory via the agent. Call `deregister()` for explicit deregistration.

| Method | Returns | Description |
|--------|---------|-------------|
| `agent_name()` | `Option<String>` | Get the name of the agent this memory is registered with |
| `deregister(&mut self)` | `Result<(), NixlError>` | Explicitly deregister the memory |

<Note>
When a `RegistrationHandle` is dropped, it automatically deregisters the memory from the agent. You do not need to call `deregister()` explicitly unless you need to handle errors.
</Note>

### Transfer and Notification Types

### XferRequest

Handle to an active transfer request. Implements `Send + Sync` (safe to share across threads) and `Drop` (auto-releases the request and underlying resources).

| Method | Returns | Description |
|--------|---------|-------------|
| `get_telemetry(&self)` | `Result<XferTelemetry, NixlError>` | Get telemetry data for this transfer |

**C++ equivalent:** [`nixlXferReqH*`](./cpp-api#handle-types)

### NotificationMap

Container for notifications received from remote agents. Implements `Drop` for automatic cleanup.

| Method | Returns | Description |
|--------|---------|-------------|
| `NotificationMap::new()` | `Result<Self, NixlError>` | Create a new empty notification map |
| `len()` | `Result<usize, NixlError>` | Number of agents with notifications |
| `is_empty()` | `Result<bool, NixlError>` | Check if there are no notifications |
| `agents()` | `NotificationMapAgentIterator` | Iterator over agent names |
| `get_notifications_size(agent_name)` | `Result<usize, NixlError>` | Number of notifications for an agent |
| `get_notifications(agent_name)` | `Result<NotificationIterator, NixlError>` | Iterator over notifications (as raw bytes) |
| `get_notification_bytes(agent_name, index)` | `Result<Vec<u8>, NixlError>` | Get a specific notification as bytes |
| `take_notifs()` | `Result<HashMap<String, Vec<String>>, NixlError>` | Extract all notifications as strings and clear the map |

```rust
use nixl_sys::NotificationMap;

let mut notifs = NotificationMap::new()?;
agent.get_notifications(&mut notifs, None)?;
for agent_name in notifs.agents() {
    let name = agent_name?;
    println!("Notifications from: {}", name);
}
```

### XferTelemetry

Transfer telemetry data containing timing and performance metrics.

| Field | Type | Description |
|-------|------|-------------|
| `start_time_us` | `u64` | Start time in microseconds since epoch |
| `post_duration_us` | `u64` | Post operation duration in microseconds |
| `xfer_duration_us` | `u64` | Transfer duration in microseconds |
| `total_bytes` | `u64` | Total bytes transferred |
| `desc_count` | `u64` | Number of descriptors in the transfer |

| Helper Method | Returns | Description |
|---------------|---------|-------------|
| `start_time()` | `Duration` | Start time as a `Duration` since Unix epoch |
| `post_duration()` | `Duration` | Post operation duration |
| `xfer_duration()` | `Duration` | Transfer duration |
| `total_duration()` | `Duration` | Total duration (post + transfer) |
| `transfer_rate_bps()` | `f64` | Transfer rate in bytes per second |

**C++ equivalent:** [`nixlXferTelemetry`](./cpp-api#nixlxfertelemetry)

### Collection Types

### MemList

List of memory types supported by a backend. Returned by `get_plugin_params()` and `get_backend_params()`. Implements `Drop` for automatic cleanup.

| Method | Returns | Description |
|--------|---------|-------------|
| `is_empty()` | `Result<bool, NixlError>` | Check if the list is empty |
| `len()` | `Result<usize, NixlError>` | Number of memory types |
| `get(index)` | `Result<MemType, NixlError>` | Get memory type at index |
| `iter()` | `MemListIterator` | Iterator over memory types |

### Params

Key-value parameter map used for backend configuration. Returned by `get_plugin_params()` and `get_backend_params()`, and accepted by `create_backend()`. Implements `Drop` for automatic cleanup.

| Method | Returns | Description |
|--------|---------|-------------|
| `Params::from(iter)` | `Result<Self, NixlError>` | Create from an iterator of `(key, value)` pairs |
| `set(key, value)` | `Result<(), NixlError>` | Set a key-value pair |
| `is_empty()` | `Result<bool, NixlError>` | Check if empty |
| `iter()` | `Result<ParamIterator, NixlError>` | Iterator over key-value pairs |
| `clone()` | `Result<Self, NixlError>` | Deep copy the parameters |

Supports `IntoIterator` (yields `(&str, &str)` pairs) and conversion to `HashMap<String, String>`.

**C++ equivalent:** [`nixl_b_params_t`](./cpp-api#type-aliases)

```rust
use nixl_sys::Params;
use std::collections::HashMap;

let params = Params::from([("key1", "value1"), ("key2", "value2")])?;
let map: HashMap<String, String> = HashMap::from(params.iter()?);
```

### QueryResponseList

List of query responses from `query_mem()`. Each response may optionally contain a `Params` object. Implements `Drop`.

| Method | Returns | Description |
|--------|---------|-------------|
| `len()` | `Result<usize, NixlError>` | Number of responses |
| `is_empty()` | `Result<bool, NixlError>` | Check if empty |
| `get(index)` | `Result<QueryResponse, NixlError>` | Get response at index |
| `iter()` | `Result<QueryResponseIterator, NixlError>` | Iterator over responses |

Each `QueryResponse` has `has_value()` and `get_params()` methods.

### SystemStorage

Built-in DRAM (`MemType::Dram`) implementation of `NixlRegistration`. Provides a simple heap-allocated buffer that can be registered with an agent.

| Method | Returns | Description |
|--------|---------|-------------|
| `SystemStorage::new(size: usize)` | `Result<Self, NixlError>` | Allocate a new zero-initialized buffer |
| `memset(value: u8)` | `()` | Fill the buffer with a byte value |
| `as_slice()` | `&[u8]` | Get a read-only slice of the data |

`SystemStorage` implements `MemoryRegion`, `NixlDescriptor` (with `mem_type() -> Dram`, `device_id() -> 0`), and `NixlRegistration`.

```rust
use nixl_sys::{Agent, SystemStorage, NixlRegistration};

let agent = Agent::new("my_agent")?;
let mut storage = SystemStorage::new(1024)?;
storage.register(&agent, None)?;
// storage is now registered with the agent
// Drop automatically deregisters when storage goes out of scope
```

### Trait Hierarchy

NIXL defines three traits for custom memory types, forming a hierarchy:

```
MemoryRegion
  |
  +-- NixlDescriptor
        |
        +-- NixlRegistration
```

**`MemoryRegion`** -- Base trait for types that represent a contiguous memory region. Requires `Debug + Send + Sync`.

```rust
pub trait MemoryRegion: std::fmt::Debug + Send + Sync {
    /// Get a raw pointer to the storage
    unsafe fn as_ptr(&self) -> *const u8;
    /// Returns the total size in bytes
    fn size(&self) -> usize;
}
```

**`NixlDescriptor: MemoryRegion`** -- Extends `MemoryRegion` with NIXL-specific metadata.

```rust
pub trait NixlDescriptor: MemoryRegion {
    /// Get the memory type
    fn mem_type(&self) -> MemType;
    /// Get the device ID
    fn device_id(&self) -> u64;
}
```

**`NixlRegistration: NixlDescriptor`** -- Extends `NixlDescriptor` with the ability to register memory with an agent.

```rust
pub trait NixlRegistration: NixlDescriptor {
    fn register(
        &mut self,
        agent: &Agent,
        opt_args: Option<&OptArgs>,
    ) -> Result<(), NixlError>;
}
```

Implement these traits on your own types to integrate custom memory (e.g., GPU memory, NVMe buffers) with NIXL. See `SystemStorage` for a reference implementation.

## Initialization and Configuration

<Tip>
For a complete workflow example, see [Quick Start -- Agent Initialization](../getting-started/quick-start#agent-initialization).
</Tip>

### new

Create a new Transfer Agent with default configuration.

**C++ equivalent:** [`nixlAgent`](./cpp-api#nixlagent)

```rust
pub fn new(name: &str) -> Result<Self, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `&str` | Unique name for this agent |
| **Returns** | `Result<Agent, NixlError>` | The constructed agent or error |

```rust
let agent = Agent::new("my_agent")?;
```

### new_configured

Create a new Transfer Agent with custom configuration.

**C++ equivalent:** [`nixlAgent`](./cpp-api#nixlagent) (with config parameter)

```rust
pub fn new_configured(name: &str, cfg: &AgentConfig) -> Result<Self, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `&str` | Unique name for this agent |
| `cfg` | `&AgentConfig` | Agent configuration struct |
| **Returns** | `Result<Agent, NixlError>` | The constructed agent or error |

```rust
let cfg = AgentConfig {
    enable_prog_thread: true,
    capture_telemetry: true,
    ..Default::default()
};
let agent = Agent::new_configured("my_agent", &cfg)?;
```

### name

Get the name of this agent.

```rust
pub fn name(&self) -> String
```

| **Returns** | `String` | The agent name |

### get_available_plugins

Discover the available backend plug-ins found in the plug-in search paths.

**C++ equivalent:** [`getAvailPlugins`](./cpp-api#getavailplugins)

```rust
pub fn get_available_plugins(&self) -> Result<StringList, NixlError>
```

| **Returns** | `Result<StringList, NixlError>` | List of available plug-in names |

```rust
let plugins = agent.get_available_plugins()?;
for i in 0..plugins.len()? {
    println!("Plugin: {}", plugins.get(i)?);
}
```

<Note>
The `Agent` destructor (`Drop`) automatically invalidates all remote metadata, destroys all backends, and releases the underlying C handle. No explicit cleanup is needed.
</Note>

## Backend Management

<Tip>
For a complete workflow example, see [Quick Start -- Backend Creation](../getting-started/quick-start#backend-creation).
</Tip>

### get_plugin_params

Get the supported memory types and initialization parameters for a backend plug-in, before creating an instance.

**C++ equivalent:** [`getPluginParams`](./cpp-api#getpluginparams)

```rust
pub fn get_plugin_params(
    &self,
    plugin_name: &str,
) -> Result<(MemList, Params), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `plugin_name` | `&str` | Plugin backend type name (e.g., `"UCX"`) |
| **Returns** | `Result<(MemList, Params), NixlError>` | Supported memory types and default parameters |

```rust
let (mems, params) = agent.get_plugin_params("UCX")?;
for mem in mems.iter() {
    println!("Supports: {:?}", mem?);
}
```

### get_backend_params

Get the parameters and memory types of an already-instantiated backend.

**C++ equivalent:** [`getBackendParams`](./cpp-api#getbackendparams)

```rust
pub fn get_backend_params(
    &self,
    backend: &Backend,
) -> Result<(MemList, Params), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | `&Backend` | Backend handle from `create_backend()` |
| **Returns** | `Result<(MemList, Params), NixlError>` | Memory types and current parameters |

### create_backend

Instantiate a backend engine with the given parameters.

**C++ equivalent:** [`createBackend`](./cpp-api#createbackend)

```rust
pub fn create_backend(
    &self,
    plugin: &str,
    params: &Params,
) -> Result<Backend, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `plugin` | `&str` | Backend type name (e.g., `"UCX"`, `"GDS"`) |
| `params` | `&Params` | Backend-specific initialization parameters |
| **Returns** | `Result<Backend, NixlError>` | Backend handle for subsequent operations |

<Note>
Multiple backends can be created on the same agent. NIXL automatically selects the best backend for each transfer based on the source and destination memory types and the backends available on both agents.
</Note>

```rust
let params = Params::from([("key", "value")])?;
let backend = agent.create_backend("UCX", &params)?;
```

### get_backend

Get a backend by name.

```rust
pub fn get_backend(&self, name: &str) -> Option<Backend>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `&str` | Backend name (same as the plug-in name used in `create_backend()`) |
| **Returns** | `Option<Backend>` | Backend handle if found, `None` otherwise |

## Memory Registration

<Tip>
For a complete workflow example, see [Quick Start -- Memory Registration](../getting-started/quick-start#memory-registration).
</Tip>

### register_memory

Register a memory descriptor with the agent. Returns a `RegistrationHandle` that automatically deregisters the memory when dropped.

**C++ equivalent:** [`registerMem`](./cpp-api#registermem)

```rust
pub fn register_memory(
    &self,
    descriptor: &impl NixlDescriptor,
    opt_args: Option<&OptArgs>,
) -> Result<RegistrationHandle, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `descriptor` | `&impl NixlDescriptor` | Memory descriptor to register (e.g., `&SystemStorage`) |
| `opt_args` | `Option<&OptArgs>` | Optional. If `backends` is set via `add_backend()`, registration is limited to those backends. |
| **Returns** | `Result<RegistrationHandle, NixlError>` | Handle that auto-deregisters on drop |

<Note>
If no backend hints are provided, NIXL auto-selects all compatible backends for the given memory type. Memory must be registered before metadata exchange.
</Note>

You can also use the `NixlRegistration` trait pattern for a more ergonomic API:

```rust
use nixl_sys::{SystemStorage, NixlRegistration};

let mut storage = SystemStorage::new(4096)?;
storage.register(&agent, None)?;
// storage now holds the RegistrationHandle internally
```

### query_mem

Query information about registered memory or storage segments from a specific backend.

**C++ equivalent:** [`queryMem`](./cpp-api#querymem)

```rust
pub fn query_mem(
    &self,
    descs: &RegDescList,
    opt_args: Option<&OptArgs>,
) -> Result<QueryResponseList, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `&RegDescList` | Registration descriptor list to query |
| `opt_args` | `Option<&OptArgs>` | The target backend should be specified via `add_backend()` |
| **Returns** | `Result<QueryResponseList, NixlError>` | Query responses for each descriptor |

```rust
let mut opts = OptArgs::new()?;
opts.add_backend(&backend)?;
let responses = agent.query_mem(&descs, Some(&opts))?;
for resp in responses.iter()? {
    if resp.has_value()? {
        let params = resp.get_params()?;
        // inspect params...
    }
}
```

### make_connection

Proactively establish a connection to a remote agent, instead of deferring it to the first transfer.

**C++ equivalent:** [`makeConnection`](./cpp-api#makeconnection)

```rust
pub fn make_connection(
    &self,
    remote_agent: &str,
    opt_args: Option<&OptArgs>,
) -> Result<(), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `&str` | Name of the remote agent to connect to |
| `opt_args` | `Option<&OptArgs>` | Optional. Limit connection to specific backends. |
| **Returns** | `Result<(), NixlError>` | Success or error |

<Note>
Connections are normally established lazily on the first transfer. Use `make_connection()` to pre-establish connections and avoid first-transfer latency.
</Note>

```rust
agent.make_connection("remote_agent", None)?;
```

## Transfer Preparation

<Tip>
For a complete workflow example, see [Quick Start -- Creating and Executing Transfers](../getting-started/quick-start#creating-and-executing-transfers).
</Tip>

### prepare_xfer_dlist

Prepare a descriptor list for use in transfer requests. Elements from the prepared list can later be selected by index in `make_xfer_req()`.

**C++ equivalent:** [`prepXferDlist`](./cpp-api#prepxferdlist-4-parameter)

```rust
pub fn prepare_xfer_dlist(
    &self,
    agent_name: &str,
    descs: &XferDescList,
    opt_args: Option<&OptArgs>,
) -> Result<XferDlistHandle, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_name` | `&str` | Agent name for the prepared list. Use `""` for local descriptors, the remote agent's name for remote descriptors. |
| `descs` | `&XferDescList` | Descriptor list to prepare |
| `opt_args` | `Option<&OptArgs>` | Optional. Limit preparation to specific backends. |
| **Returns** | `Result<XferDlistHandle, NixlError>` | Prepared descriptor list handle (auto-released on drop) |

<Note>
Use `""` (empty string) as `agent_name` for local descriptors, equivalent to `NIXL_INIT_AGENT` in C++. Use the remote agent's name for remote-side descriptors.
</Note>

```rust
let local_hndl = agent.prepare_xfer_dlist("", &local_descs, None)?;
let remote_hndl = agent.prepare_xfer_dlist("remote_agent", &remote_descs, None)?;
```

### make_xfer_req

Create a transfer request by selecting indices from already-prepared descriptor list handles.

**C++ equivalent:** [`makeXferReq`](./cpp-api#makexferreq)

```rust
pub fn make_xfer_req(
    &self,
    operation: XferOp,
    local_descs: &XferDlistHandle,
    local_indices: &[i32],
    remote_descs: &XferDlistHandle,
    remote_indices: &[i32],
    opt_args: Option<&OptArgs>,
) -> Result<XferRequest, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `operation` | `XferOp` | Transfer direction (`Read` or `Write`) |
| `local_descs` | `&XferDlistHandle` | Local prepared descriptor list handle |
| `local_indices` | `&[i32]` | Indices into the local descriptor list |
| `remote_descs` | `&XferDlistHandle` | Remote prepared descriptor list handle |
| `remote_indices` | `&[i32]` | Indices into the remote descriptor list |
| `opt_args` | `Option<&OptArgs>` | Optional. Limit backend selection or attach notification. |
| **Returns** | `Result<XferRequest, NixlError>` | Transfer request handle (auto-released on drop) |

```rust
let indices = [0, 1, 2];
let req = agent.make_xfer_req(
    XferOp::Write,
    &local_hndl, &indices,
    &remote_hndl, &indices,
    None,
)?;
```

### create_xfer_req

Combined API that creates a transfer request directly from two descriptor lists. Internally prepares both sides and creates the transfer handle. Equivalent to calling `prepare_xfer_dlist()` for each side followed by `make_xfer_req()` with all indices.

**C++ equivalent:** [`createXferReq`](./cpp-api#createxferreq)

```rust
pub fn create_xfer_req(
    &self,
    operation: XferOp,
    local_descs: &XferDescList,
    remote_descs: &XferDescList,
    remote_agent: &str,
    opt_args: Option<&OptArgs>,
) -> Result<XferRequest, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `operation` | `XferOp` | Transfer direction (`Read` or `Write`) |
| `local_descs` | `&XferDescList` | Local descriptor list |
| `remote_descs` | `&XferDescList` | Remote descriptor list |
| `remote_agent` | `&str` | Remote agent name |
| `opt_args` | `Option<&OptArgs>` | Optional. Limit backend selection or attach notification. |
| **Returns** | `Result<XferRequest, NixlError>` | Transfer request handle (auto-released on drop) |

<Note>
If the same descriptors are reused across multiple transfers, prefer `prepare_xfer_dlist()` + `make_xfer_req()` to avoid repeated preparation overhead. `create_xfer_req()` is simpler for one-off transfers.
</Note>

```rust
let req = agent.create_xfer_req(
    XferOp::Write,
    &local_descs,
    &remote_descs,
    "remote_agent",
    None,
)?;
```

## Transfer Operations

<Tip>
For a complete workflow example, see [Quick Start -- Creating and Executing Transfers](../getting-started/quick-start#creating-and-executing-transfers).
</Tip>

### estimate_xfer_cost

Estimate the cost (duration) of executing a transfer request before posting it.

**C++ equivalent:** [`estimateXferCost`](./cpp-api#estimatexfercost)

```rust
pub fn estimate_xfer_cost(
    &self,
    req: &XferRequest,
    opt_args: Option<&OptArgs>,
) -> Result<(i64, i64, CostMethod), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `req` | `&XferRequest` | Transfer request handle |
| `opt_args` | `Option<&OptArgs>` | Optional arguments |
| **Returns** | `Result<(i64, i64, CostMethod), NixlError>` | Tuple of (duration_us, error_margin_us, method) |

```rust
let (duration_us, margin_us, method) = agent.estimate_xfer_cost(&req, None)?;
println!("Estimated: {}us +/- {}us ({:?})", duration_us, margin_us, method);
```

### post_xfer_req

Submit a transfer request, initiating the data transfer. The operation is non-blocking.

**C++ equivalent:** [`postXferReq`](./cpp-api#postxferreq)

```rust
pub fn post_xfer_req(
    &self,
    req: &XferRequest,
    opt_args: Option<&OptArgs>,
) -> Result<bool, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `req` | `&XferRequest` | Transfer request handle |
| `opt_args` | `Option<&OptArgs>` | Optional. Notification message can be provided. |
| **Returns** | `Result<bool, NixlError>` | `false` if transfer completed inline, `true` if in progress |

```rust
let in_progress = agent.post_xfer_req(&req, None)?;
if in_progress {
    // Poll with get_xfer_status()
}
```

### get_xfer_status

Check the status of a transfer request.

**C++ equivalent:** [`getXferStatus`](./cpp-api#getxferstatus)

```rust
pub fn get_xfer_status(&self, req: &XferRequest) -> Result<XferStatus, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `req` | `&XferRequest` | Transfer request handle |
| **Returns** | `Result<XferStatus, NixlError>` | `Success` or `InProgress` |

```rust
loop {
    match agent.get_xfer_status(&req)? {
        XferStatus::Success => {
            println!("Transfer complete");
            break;
        }
        XferStatus::InProgress => {
            // continue polling
        }
    }
}
```

### get_telemetry (on XferRequest)

Get telemetry data for a completed transfer request. Called on the `XferRequest` directly.

**C++ equivalent:** [`getXferTelemetry`](./cpp-api#getxfertelemetry)

```rust
pub fn get_telemetry(&self) -> Result<XferTelemetry, NixlError>
```

| **Returns** | `Result<XferTelemetry, NixlError>` | Timing and performance metrics |

<Note>
Telemetry is only available if `capture_telemetry` was set to `true` in `AgentConfig` and the transfer has completed. Returns `NixlError::NoTelemetry` otherwise.
</Note>

```rust
let telemetry = req.get_telemetry()?;
println!("Transfer rate: {:.2} MB/s",
    telemetry.transfer_rate_bps() / 1_000_000.0);
```

### query_xfer_backend

Query which backend was selected for a transfer request.

**C++ equivalent:** [`queryXferBackend`](./cpp-api#queryxferbackend)

```rust
pub fn query_xfer_backend(&self, req: &XferRequest) -> Result<Backend, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `req` | `&XferRequest` | Transfer request handle |
| **Returns** | `Result<Backend, NixlError>` | Backend handle used for this transfer |

<Note>
`XferRequest` implements `Drop`, which automatically releases the transfer request and underlying resources. No explicit release call is needed -- simply let the request go out of scope.
</Note>

## Memory View

<Note>
Memory View is not currently exposed in the Rust bindings. For Memory View functionality, use the [C++ API](./cpp-api#memory-view) directly or request this feature in the NIXL repository.
</Note>

## Notification Handling

### get_notifications

Retrieve pending notifications from remote agents.

**C++ equivalent:** [`getNotifs`](./cpp-api#getnotifs)

```rust
pub fn get_notifications(
    &self,
    notifs: &mut NotificationMap,
    opt_args: Option<&OptArgs>,
) -> Result<(), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `notifs` | `&mut NotificationMap` | Notification map to populate |
| `opt_args` | `Option<&OptArgs>` | Optional. Use `add_backend()` to filter by backend. |
| **Returns** | `Result<(), NixlError>` | Success or error |

```rust
let mut notifs = NotificationMap::new()?;
agent.get_notifications(&mut notifs, None)?;

// Iterate through notifications
let all = notifs.take_notifs()?;
for (agent_name, messages) in &all {
    for msg in messages {
        println!("{}: {}", agent_name, msg);
    }
}
```

### send_notification

Send a notification to a remote agent.

**C++ equivalent:** [`genNotif`](./cpp-api#gennotif)

```rust
pub fn send_notification(
    &self,
    remote_agent: &str,
    message: &[u8],
    backend: Option<&Backend>,
) -> Result<(), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `&str` | Name of the remote agent |
| `message` | `&[u8]` | Notification message as raw bytes |
| `backend` | `Option<&Backend>` | Optional backend to use for sending |
| **Returns** | `Result<(), NixlError>` | Success or error |

```rust
agent.send_notification("remote_agent", b"transfer_done", None)?;
```

## Metadata -- Side Channel

<Tip>
For a complete workflow example, see [Quick Start -- Metadata Exchange](../getting-started/quick-start#metadata-exchange).
</Tip>

The side channel uses serialized byte arrays to exchange metadata between agents without relying on etcd. One agent calls `get_local_md()` to serialize its metadata, sends the bytes over an application-level channel (TCP, gRPC, shared memory, etc.), and the other agent calls `load_remote_md()` to deserialize and load it.

### get_local_md

Get this agent's metadata serialized as a byte array, suitable for sending to a remote agent via an application-level channel.

**C++ equivalent:** [`getLocalMD`](./cpp-api#getlocalmd)

```rust
pub fn get_local_md(&self) -> Result<Vec<u8>, NixlError>
```

| **Returns** | `Result<Vec<u8>, NixlError>` | Serialized metadata bytes |

```rust
let metadata = agent.get_local_md()?;
// Send metadata bytes to remote agent via your transport
```

### get_local_partial_md

Get partial metadata for specific registered memory regions.

**C++ equivalent:** [`getLocalPartialMD`](./cpp-api#getlocalpartialmd)

```rust
pub fn get_local_partial_md(
    &self,
    descs: &RegDescList,
    opt_args: Option<&OptArgs>,
) -> Result<Vec<u8>, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `&RegDescList` | Registration descriptor list to get metadata for |
| `opt_args` | `Option<&OptArgs>` | Optional arguments |
| **Returns** | `Result<Vec<u8>, NixlError>` | Serialized partial metadata bytes |

### load_remote_md

Load a remote agent's metadata from a byte slice received via an application-level channel.

**C++ equivalent:** [`loadRemoteMD`](./cpp-api#loadremotemd)

```rust
pub fn load_remote_md(&self, metadata: &[u8]) -> Result<String, NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadata` | `&[u8]` | Serialized metadata bytes from the remote agent |
| **Returns** | `Result<String, NixlError>` | Name of the remote agent whose metadata was loaded |

```rust
// Receive metadata bytes from remote agent
let remote_name = agent.load_remote_md(&remote_metadata_bytes)?;
println!("Loaded metadata for: {}", remote_name);
```

### invalidate_remote_md

Invalidate a specific remote agent's cached metadata.

**C++ equivalent:** [`invalidateRemoteMD`](./cpp-api#invalidateremotemd)

```rust
pub fn invalidate_remote_md(&self, remote_agent: &str) -> Result<(), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `&str` | Name of the remote agent to invalidate |
| **Returns** | `Result<(), NixlError>` | Success or error |

### invalidate_all_remotes

Invalidate all cached remote metadata.

```rust
pub fn invalidate_all_remotes(&self) -> Result<(), NixlError>
```

| **Returns** | `Result<(), NixlError>` | Success or error |

## Metadata -- Direct Channel

<Tip>
For a complete workflow example, see [Quick Start -- Metadata Exchange](../getting-started/quick-start#metadata-exchange).
</Tip>

The direct channel uses etcd as a shared key-value store for automatic metadata discovery. Agents publish their metadata to etcd and fetch remote agent metadata by name, without needing an application-level transport.

### send_local_md

Publish this agent's metadata to etcd for discovery by other agents.

**C++ equivalent:** [`sendLocalMD`](./cpp-api#sendlocalmd)

```rust
pub fn send_local_md(&self, opt_args: Option<&OptArgs>) -> Result<(), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `opt_args` | `Option<&OptArgs>` | Optional. Use `set_ip_addr()` and `set_port()` for etcd connection. |
| **Returns** | `Result<(), NixlError>` | Success or error |

```rust
let mut opts = OptArgs::new()?;
opts.set_ip_addr("127.0.0.1")?;
opts.set_port(2379)?;
agent.send_local_md(Some(&opts))?;
```

### send_local_partial_md

Publish partial metadata for specific registered memory regions to etcd.

**C++ equivalent:** [`sendLocalPartialMD`](./cpp-api#sendlocalpartialmd)

```rust
pub fn send_local_partial_md(
    &self,
    descs: &RegDescList,
    opt_args: Option<&OptArgs>,
) -> Result<(), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `descs` | `&RegDescList` | Registration descriptor list |
| `opt_args` | `Option<&OptArgs>` | Optional. etcd connection settings. |
| **Returns** | `Result<(), NixlError>` | Success or error |

### fetch_remote_md

Fetch a remote agent's metadata from etcd. Once fetched, the metadata is loaded and cached locally, enabling communication with the remote agent.

**C++ equivalent:** [`fetchRemoteMD`](./cpp-api#fetchremotemd)

```rust
pub fn fetch_remote_md(
    &self,
    remote_name: &str,
    opt_args: Option<&OptArgs>,
) -> Result<(), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_name` | `&str` | Name of the remote agent to fetch metadata for |
| `opt_args` | `Option<&OptArgs>` | Optional. etcd connection settings. |
| **Returns** | `Result<(), NixlError>` | Success or error |

```rust
agent.fetch_remote_md("remote_agent", Some(&opts))?;
```

### invalidate_local_md

Remove this agent's metadata from etcd, signaling to other agents that it is no longer available.

**C++ equivalent:** [`invalidateLocalMD`](./cpp-api#invalidatelocalmd)

```rust
pub fn invalidate_local_md(&self, opt_args: Option<&OptArgs>) -> Result<(), NixlError>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `opt_args` | `Option<&OptArgs>` | Optional. etcd connection settings. |
| **Returns** | `Result<(), NixlError>` | Success or error |

### check_remote_metadata

Check if a remote agent's metadata is available and optionally if specific descriptors can be found.

**C++ equivalent:** [`checkRemoteMD`](./cpp-api#checkremotemd)

```rust
pub fn check_remote_metadata(
    &self,
    remote_agent: &str,
    descs: Option<&XferDescList>,
) -> bool
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_agent` | `&str` | Name of the remote agent to check |
| `descs` | `Option<&XferDescList>` | Optional descriptor list to validate against remote metadata |
| **Returns** | `bool` | `true` if metadata is available (and descriptors found if provided) |

```rust
if agent.check_remote_metadata("remote_agent", None) {
    println!("Remote agent is available");
}
```
