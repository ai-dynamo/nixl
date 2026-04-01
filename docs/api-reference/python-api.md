---
title: Python API Reference
description: Complete Python API reference for NIXL data transfers.
---

This is the Python API reference for the NVIDIA Inference Xfer Library (NIXL). Python provides a high-level, Pythonic interface to NIXL via the `nixl_agent` class. For the C++ API, see the [C++ API Reference](./cpp-api). For Rust bindings, see the [Rust API Reference](./rust-api).

Key Python-specific features:

- **Auto-backend initialization**: Backends specified in the config are created automatically during agent construction
- **String status returns**: Transfer methods return `"DONE"`, `"PROC"`, or `"ERR"` instead of enum values
- **Tensor/numpy auto-conversion**: Methods accept PyTorch tensors, numpy arrays, or raw tuples for memory descriptors
- **Pythonic method names**: `register_memory()` instead of `registerMem()`, `transfer()` instead of `postXferReq()`

To use the NIXL Python API:

```python
from nixl._api import nixl_agent
```

## Types, Enums and Defines

This section documents the configuration class, handle types, and Python-specific conventions used throughout the NIXL Python API. Subsections are organized to mirror the C++ API structure.

### Configuration

### nixl_agent_config

Configuration dataclass for creating a Transfer Agent. Pass an instance to the `nixl_agent` constructor to customize agent behavior.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_prog_thread` | `bool` | `True` | Enable the progress thread for asynchronous transfer completion |
| `enable_listen_thread` | `bool` | `False` | Enable the listener thread for direct metadata communication |
| `listen_port` | `int` | `0` | Port for the listener thread (0 for auto-assign) |
| `capture_telemetry` | `bool` | `False` | Enable telemetry capture for transfer performance metrics |
| `num_threads` | `int` | `0` | Number of threads for supported multi-threaded backends |
| `backends` | `list[str]` | `["UCX"]` | Backends to auto-create at agent initialization |

```python
from nixl._api import nixl_agent, nixl_agent_config

# Default config -- UCX backend auto-created
config = nixl_agent_config()

# Custom config with multiple backends and telemetry
config = nixl_agent_config(
    backends=["UCX", "GDS"],
    capture_telemetry=True,
    num_threads=4
)
```

<Note>
The `backends` field triggers auto-creation of the specified backends during agent construction. This is a Python-only convenience -- C++ and Rust require explicit `createBackend()` calls. You can still create additional backends later with `create_backend()`.
</Note>

### nixl_agent

The main agent class. All NIXL operations are performed through methods on this class. See the functional group sections below for the complete method reference.

### Handle Types

### nixl_prepped_dlist_handle

Opaque handle returned by `prep_xfer_dlist()`. Represents a prepared transfer descriptor list that can be reused across multiple transfers.

| Method | Description |
|--------|-------------|
| `release()` | Explicitly free resources associated with this handle |

The handle performs best-effort cleanup through `__del__` if `release()` is not called. If finalization fails, an error is logged.

```python
handle = agent.prep_xfer_dlist("remote_agent", descs)
# ... use handle ...
handle.release()  # Explicit cleanup
```

### nixl_xfer_handle

Opaque handle returned by `make_prepped_xfer()` and `initialize_xfer()`. Represents an active or prepared transfer operation.

| Method | Description |
|--------|-------------|
| `release()` | Explicitly free resources. If the transfer is active, NIXL will attempt to cancel it |

If `release()` is not called, `__del__` attempts cleanup. Failed finalization queues the handle in an internal leaked-handles list for re-release during agent destruction.

```python
xfer_handle = agent.make_prepped_xfer("WRITE", local_h, [0], remote_h, [0])
agent.transfer(xfer_handle)
# ... wait for completion ...
xfer_handle.release()
```

<Warning>
Always call `release()` on transfer handles when done. Relying on garbage collection may delay cleanup and leak resources until agent destruction.
</Warning>

### nixl_backend_handle

Type alias for `int`. Returned by `create_backend()` and used internally. You typically interact with backends by their string names in the Python API.

### Memory Type and Operation Strings

Python uses string names instead of C++ enumerations for memory types and transfer operations.

**Memory types:**

| String | C++ Equivalent | Description |
|--------|---------------|-------------|
| `"DRAM"` | `DRAM_SEG` | Standard host memory (CPU DRAM) |
| `"VRAM"` | `VRAM_SEG` | GPU high-bandwidth memory |
| `"BLOCK"` | `BLK_SEG` | Block-level storage devices |
| `"OBJ"` | `OBJ_SEG` | Distributed object stores |
| `"FILE"` | `FILE_SEG` | Local and remote file systems |

The Python API also accepts device-style aliases `"cpu"` (maps to `DRAM`) and `"cuda"` (maps to `VRAM`) for convenience.

**Transfer operations:**

| String | C++ Equivalent | Description |
|--------|---------------|-------------|
| `"READ"` | `NIXL_READ` | Read data from the remote side into local buffers |
| `"WRITE"` | `NIXL_WRITE` | Write data from local buffers to the remote side |

**Transfer status strings:**

| String | C++ Equivalent | Description |
|--------|---------------|-------------|
| `"DONE"` | `NIXL_SUCCESS` | Transfer completed successfully |
| `"PROC"` | `NIXL_IN_PROG` | Transfer is in progress |
| `"ERR"` | `NIXL_ERR_*` | Transfer encountered an error |

## Initialization and Configuration

<Tip>
For a complete walkthrough of agent setup and the overall transfer workflow, see [Quick Start -- Agent Initialization](../getting-started/quick-start#agent-initialization).
</Tip>

### \_\_init\_\_

Create a new Transfer Agent. The agent manages backends, memory registrations, metadata, and transfer operations for a single process.

**C++ equivalent:** [`nixlAgent`](./cpp-api#nixlagent)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | `str` | *(required)* | Unique name for this agent |
| `nixl_conf` | `nixl_agent_config` | `None` | Configuration object. If `None`, uses default config |
| `instantiate_all` | `bool` | `False` | If `True`, auto-create all available plug-in backends |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `nixl_agent` | New agent instance |

```python
from nixl._api import nixl_agent, nixl_agent_config

# Default agent -- auto-creates UCX backend
agent = nixl_agent("my_agent")

# Agent with custom configuration
config = nixl_agent_config(backends=["UCX", "GDS"], capture_telemetry=True)
agent = nixl_agent("my_agent", config)
```

<Note>
Unlike C++, Python auto-creates backends listed in the config during construction. The constructor queries all available plug-ins, caches their parameters and memory types, and then initializes the specified backends. If a requested backend plug-in is not available, a warning is logged and that backend is skipped.
</Note>

### get_plugin_list

Get the list of all available backend plug-ins discovered at agent initialization.

**C++ equivalent:** [`getAvailPlugins`](./cpp-api#getavailplugins)

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `list[str]` | List of plug-in names (e.g., `["UCX", "GDS", "POSIX"]`) |

```python
plugins = agent.get_plugin_list()
# ['UCX', 'GDS', 'POSIX', ...]
```

<Note>
The plug-in list is cached at agent initialization. This call returns the cached list without re-querying the system.
</Note>

### get_plugin_mem_types

Get the memory types supported by a specific plug-in.

**C++ equivalent:** [`getPluginParams`](./cpp-api#getpluginparams) (memory types portion)

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | `str` | Name of the plug-in |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `list[str]` | Supported memory type strings (e.g., `["DRAM", "VRAM"]`) |

```python
mem_types = agent.get_plugin_mem_types("UCX")
# ['DRAM', 'VRAM']
```

### get_plugin_params

Get the initialization parameters of a plug-in. Returns a dictionary where keys are parameter names and values are their default values.

**C++ equivalent:** [`getPluginParams`](./cpp-api#getpluginparams) (parameters portion)

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | `str` | Name of the plug-in |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `dict[str, str]` | Parameter name to default value mapping |

```python
params = agent.get_plugin_params("UCX")
# {'num_threads': '0', ...}
```

## Backend Management

<Tip>
For a walkthrough of backend creation and selection, see [Quick Start -- Backend Creation](../getting-started/quick-start#backend-creation).
</Tip>

### get_backend_mem_types

Get the memory types supported by an initialized backend. After initialization, supported memory types may differ from the plug-in defaults.

**C++ equivalent:** [`getBackendParams`](./cpp-api#getbackendparams) (memory types portion)

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | `str` | Name of the backend |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `list[str]` | Supported memory type strings |

```python
mem_types = agent.get_backend_mem_types("UCX")
```

### get_backend_params

Get the parameters of an initialized backend. Available parameters may differ from the plug-in defaults after initialization.

**C++ equivalent:** [`getBackendParams`](./cpp-api#getbackendparams) (parameters portion)

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | `str` | Name of the backend |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `dict[str, str]` | Parameter name to value mapping |

```python
params = agent.get_backend_params("UCX")
```

### create_backend

Initialize a backend with the specified parameters. This is only needed for backends not listed in the `backends` config field at agent creation time.

**C++ equivalent:** [`createBackend`](./cpp-api#createbackend)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | *(required)* | Name of the backend plug-in to initialize |
| `initParams` | `dict[str, str]` | `{}` | Initialization parameters |

```python
# Create a GDS backend with custom threads
agent.create_backend("GDS", {"num_threads": "4"})
```

<Note>
The Python API automatically caches the backend's parameters and supported memory types after creation. You can query them later with `get_backend_params()` and `get_backend_mem_types()`.
</Note>

## Memory Registration

<Tip>
For a complete walkthrough of memory registration in the transfer workflow, see [Quick Start -- Memory Registration](../getting-started/quick-start#memory-registration).
</Tip>

### register_memory

Register memory regions with one or more backends. Accepts multiple input formats including PyTorch tensors for convenience.

**C++ equivalent:** [`registerMem`](./cpp-api#registermem)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reg_list` | *see below* | *(required)* | Memory regions to register |
| `mem_type` | `str` | `None` | Memory type string (required for tuples and numpy) |
| `backends` | `list[str]` | `[]` | Backend names to register with; empty means all compatible backends |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `nixlRegDList` | Registration descriptor list for use with `deregister_memory()` |

**Accepted input types for `reg_list`:**

| Input Type | `mem_type` Required? | Description |
|------------|---------------------|-------------|
| `torch.Tensor` | No | Single contiguous tensor; memory type auto-detected from device |
| `list[torch.Tensor]` | No | List of contiguous tensors on the same device |
| `numpy.ndarray` (Nx3) | Yes | Each row is `[address, size, device_id]` |
| `list[tuple]` | Yes | List of 4-tuples `(address, size, device_id, meta_info)` |
| `nixlRegDList` | No | Passed through directly |

```python
import torch

# Register a GPU tensor (simplest form)
tensor = torch.zeros(1024, device="cuda:0")
reg_descs = agent.register_memory(tensor)

# Register multiple tensors
tensors = [torch.zeros(512, device="cuda:0") for _ in range(4)]
reg_descs = agent.register_memory(tensors)

# Register with raw tuples and specific backend
descs = [(addr, size, 0, "")]
reg_descs = agent.register_memory(descs, mem_type="DRAM", backends=["UCX"])
```

<Warning>
You can pass tensors directly -- there is no need to manually extract `data_ptr()` and create tuple lists. The Python API handles tensor-to-descriptor conversion internally via `get_reg_descs()`.
</Warning>

### deregister_memory

Deregister memory regions from backends.

**C++ equivalent:** [`deregisterMem`](./cpp-api#deregistermem)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dereg_list` | `nixlRegDList` | *(required)* | Descriptor list from `register_memory()` or `get_reg_descs()` |
| `backends` | `list[str]` | `[]` | Backend names to deregister from; empty means all backends that have these regions |

```python
agent.deregister_memory(reg_descs)
```

### query_memory

Query information about registered memory or storage for a specific backend.

**C++ equivalent:** [`queryMem`](./cpp-api#querymem)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reg_list` | *various* | *(required)* | Memory regions, tensors, or `nixlRegDList` to query |
| `backend` | `str` | *(required)* | Backend name to query |
| `mem_type` | `str` | `None` | Memory type (required for tuples/numpy) |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `list[Optional[dict[str, str]]]` | Query results; `None` for entries not found, otherwise a dict with backend-specific info |

```python
results = agent.query_memory(reg_descs, "UCX")
```

### make_connection

Proactively establish a connection with a remote agent to reduce first-transfer latency. This is optional -- NIXL establishes connections on demand if not called.

**C++ equivalent:** [`makeConnection`](./cpp-api#makeconnection)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remote_agent` | `str` | *(required)* | Name of the remote agent |
| `backends` | `list[str]` | `[]` | Limit connections to specific backends; empty means all applicable backends |

```python
agent.make_connection("remote_agent")
```

### Convenience Methods: Descriptor Conversion

These Python-only helper methods convert various input formats into NIXL descriptor lists. They are called internally by `register_memory()`, `prep_xfer_dlist()`, and `initialize_xfer()`, but can also be used directly for advanced workflows.

#### get_reg_descs

Convert various input types into a registration descriptor list.

**C++ equivalent:** None (Python-only convenience)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `descs` | *various* | *(required)* | Memory descriptors in any supported format |
| `mem_type` | `str` | `None` | Memory type (required for tuples and numpy inputs) |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `nixlRegDList` | Registration descriptor list |

**Accepted input types:**

| Input Type | `mem_type` Required? | Description |
|------------|---------------------|-------------|
| `torch.Tensor` | No | Single contiguous tensor |
| `list[torch.Tensor]` | No | List of contiguous tensors on the same device |
| `numpy.ndarray` (Nx3) | Yes | Each row is `[address, size, device_id]`; empty meta info |
| `list[tuple]` | Yes | List of 4-tuples `(address, size, device_id, meta_info)` |
| `nixlRegDList` | No | Passed through directly |

```python
import torch
import numpy as np

# From a single tensor
reg_descs = agent.get_reg_descs(torch.zeros(1024, device="cuda:0"))

# From a tensor list
reg_descs = agent.get_reg_descs([t1, t2], "VRAM")

# From a numpy array
reg_descs = agent.get_reg_descs(np.array([[addr, size, dev]]), "DRAM")

# From raw tuples
reg_descs = agent.get_reg_descs([(addr, size, dev, "meta")], "DRAM")
```

#### get_xfer_descs

Convert various input types into a transfer descriptor list.

**C++ equivalent:** None (Python-only convenience)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `descs` | *various* | *(required)* | Transfer descriptors in any supported format |
| `mem_type` | `str` | `None` | Memory type (required for tuples and numpy inputs) |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `nixlXferDList` | Transfer descriptor list |

**Accepted input types:**

| Input Type | `mem_type` Required? | Description |
|------------|---------------------|-------------|
| `torch.Tensor` | No | Single contiguous tensor |
| `list[torch.Tensor]` | No | List of contiguous tensors on the same device |
| `numpy.ndarray` (Nx3) | Yes | Each row is `[address, size, device_id]` |
| `list[tuple]` | Yes | List of 3-tuples `(address, size, device_id)` |
| `nixlXferDList` | No | Passed through directly |

```python
# From a tensor
xfer_descs = agent.get_xfer_descs(torch.zeros(1024, device="cuda:0"))

# From raw tuples
xfer_descs = agent.get_xfer_descs([(addr, size, 0)], "VRAM")
```

<Note>
Transfer descriptors use 3-tuples `(address, size, device_id)` while registration descriptors use 4-tuples `(address, size, device_id, meta_info)`. The extra `meta_info` field in registration descriptors carries opaque metadata (e.g., file paths for FILE_SEG).
</Note>

## Transfer Preparation

<Tip>
For a walkthrough of the full transfer lifecycle including preparation, posting, and status checking, see [Quick Start -- Creating and Executing Transfers](../getting-started/quick-start#creating-and-executing-transfers).
</Tip>

### prep_xfer_dlist

Prepare a transfer descriptor list for efficient reuse across multiple transfers. Both the local and remote sides of a transfer must be prepared before creating a transfer request.

**C++ equivalent:** [`prepXferDlist`](./cpp-api#prepxferdlist-4-parameter)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | `str` | *(required)* | `"NIXL_INIT_AGENT"` or `""` for local descriptors; remote agent name for remote descriptors; local agent name for loopback |
| `xfer_list` | *various* | *(required)* | Transfer descriptors (tensors, tuples, numpy, or `nixlXferDList`) |
| `mem_type` | `str` | `None` | Memory type (required for tuples/numpy) |
| `backends` | `list[str]` | `[]` | Limit which backends are used during preparation |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `nixl_prepped_dlist_handle` | Opaque handle to the prepared descriptor list |

```python
# Prepare local descriptors
local_handle = agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_tensors)

# Prepare remote descriptors
remote_handle = agent.prep_xfer_dlist("remote_agent", remote_descs, "VRAM")
```

<Note>
Preparation succeeds if at least one backend can handle all elements in the descriptor list. The Python API internally detects whether the call is local (when `agent_name` is `"NIXL_INIT_AGENT"` or `""`) and dispatches to the appropriate C++ overload.
</Note>

### estimate_xfer_cost

Estimate the cost of a transfer operation.

**C++ equivalent:** [`estimateXferCost`](./cpp-api#estimatexfercost)

| Parameter | Type | Description |
|-----------|------|-------------|
| `req_handle` | `nixl_xfer_handle` | Handle to the transfer operation |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `tuple[int, int, str]` | `(duration, error_margin, method)` -- times are in microseconds, `method` is `"ANALYTICAL_BACKEND"` or `"UNKNOWN"` |

```python
duration, err_margin, method = agent.estimate_xfer_cost(xfer_handle)
```

### make_prepped_xfer

Create a transfer request from prepared descriptor list handles. This is the recommended approach when the same descriptors are used in multiple transfers.

**C++ equivalent:** [`makeXferReq`](./cpp-api#makexferreq)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | `str` | *(required)* | `"WRITE"` or `"READ"` |
| `local_xfer_side` | `nixl_prepped_dlist_handle` | *(required)* | Local descriptor handle from `prep_xfer_dlist()` |
| `local_indices` | `list[int]` or `numpy.ndarray` | *(required)* | Indices selecting local descriptors |
| `remote_xfer_side` | `nixl_prepped_dlist_handle` | *(required)* | Remote descriptor handle from `prep_xfer_dlist()` |
| `remote_indices` | `list[int]` or `numpy.ndarray` | *(required)* | Indices selecting remote descriptors |
| `notif_msg` | `bytes` | `b""` | Notification message sent after transfer completion |
| `backends` | `list[str]` | `[]` | Limit which backends NIXL can use |
| `skip_desc_merge` | `bool` | `False` | *Deprecated.* Whether to skip descriptor merging optimization |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `nixl_xfer_handle` | Opaque handle for posting and checking the transfer |

```python
xfer_handle = agent.make_prepped_xfer(
    "WRITE", local_handle, [0, 1], remote_handle, [0, 1],
    notif_msg=b"transfer_complete"
)
```

### initialize_xfer

Create a transfer request directly from descriptor lists without prior preparation. This is a combined API that prepares the descriptor lists and creates the transfer in one call.

**C++ equivalent:** [`createXferReq`](./cpp-api#createxferreq)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | `str` | *(required)* | `"WRITE"` or `"READ"` |
| `local_descs` | `nixlXferDList` | *(required)* | Local transfer descriptors (from `get_xfer_descs()`) |
| `remote_descs` | `nixlXferDList` | *(required)* | Remote transfer descriptors (from `get_xfer_descs()`) |
| `remote_agent` | `str` | *(required)* | Name of the remote agent |
| `notif_msg` | `bytes` | `b""` | Notification message sent after transfer completion |
| `backends` | `list[str]` | `[]` | Limit which backends NIXL can use |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `nixl_xfer_handle` | Opaque handle for posting and checking the transfer |

```python
local_descs = agent.get_xfer_descs(local_tensor)
remote_descs = agent.get_xfer_descs(remote_tensor)
xfer_handle = agent.initialize_xfer("WRITE", local_descs, remote_descs, "remote_agent")
```

<Note>
If you share common descriptors across different transfer requests, prefer `prep_xfer_dlist()` with `make_prepped_xfer()` to avoid repeated preparation overhead. Use `initialize_xfer()` for one-off transfers where simplicity matters more than reuse.
</Note>

## Transfer Operations

<Tip>
For a walkthrough of transfer posting and status checking, see [Quick Start -- Creating and Executing Transfers](../getting-started/quick-start#creating-and-executing-transfers).
</Tip>

### transfer

Initiate a data transfer. After calling this, poll `check_xfer_state()` until completion.

**C++ equivalent:** [`postXferReq`](./cpp-api#postxferreq)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `handle` | `nixl_xfer_handle` | *(required)* | Transfer handle from `make_prepped_xfer()` or `initialize_xfer()` |
| `notif_msg` | `bytes` | `b""` | Notification message (can override or set per transfer call) |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `str` | `"DONE"` if completed immediately, `"PROC"` if in progress, `"ERR"` on error |

```python
status = agent.transfer(xfer_handle)
if status == "PROC":
    while agent.check_xfer_state(xfer_handle) == "PROC":
        pass  # Poll until complete
```

### check_xfer_state

Check the current state of a transfer operation.

**C++ equivalent:** [`getXferStatus`](./cpp-api#getxferstatus)

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `nixl_xfer_handle` | Transfer handle |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `str` | `"DONE"` if complete, `"PROC"` if in progress, `"ERR"` on error |

```python
status = agent.check_xfer_state(xfer_handle)
```

### get_xfer_telemetry

Get telemetry data for a transfer request. Requires `capture_telemetry=True` in the agent config.

**C++ equivalent:** [`getXferTelemetry`](./cpp-api#getxfertelemetry)

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `nixl_xfer_handle` | Transfer handle |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `nixlXferTelemetry` | Telemetry object with `startTime`, `postDuration`, `xferDuration` (microseconds), `totalBytes`, and `descCount` fields |

```python
telem = agent.get_xfer_telemetry(xfer_handle)
print(f"Transfer took {telem.xferDuration} us for {telem.totalBytes} bytes")
```

### query_xfer_backend

Query which backend was selected for a transfer operation.

**C++ equivalent:** [`queryXferBackend`](./cpp-api#queryxferbackend)

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `nixl_xfer_handle` | Transfer handle |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `str` | Name of the backend chosen for this transfer (e.g., `"UCX"`) |

```python
backend = agent.query_xfer_backend(xfer_handle)
# "UCX"
```

### release_xfer_handle

Release a transfer handle, freeing associated resources. If the transfer is active, NIXL will attempt to cancel it.

**C++ equivalent:** [`releaseXferReq`](./cpp-api#releasexferreq)

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `nixl_xfer_handle` | Transfer handle to release |

```python
agent.release_xfer_handle(xfer_handle)
```

<Note>
This delegates to `handle.release()`. You can also call `release()` directly on the handle object.
</Note>

### release_dlist_handle

Release a prepared descriptor list handle, freeing associated resources.

**C++ equivalent:** [`releasedDlistH`](./cpp-api#releaseddlisth)

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `nixl_prepped_dlist_handle` | Descriptor list handle to release |

```python
agent.release_dlist_handle(prep_handle)
```

## Memory View

<Note>
Memory View APIs (`prepMemView`, `releaseMemView`) are not currently exposed in the Python bindings. For memory view operations, refer to the [C++ API Reference -- Memory View](./cpp-api#memory-view).
</Note>

## Notification Handling

### get_new_notifs

Get new notifications that have arrived at the agent since the last call. Returns a fresh dictionary each time (does not accumulate).

**C++ equivalent:** [`getNotifs`](./cpp-api#getnotifs)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backends` | `list[str]` | `[]` | Limit which backends are checked for notifications; empty means all |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `dict[str, list[bytes]]` | Map of remote agent names to lists of notification messages |

```python
notifs = agent.get_new_notifs()
for agent_name, messages in notifs.items():
    print(f"From {agent_name}: {messages}")
```

### update_notifs

Get new notifications and accumulate them in the agent's internal notification map. Unlike `get_new_notifs()`, this builds up all unhandled notifications over time.

**C++ equivalent:** [`getNotifs`](./cpp-api#getnotifs) (with accumulation)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backends` | `list[str]` | `[]` | Limit which backends are checked; empty means all |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `dict[str, list[bytes]]` | Accumulated notification map (same reference as `agent.notifs`) |

```python
all_notifs = agent.update_notifs()
```

### check_remote_xfer_done

Check whether a specific notification has been received from a remote agent. This is a Python-only convenience method that combines `update_notifs()` with a tag-based lookup.

**C++ equivalent:** None (Python-only convenience)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remote_agent_name` | `str` | *(required)* | Name of the remote agent |
| `lookup_tag` | `bytes` | *(required)* | Tag to match against notification messages |
| `backends` | `list[str]` | `[]` | Limit which backends are checked |
| `tag_is_prefix` | `bool` | `True` | If `True`, match `lookup_tag` as a prefix; if `False`, match as substring |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `bool` | `True` if a matching notification was found and removed; `False` otherwise |

```python
# Poll until remote agent signals completion
while not agent.check_remote_xfer_done("remote_agent", b"xfer_done"):
    pass
```

<Note>
When a matching notification is found, it is removed from the internal notification map. Subsequent calls will not find the same notification.
</Note>

### send_notif

Send a standalone notification to a remote agent, not bound to any transfer.

**C++ equivalent:** [`genNotif`](./cpp-api#gennotif)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remote_agent_name` | `str` | *(required)* | Name of the remote agent |
| `notif_msg` | `bytes` | *(required)* | Message to send (received as `bytes` by the target) |
| `backend` | `str` | `None` | Specific backend to send through; `None` uses the default |

```python
agent.send_notif("remote_agent", b"step_complete")
```

## Metadata -- Side Channel

Side-channel metadata exchange uses an out-of-band mechanism to serialize and transfer agent metadata as opaque byte blobs. This is useful when agents communicate through a shared store or message queue.

<Tip>
For a walkthrough of metadata exchange patterns, see [Quick Start -- Metadata Exchange](../getting-started/quick-start#metadata-exchange).
</Tip>

### get_agent_metadata

Get the full serialized metadata of the local agent, including all registered memory regions and backend connection information.

**C++ equivalent:** [`getLocalMD`](./cpp-api#getlocalmd)

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `bytes` | Serialized agent metadata |

```python
metadata = agent.get_agent_metadata()
# Send metadata to remote agent via your preferred transport
```

### get_partial_agent_metadata

Get partial metadata containing only specified descriptors and optionally connection information. Useful for incremental metadata updates.

**C++ equivalent:** [`getLocalPartialMD`](./cpp-api#getlocalpartialmd)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `descs` | `nixlRegDList` | *(required)* | Descriptors to include; can be empty if only sending connection info |
| `inc_conn_info` | `bool` | `False` | Whether to include backend connection information |
| `backends` | `list[str]` | `[]` | Backends to consider when constructing metadata |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `bytes` | Serialized partial metadata |

```python
partial_md = agent.get_partial_agent_metadata(reg_descs, inc_conn_info=True)
```

### add_remote_agent

Add a remote agent using its serialized metadata. After this call, the local agent can initiate transfers toward the remote agent.

**C++ equivalent:** [`loadRemoteMD`](./cpp-api#loadremotemd)

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadata` | `bytes` | Serialized metadata from the remote agent |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `str` | Name of the added remote agent |

```python
remote_name = agent.add_remote_agent(remote_metadata)
```

### remove_remote_agent

Remove a remote agent. After this call, the local agent cannot initiate transfers toward that remote agent. This also disconnects the two agents.

**C++ equivalent:** [`invalidateRemoteMD`](./cpp-api#invalidateremotemd)

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `str` | Name of the remote agent to remove |

```python
agent.remove_remote_agent("old_remote_agent")
```

## Metadata -- Direct Channel

Direct-channel metadata exchange uses the NIXL listen thread to send and receive metadata over the network, or communicates through a central metadata server (e.g., etcd). The listen thread must be enabled in the agent config.

<Tip>
For a walkthrough of direct metadata exchange and etcd-based patterns, see [Quick Start -- Metadata Exchange](../getting-started/quick-start#metadata-exchange).
</Tip>

### send_local_metadata

Send all local metadata to a specific peer or to a central metadata server.

**C++ equivalent:** [`sendLocalMD`](./cpp-api#sendlocalmd)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ip_addr` | `str` | `""` | Peer IP address; empty sends to central metadata server |
| `port` | `int` | `DEFAULT_COMM_PORT` | Port for the peer; ignored when sending to central server |

```python
# Send to a specific peer
agent.send_local_metadata(ip_addr="192.168.1.10", port=12345)

# Send to central metadata server
agent.send_local_metadata()
```

### send_partial_agent_metadata

Send partial metadata (specific descriptors and optional connection info) to a peer or central server.

**C++ equivalent:** [`sendLocalPartialMD`](./cpp-api#sendlocalpartialmd)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `descs` | `nixlRegDList` | *(required)* | Descriptors to include; can be empty for connection info only |
| `inc_conn_info` | `bool` | `False` | Whether to include connection information |
| `backends` | `list[str]` | `[]` | Backends to consider |
| `ip_addr` | `str` | `""` | Peer IP address; empty sends to central server |
| `port` | `int` | `DEFAULT_COMM_PORT` | Port for the peer |
| `label` | `str` | `""` | Label for central metadata server; ignored for peer sends |

```python
agent.send_partial_agent_metadata(
    reg_descs, inc_conn_info=True, ip_addr="192.168.1.10", port=12345
)
```

### fetch_remote_metadata

Request metadata from a central metadata server or a specific peer.

**C++ equivalent:** [`fetchRemoteMD`](./cpp-api#fetchremotemd)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remote_agent` | `str` | *(required)* | Name of the remote agent to fetch metadata for |
| `ip_addr` | `str` | `""` | Peer IP address; empty uses central server |
| `port` | `int` | `DEFAULT_COMM_PORT` | Port for the peer |
| `label` | `str` | `""` | Label for central metadata server lookup |

```python
agent.fetch_remote_metadata("remote_agent", ip_addr="192.168.1.10")
```

### invalidate_local_metadata

Invalidate local metadata from a central metadata server or a specific peer.

**C++ equivalent:** [`invalidateLocalMD`](./cpp-api#invalidatelocalmd)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ip_addr` | `str` | `""` | Peer IP address; empty invalidates from central server |
| `port` | `int` | `DEFAULT_COMM_PORT` | Port for the peer |

```python
agent.invalidate_local_metadata()
```

### check_remote_metadata

Check if remote metadata for a specific agent is available. When partial metadata methods are used, you can specify which descriptors to check for.

**C++ equivalent:** [`checkRemoteMD`](./cpp-api#checkremotemd)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `str` | *(required)* | Name of the remote agent |
| `descs` | `nixlXferDList` | `None` | Specific descriptors to check for; `None` checks for any metadata |

| **Returns** | Type | Description |
|-------------|------|-------------|
| | `bool` | `True` if metadata is available, `False` otherwise |

```python
if agent.check_remote_metadata("remote_agent"):
    print("Remote metadata is available")
```
