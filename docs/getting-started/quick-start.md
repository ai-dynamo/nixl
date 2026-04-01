---
title: Quick Start
description: Install NIXL and learn the complete transfer workflow -- from agent initialization to transfer completion.
---

## Install

Install via PyPI:

```bash
pip install nixl[cu12]
```

For CUDA 13:

```bash
pip install nixl[cu13]
```

<Tip>
Bare `pip install nixl` defaults to `nixl[cu12]` for backwards compatibility, so existing workflows continue to work without changes.
</Tip>

<Note>
NIXL is supported on Linux only. It is tested on Ubuntu (22.04/24.04) and Fedora. macOS and Windows are not currently supported.
</Note>

To verify the installation:

```bash
python3 -c "import nixl; agent = nixl.nixl_agent('agent1')"
```

Expected output:

```
NIXL INFO    _api.py:363 Backend UCX was instantiated
NIXL INFO    _api.py:253 Initialized NIXL agent: agent1
```

---

The sections below follow the Transfer Agent lifecycle: initialization, backend creation, memory registration, metadata exchange, transfer, and teardown.

<Note>
If you installed NIXL via pip above, you're ready to go. For building from source, see [Building NIXL from Source](/docs/user-guide/building-nixl).
</Note>

The NIXL workflow follows a strict order:

1. Create an agent
2. Create backends (C++ and Rust only -- Python auto-initializes)
3. Register memory
4. Exchange metadata with remote agents
5. Create and execute transfers
6. Check transfer status
7. Clean up resources

<Warning>
Memory must be registered before metadata exchange. Metadata exchanged before memory registration will not include the memory segment information needed for transfers.
</Warning>

## Agent Initialization

A Transfer Agent represents one endpoint in a data transfer. Create one per process with a unique name.

<CodeBlocks>
```python title="Python"
from nixl._api import nixl_agent, nixl_agent_config

config = nixl_agent_config(
    enable_prog_thread=True,
    enable_listen_thread=True,
    listen_port=5555
)
agent = nixl_agent("my_agent", config)
```

```cpp title="C++"
#include "nixl.h"

nixlAgentConfig cfg;
cfg.useProgThread = true;

nixlAgent agent("my_agent", cfg);
```

```rust title="Rust"
use nixl_sys::{Agent, MemType, XferOp};

let agent = Agent::new("my_agent")?;
```
</CodeBlocks>

<Note title="Python Auto-Initialization">
The Python API auto-initializes backends listed in `nixl_agent_config.backends`
(default: `['UCX']`). In C++ and Rust, backends must be created explicitly as
shown in the next section.
</Note>

## Backend Creation

Backends handle data transfer over specific transports. Python auto-initializes backends from the agent config. C++ and Rust require explicit creation.

<CodeBlocks>
```python title="Python"
# Backends are auto-created from nixl_agent_config.backends (default: ['UCX']).
# To add additional backends after agent creation:
agent.create_backend("GDS")
```

```cpp title="C++"
// Query backend parameters
nixl_b_params_t init_params;
nixl_mem_list_t mems;
agent.getPluginParams("UCX", mems, init_params);

// Create the backend
nixlBackendH* backend;
agent.createBackend("UCX", init_params, backend);
```

```rust title="Rust"
// Query backend parameters
let (_, params) = agent.get_plugin_params("UCX")?;

// Create the backend
let backend = agent.create_backend("UCX", &params)?;
```
</CodeBlocks>

<Tip>
You can create multiple backends on the same agent. NIXL will automatically select
the best one for each transfer based on source and destination memory types.
See [NIXL Backends](/docs/user-guide/backend-selection) for guidance on which backends to enable.
</Tip>

## Memory Registration

Register memory segments before they can participate in transfers. Registration creates the internal structures backends need for tracking and remote access metadata.

<CodeBlocks>
```python title="Python"
import torch

torch.set_default_device("cuda:0")
tensor = torch.zeros((10, 16), dtype=torch.float32)

# Register the tensor -- NIXL detects GPU memory automatically
reg_descs = agent.register_memory(tensor)
```

```cpp title="C++"
// Allocate memory
void* buf = calloc(1, 256);

// Create a descriptor for the memory region
nixlBlobDesc desc;
desc.addr = (uintptr_t)buf;
desc.len = 256;
desc.devId = 0;

nixl_reg_dlist_t reg_list(DRAM_SEG);
reg_list.addDesc(desc);

// Register with NIXL
agent.registerMem(reg_list);
```

```rust title="Rust"
// Create system storage for DRAM
let mut storage = nixl_sys::SystemStorage::new(1024)?;

// Register the memory with the agent
storage.register(&agent, None)?;
```
</CodeBlocks>

<Note title="Python Memory Type Detection">
Python automatically detects the memory type from the tensor's device. A CUDA
tensor registers as VRAM, a CPU tensor as DRAM. You can also pass raw memory
tuples `(address, size, device_id, tag)` for manual control.
</Note>

## Metadata Exchange

Transfer Agents must exchange metadata before transfers. NIXL supports three modes:

<Tabs>
<Tab title="Side-Channel (Direct)">

The simplest approach for getting started. Agents exchange metadata directly over a TCP connection. One agent listens for incoming metadata, while the other fetches and sends.

<CodeBlocks>
```python title="Python"
# On the initiator side:
# Fetch the target's metadata (target is listening on ip:port)
agent.fetch_remote_metadata("target", target_ip, target_port)

# Send this agent's metadata to the target
agent.send_local_metadata(target_ip, target_port)

# Wait for metadata to be available
while not agent.check_remote_metadata("target"):
    pass
```

```cpp title="C++"
// Get local metadata as a serialized blob
std::string meta;
agent.getLocalMD(meta);

// On the other side, load the received metadata
std::string remote_name;
agent.loadRemoteMD(meta, remote_name);
```

```rust title="Rust"
// Get local metadata as a byte vector
let md = agent.get_local_md()?;

// On the other side, load the received metadata
let remote_name = agent.load_remote_md(&md)?;
```
</CodeBlocks>

<Note>
The Python side-channel API (`send_local_metadata`/`fetch_remote_metadata`) uses
built-in TCP communication. The C++ and Rust examples above show the programmatic
approach for same-process usage. For cross-process C++/Rust with side-channel,
use the etcd mode or implement your own transport for the metadata blobs.
</Note>

</Tab>
<Tab title="etcd (Distributed)">

For centralized metadata exchange using a distributed key-value store. Agents publish metadata to an etcd server and fetch from it by agent name. Requires an etcd server running and the `NIXL_ETCD_ENDPOINTS` environment variable set.

<Note>
For etcd server setup, configuration, and key prefix scheme details, see the [Metadata Exchange with etcd](../user-guide/etcd-metadata-exchange) guide. This section
shows only the API calls for etcd-based metadata exchange.
</Note>

<CodeBlocks>
```python title="Python"
# Send this agent's metadata to ETCD
agent.send_local_metadata()

# Fetch a remote agent's metadata from ETCD
agent.fetch_remote_metadata("remote_agent_name")

# Wait for metadata to be available
while not agent.check_remote_metadata("remote_agent_name"):
    pass
```

```cpp title="C++"
// Both agents send metadata to ETCD
nixl_status_t status = agent.sendLocalMD();

// Each agent fetches the other's metadata from ETCD
status = agent.fetchRemoteMD("remote_agent_name");
```

```rust title="Rust"
// Send this agent's metadata to ETCD
agent.send_local_md(None)?;

// Fetch a remote agent's metadata from ETCD
agent.fetch_remote_md("remote_agent_name", None)?;
```
</CodeBlocks>

<Tip>
When calling `send_local_metadata()` or `fetch_remote_metadata()` in Python
without IP/port arguments, NIXL uses the etcd backend for metadata exchange.
The same methods handle both side-channel and etcd modes.
</Tip>

</Tab>
<Tab title="Programmatic">

For full control over metadata distribution. Get metadata as a serialized blob and transport it using your own mechanism (shared memory, message queue, custom RPC, etc.). This is useful when you have an existing metadata distribution system.

<CodeBlocks>
```python title="Python"
# Get this agent's metadata as bytes
metadata = agent.get_agent_metadata()

# Transport metadata using your own mechanism...

# On the other side, load the received metadata
remote_name = agent.add_remote_agent(metadata)
```

```cpp title="C++"
// Get local metadata as a serialized blob
std::string md;
agent.getLocalMD(md);

// Transport metadata using your own mechanism...

// On the other side, load the received metadata
std::string remote_name;
agent.loadRemoteMD(md, remote_name);
```

```rust title="Rust"
// Get local metadata as a byte vector
let md = agent.get_local_md()?;

// Transport metadata using your own mechanism...

// On the other side, load the received metadata
let remote_name = agent.load_remote_md(&md)?;
```
</CodeBlocks>

</Tab>
</Tabs>

## Creating and Executing Transfers

Create a transfer request, post it (non-blocking), and poll for completion.

<CodeBlocks>
```python title="Python"
# Build transfer descriptors from tensors
local_rows = [tensor[i, :] for i in range(tensor.shape[0])]
local_descs = agent.get_xfer_descs(local_rows)

# Initialize transfer (READ from target into local memory)
xfer_handle = agent.initialize_xfer(
    "READ", local_descs, target_descs, "target", b"notification_msg"
)

# Post transfer (non-blocking)
state = agent.transfer(xfer_handle)
```

```cpp title="C++"
// Create source and destination descriptor lists
nixl_xfer_dlist_t src_descs(DRAM_SEG);
nixlBasicDesc src;
src.addr = (uintptr_t)src_buf;
src.len = size;
src.devId = 0;
src_descs.addDesc(src);

nixl_xfer_dlist_t dst_descs(DRAM_SEG);
nixlBasicDesc dst;
dst.addr = (uintptr_t)dst_buf;
dst.len = size;
dst.devId = 0;
dst_descs.addDesc(dst);

// Create transfer request
nixlXferReqH* xfer_req;
nixl_opt_args_t extra_params;
extra_params.notif = "notification_msg";
agent.createXferReq(NIXL_WRITE, src_descs, dst_descs,
                    "target", xfer_req, &extra_params);

// Post transfer (non-blocking)
agent.postXferReq(xfer_req);
```

```rust title="Rust"
// Create source and destination descriptor lists
let mut src_desc = nixl_sys::XferDescList::new(MemType::Dram)?;
src_desc.add_desc(src_addr, size, 0)?;

let mut dst_desc = nixl_sys::XferDescList::new(MemType::Dram)?;
dst_desc.add_desc(dst_addr, size, 0)?;

// Create transfer request with notification
let mut opt_args = nixl_sys::OptArgs::new()?;
opt_args.set_notification_message(b"notification_msg")?;
opt_args.set_has_notification(true)?;

let xfer_req = agent.create_xfer_req(
    XferOp::Write,
    &src_desc,
    &dst_desc,
    "target",
    Some(&opt_args),
)?;

// Post transfer (non-blocking)
agent.post_xfer_req(&xfer_req, None)?;
```
</CodeBlocks>

## Checking Transfer Status

Poll transfer status after posting. The post call returns immediately.

<CodeBlocks>
```python title="Python"
# Poll for completion
while True:
    state = agent.check_xfer_state(xfer_handle)
    if state == "DONE":
        break
    elif state == "ERR":
        raise RuntimeError("Transfer failed")
    # state == "PROC" means still in progress
```

```cpp title="C++"
nixl_status_t status;
do {
    status = agent.getXferStatus(xfer_req);
} while (status == NIXL_IN_PROG);

if (status != NIXL_SUCCESS) {
    // Handle error
}
```

```rust title="Rust"
loop {
    match agent.get_xfer_status(&xfer_req)? {
        XferStatus::Success => break,
        XferStatus::InProgress => continue,
    }
}
```
</CodeBlocks>

<Warning>
Transfer status returns differ across languages. Python returns strings
(`"DONE"`, `"PROC"`, `"ERR"`). C++ returns `nixl_status_t` enum values
(`NIXL_SUCCESS`, `NIXL_IN_PROG`, negative error codes). Rust returns
`Result<XferStatus, NixlError>` with variants `Success` and `InProgress`.
Always use the language-appropriate status check.
</Warning>

## Teardown

Release resources in order: transfer handles, then memory, then remote metadata.

<CodeBlocks>
```python title="Python"
# Release transfer handle
agent.release_xfer_handle(xfer_handle)

# Deregister memory
agent.deregister_memory(reg_descs)

# Remove remote agent metadata
agent.remove_remote_agent("target")
```

```cpp title="C++"
// Release transfer request
agent.releaseXferReq(xfer_req);

// Deregister memory
agent.deregisterMem(reg_list);

// Invalidate remote metadata
agent.invalidateRemoteMD("target");
```

```rust title="Rust"
// Resources are cleaned up via RAII (Drop trait)
// Explicit cleanup is optional
drop(xfer_req);
drop(storage);
```
</CodeBlocks>

<Tip>
In Rust, resources are automatically cleaned up when they go out of scope via the
`Drop` trait. Explicit cleanup calls are optional but can be useful for controlling
the order of resource release.
</Tip>
