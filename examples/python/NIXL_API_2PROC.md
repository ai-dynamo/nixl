# NIXL Two-Process Example

## Overview

A **basic two-process communication pattern** using NIXL demonstrating fundamental data transfer operations.

**Key Features:**
- Two transfer modes (initialize vs. prepared)
- Target-initiator pattern
- Synchronous transfer completion
- Reusable utility functions
- Simple TCP-based metadata exchange

---

## Quick Start

### Usage

```bash
# Run the example (assumes NIXL is properly installed)
python3 nixl_api_2proc.py
```

**Expected Output:**
```
[main] Starting TCP metadata server...
[main] Starting target and initiator processes...
[target] Starting target process
[initiator] Starting initiator process
[target] Memory registered
[target] Published metadata and xfer descriptors to TCP server
[target] Waiting for transfers...
[initiator] Memory registered
[initiator] Waiting for target metadata...
[initiator] Loaded remote agent: target
[initiator] Successfully retrieved target descriptors
[initiator] Starting transfer 1 (READ)...
[initiator] Initial transfer state: PROC
[initiator] Transfer 1 done
[target] Transfer 1 done
[initiator] Starting transfer 2 (WRITE)...
[initiator] Transfer 2 done
[target] Transfer 2 done
[target] Target process complete
[initiator] Initiator process complete
[main] ✓ Test Complete - Both processes succeeded!
```

---

## Architecture Summary

### Processes

**Target Process:**
- Allocates and registers 2 buffers (256 bytes each)
- Publishes metadata and descriptors to TCP server
- Waits for transfers to complete (polling `check_remote_xfer_done`)
- Passive role - data is written to/read from its buffers

**Initiator Process:**
- Allocates and registers 2 buffers (256 bytes each)
- Retrieves target's metadata and descriptors
- Performs Transfer 1: READ (using `initialize_xfer`)
- Performs Transfer 2: WRITE (using `make_prepped_xfer`)
- Active role - initiates all transfers

### Transfer Modes

**Transfer 1 - Initialize Mode:**
```python
xfer_handle_1 = initiator_agent.initialize_xfer(
    "READ", initiator_xfer_descs, target_xfer_descs, remote_name, b"UUID1"
)
state = initiator_agent.transfer(xfer_handle_1)
# Poll for completion
while initiator_agent.check_xfer_state(xfer_handle_1) != "DONE":
    time.sleep(0.001)
```

**Transfer 2 - Prepared Mode:**
```python
local_prep_handle = initiator_agent.prep_xfer_dlist(
    "NIXL_INIT_AGENT", [(initiator_addr, buf_size, 0), (initiator_addr2, buf_size, 0)], "DRAM"
)
remote_prep_handle = initiator_agent.prep_xfer_dlist(
    remote_name, target_xfer_descs, "DRAM"
)
xfer_handle_2 = initiator_agent.make_prepped_xfer(
    "WRITE", local_prep_handle, [0, 1], remote_prep_handle, [1, 0], b"UUID2"
)
```

---

## Code Structure

### Phase 1: Setup

**Target:**
```python
# Create agent with UCX backend
agent_config = nixl_agent_config(backends=["UCX"])
target_agent = nixl_agent("target", agent_config)

# Allocate memory (2 buffers, 256 bytes each)
target_addr = nixl_utils.malloc_passthru(buf_size * 2)
target_addr2 = target_addr + buf_size

# Create descriptors (4-tuple for registration, 3-tuple for transfer)
target_reg_descs = target_agent.get_reg_descs(target_strings, "DRAM")
target_xfer_descs = target_agent.get_xfer_descs(target_addrs, "DRAM")

# Register with NIXL
target_agent.register_memory(target_reg_descs)
```

**Initiator:**
```python
# Create agent (uses default config)
initiator_agent = nixl_agent("initiator", None)
# Similar allocation and registration...
```

### Phase 2: Metadata Exchange

```python
# Target: Publish
publish_agent_metadata(target_agent, "target_meta")
publish_descriptors(target_agent, target_xfer_descs, "target_descs")

# Initiator: Retrieve
remote_name = retrieve_agent_metadata(initiator_agent, "target_meta",
                                     timeout=10.0, role_name="initiator")
target_xfer_descs = retrieve_descriptors(initiator_agent, "target_descs")
```

### Phase 3: Transfers

**Transfer 1 - Simple approach:**
- Use `initialize_xfer()` for one-time transfers
- Simpler API, creates transfer on-the-fly
- Good for occasional transfers

**Transfer 2 - Optimized approach:**
- Use `prep_xfer_dlist()` + `make_prepped_xfer()`
- Pre-creates reusable transfer handles
- Better for repeated transfers

### Phase 4: Synchronization

**Target waits for completion:**
```python
while not target_agent.check_remote_xfer_done("initiator", b"UUID1"):
    time.sleep(0.001)
```

**Initiator polls transfer state:**
```python
while initiator_agent.check_xfer_state(xfer_handle_1) != "DONE":
    time.sleep(0.001)
```

---

## Utility Functions

### From `nixl_metadata_utils.py`

- **`publish_agent_metadata(agent, key)`** - Publish agent metadata to TCP server
- **`retrieve_agent_metadata(agent, key, timeout=10.0, role_name)`** - Retrieve remote agent (customizable timeout)
- **`publish_descriptors(agent, xfer_descs, key)`** - Publish serialized descriptors
- **`retrieve_descriptors(agent, key)`** - Retrieve and deserialize descriptors

### From `tcp_server.py`

- Simple key-value store for metadata exchange
- Only used during setup phase
- Not involved in actual data transfers

---

## Key NIXL Concepts

1. **Memory Registration**: `agent.register_memory(reg_descs)` before transfers
2. **Agent Metadata**: Exchange via `get_agent_metadata()` and `add_remote_agent()`
3. **Descriptor Types**:
   - **Registration descriptors**: 4-tuple `(addr, size, dev_id, name)` for memory registration
   - **Transfer descriptors**: 3-tuple `(addr, size, dev_id)` for actual transfers
4. **Transfer Modes**:
   - **Initialize mode**: `initialize_xfer()` - simple, one-shot
   - **Prepared mode**: `prep_xfer_dlist()` + `make_prepped_xfer()` - optimized, reusable
5. **Transfer Operations**:
   - **READ**: Initiator reads from target's memory
   - **WRITE**: Initiator writes to target's memory
6. **Synchronization**:
   - **Local**: `check_xfer_state()` - check local transfer status
   - **Remote**: `check_remote_xfer_done()` - check if remote agent completed transfer

---

## Comparison: Initialize vs. Prepared Transfer

| Aspect | Initialize Mode | Prepared Mode |
|--------|----------------|---------------|
| **API** | `initialize_xfer()` | `prep_xfer_dlist()` + `make_prepped_xfer()` |
| **Setup** | Simple, one call | More complex, two-step |
| **Reusability** | One-time use | Reusable handles |
| **Performance** | Good for occasional | Better for repeated |
| **Use Case** | Simple transfers | High-frequency transfers |
| **Example** | Transfer 1 in code | Transfer 2 in code |

---

## References

- **General Guide**: `NIXL_PYTHON_GUIDE.md` - Transfer modes, polling, notifications
- **Advanced Example**: `nixl_sender_receiver.py` - Streaming with backpressure
- **Utility Functions**: `nixl_metadata_utils.py`, `nixl_memory_utils.py`

---

## Detailed Architecture Diagrams

### Setup and Metadata Exchange

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SETUP PHASE                                        │
└─────────────────────────────────────────────────────────────────────────────┘

 TCP Metadata Server         Target Process                Initiator Process
      (Port 9998)            ┌──────────────┐             ┌──────────────┐
           │                 │              │             │              │
           │                 │ 1. Create    │             │ 1. Create    │
           │                 │    Agent     │             │    Agent     │
           │                 │    (UCX)     │             │    (default) │
           │                 │              │             │              │
           │                 │ 2. Allocate  │             │ 2. Allocate  │
           │                 │    2 buffers │             │    2 buffers │
           │                 │    (256B ea) │             │    (256B ea) │
           │                 │              │             │              │
           │                 │ 3. Register  │             │ 3. Register  │
           │                 │    Memory    │             │    Memory    │
           │                 │              │             │              │
           │◄────(publish)───┤              │             │              │
           │   "target_meta" │              │             │              │
           │  + descriptors  │              │             │              │
           │                 │              │             │              │
           │                 │ 4. Wait for  │             │              │
           │                 │    transfers │             │              │
           │                 │              │             │              │
           │─(retrieve)──────────────────────────────────►│ 4. Retrieve  │
           │             "target_meta" + descriptors      │    metadata  │
           │                 │              │             │              │
           │                 │              │             │ 5. Add       │
           │                 │              │             │    remote    │
           │                 │              │             │    agent     │
           │                 └──────────────┘             └──────┬───────┘
           │                                                     │
           │                        Connection Established       │
           │                                                     │
 (TCP server only used for metadata exchange, not for data transfers)
```

### Transfer Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRANSFER PHASE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

     Target                                                      Initiator
     (Passive)                                                   (Active)
        │                                                            │
        │  Waiting for transfers...                                  │
        │                                                            │
        │                                                            │
        │                                            ┌───────────────┴──────────┐
        │                                            │ Transfer 1: READ         │
        │                                            │ Mode: initialize_xfer    │
        │                                            └───────────────┬──────────┘
        │                                                            │
        │◄──────────────────────RDMA READ (256B)─────────────────────┤
        │  Initiator reads from target's buffer                      │
        │                                                            │
        │  check_remote_xfer_done("initiator", "UUID1")              │
        │  → TRUE                                                    │
        │                                                            │
        │  Transfer 1 done ───────────────────────────────────────► check_xfer_state()
        │                                                            → DONE
        │                                                            │
        │                                                            │
        │                                            ┌───────────────┴──────────┐
        │                                            │ Transfer 2: WRITE        │
        │                                            │ Mode: make_prepped_xfer  │
        │                                            └───────────────┬──────────┘
        │                                                            │
        │◄──────────────────────RDMA WRITE (512B)────────────────────┤
        │  Initiator writes to target's buffer                       │
        │  (crossing: writes buf2→buf1, buf1→buf2)                   │
        │                                                            │
        │  check_remote_xfer_done("initiator", "UUID2")              │
        │  → TRUE                                                    │
        │                                                            │
        │  Transfer 2 done ───────────────────────────────────────► check_xfer_state()
        │                                                            → DONE
        │                                                            │
        │  Cleanup & exit ◄───────────────────────────────────────► Cleanup & exit
        │                                                            │
```

---

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

