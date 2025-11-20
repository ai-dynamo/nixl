# NIXL Sender-Receiver Example

## Overview

A **queue-based producer-consumer pattern** using NIXL with head/tail pointer flow control. Demonstrates circular buffer management with RDMA WRITE operations for high-throughput streaming.

**Key Features:**
- Queue-based flow control (head/tail pointers)
- RDMA WRITE for data and control
- Circular buffer management
- Bandwidth measurement
- Reusable utility functions

---

## Quick Start

### Configuration

Edit constants at the top of `nixl_sender_receiver.py`:

```python
NUM_BUFFERS = 2                    # Queue size (2 optimal for point-to-point)
BUFFER_SIZE = 16 * 1024 * 1024     # 16MB per buffer
NUM_TRANSFERS = 100                # Number of transfers to perform
```

### Usage

```bash
# Run the example (assumes NIXL is properly installed)
python3 nixl_sender_receiver.py
```

**Expected Output:**
```
[main] Starting sender-receiver: queue_size=2, num_transfers=100, buffer_size=16777216
[receiver] Starting
[sender] Starting
...
[sender] Completed 100 transfers in 1.467s
[sender] Bandwidth: 1091.01 MB/s
[receiver] Completed 100 transfers in 1.447s
[receiver] Bandwidth: 1105.43 MB/s
[main] ✓ Success!
```

---

## Architecture Summary

### Memory Layout

**Receiver:**
```
[Tail(8B)][Buffer0][Buffer1]...  ← Sender WRITES data here
[Head(8B)]                       ← Receiver WRITES to sender
```

**Sender:**
```
[Tail(8B)]                       ← Local update only
[Buffer0][Buffer1]...            ← Local data preparation
[Head(8B)]                       ← Receiver WRITES here
```

### Flow Control

**Queue States:**
- Empty: `Tail == Head`
- Full: `(Tail + 1) % NUM_BUFFERS == Head`
- Buffer index: `Tail % NUM_BUFFERS` (sender), `Head % NUM_BUFFERS` (receiver)

**Sender:** Check queue not full → prepare data → RDMA WRITE data → update tail → RDMA WRITE tail

**Receiver:** Check queue not empty → process data → update head → RDMA WRITE head

---

## Code Structure

### Phase 1: Setup (lines 50-88, 208-241)
```python
# Create NIXL agent
agent = nixl_agent("receiver", nixl_agent_config(backends=["UCX"]))

# Allocate and register memory
tail_and_buffers_addr = nixl_utils.malloc_passthru(8 + NUM_BUFFERS * BUFFER_SIZE)
head_addr = nixl_utils.malloc_passthru(8)
agent.register_memory(reg_descs)
```

### Phase 2: Metadata Exchange (lines 90-102, 243-255)
```python
# Publish own metadata and descriptors
publish_agent_metadata(agent, "receiver_meta")
publish_descriptors(agent, tail_descs, "receiver_tail_desc")
publish_descriptors(agent, head_descs, "receiver_head_desc")

# Retrieve remote agent
remote_name = retrieve_agent_metadata(agent, "sender_meta",
                                     timeout=10.0, role_name="receiver")
sender_descs = retrieve_descriptors(agent, "sender_tail_desc")
```

### Phase 3: Transfer Preparation (lines 106-115, 259-290)
```python
# Prepare reusable transfer handles
local_prep = agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_list, "DRAM")
remote_prep = agent.prep_xfer_dlist(remote_name, remote_descs, "DRAM")
xfer_handle = agent.make_prepped_xfer("WRITE", local_prep, [0],
                                      remote_prep, [0], b"UUID")
```

### Phase 4: Main Loop (lines 134-177, 317-371)
```python
# Receiver: consume queue
while transfers_received < NUM_TRANSFERS:
    remote_tail = read_uint64(local_tail_addr)
    if remote_tail != local_head:  # Not empty
        process_buffer(local_head % NUM_BUFFERS)
        local_head = (local_head + 1) % NUM_BUFFERS
        write_uint64(head_addr, local_head)
        agent.transfer(head_xfer_handle)  # RDMA WRITE head

# Sender: fill queue
while transfers_sent < NUM_TRANSFERS:
    remote_head = read_uint64(head_addr)
    if (local_tail + 1) % NUM_BUFFERS != remote_head:  # Not full
        prepare_buffer(local_tail % NUM_BUFFERS)
        agent.transfer(buffer_xfer_handles[local_tail % NUM_BUFFERS])
        local_tail = (local_tail + 1) % NUM_BUFFERS
        write_uint64(tail_addr, local_tail)
        agent.transfer(tail_xfer_handle)  # RDMA WRITE tail
```

---

## Utility Functions

### `nixl_metadata_utils.py`

- **`publish_agent_metadata(agent, key)`** - Publish agent metadata to TCP server
- **`retrieve_agent_metadata(agent, key, timeout=10.0, role_name)`** - Retrieve remote agent (customizable timeout)
- **`publish_descriptors(agent, xfer_descs, key)`** - Publish serialized descriptors
- **`retrieve_descriptors(agent, key)`** - Retrieve and deserialize descriptors

### `nixl_memory_utils.py`

- **`write_uint64(addr, value)`** - Write 64-bit integer
- **`read_uint64(addr)`** - Read 64-bit integer
- **`write_data(addr, data)`** - Write NumPy array/bytes
- **`read_data(addr, size)`** - Read data as NumPy array

---


## Key NIXL Concepts

1. **Memory Registration**: `agent.register_memory(reg_descs)` before transfers
2. **Descriptor Serialization**: Share memory regions via `get_serialized_descs()`/`deserialize_descs()`
3. **Prepared Transfers**: Pre-create handles with `prep_xfer_dlist()` + `make_prepped_xfer()` for reuse
4. **RDMA WRITE**: One-sided operation, direct remote memory write
5. **Asynchronous Transfers**: `transfer()` is non-blocking, poll with `check_xfer_state()`

---

---

## References

- **Simple Example**: `nixl_api_2proc.py` - Basic two-process transfers
- **Utility Functions**: `nixl_metadata_utils.py`, `nixl_memory_utils.py`
- **NIXL Examples**: `nixl_api_example.py`

---

## Detailed Architecture Diagrams

### Setup Phase

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SETUP PHASE                                        │
└─────────────────────────────────────────────────────────────────────────────┘

 TCP Metadata Server         Receiver Process              Sender Process
      (Port 9998)            ┌──────────────┐             ┌──────────────┐
           │                 │              │             │              │
           │                 │ 1. Create    │             │ 1. Create    │
           │                 │    NIXL      │             │    NIXL      │
           │                 │    Agent     │             │    Agent     │
           │                 │              │             │              │
           │                 │ 2. Allocate: │             │ 2. Allocate: │
           │                 │    • Tail +  │             │    • Tail    │
           │                 │      Buffers │             │    • Head    │
           │                 │    • Head    │             │    • Buffers │
           │                 │              │             │              │
           │◄────(publish)───┤              │             │              │
           │   "receiver_    │              │             │              │
           │    metadata"    │              │             │              │
           │  + descriptors  │              │             │              │
           │                 │              │             │              │
           │◄────────────────────────────(publish)────────┤              │
           │                 "sender_metadata"            │              │
           │                  + descriptors               │              │
           │                 │              │             │              │
           │─(retrieve)──────►              │             │              │
           │ "sender_meta"   │              │             │              │
           │  + descriptors  │              │             │              │
           │                 │              │             │              │
           │─(retrieve)──────────────────────────────────►│              │
           │             "receiver_meta" + descriptors    │              │
           │                 │              │             │              │
           │                 │ 3. Add       │             │ 3. Add       │
           │                 │    Remote    │             │    Remote    │
           │                 │    Agent     │             │    Agent     │
           │                 └──────┬───────┘             └──────┬───────┘
           │                        │                            │
           │                        └──────NIXL Connection───────┘
           │                               Established!
           │
 (TCP server only used for initial metadata/descriptor exchange)
```

### Memory Layout Details

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MEMORY LAYOUT                                          │
└─────────────────────────────────────────────────────────────────────────────┘

Receiver's Memory:
┌────────────┬──────────────┬──────────────┐
│ Tail (8B)  │  Buffer 0    │  Buffer 1    │  ← Sender WRITES here
└────────────┴──────────────┴──────────────┘
     ▲
     └─ Sender updates this via RDMA WRITE

┌────────────┐
│ Head (8B)  │  ← Receiver updates locally, WRITES to sender
└────────────┘

Sender's Memory:
┌────────────┐
│ Tail (8B)  │  ← Sender updates locally
└────────────┘

┌────────────┬──────────────┬──────────────┐
│ Buffers    │  Buffer 0    │  Buffer 1    │  ← Sender fills locally
└────────────┴──────────────┴──────────────┘

┌────────────┐
│ Head (8B)  │  ← Receiver WRITES here
└────────────┘
```

### Main Transfer Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MAIN TRANSFER LOOP (Queue-Based Flow Control)             │
└─────────────────────────────────────────────────────────────────────────────┘

   Receiver                                                         Sender
   (Consumer)                                                    (Producer)
      │                                                               │
      │   Initialize: Head = 0, Tail = 0 (queue empty)                │
      │                                                               │
      │◄───────────RDMA WRITE: Head = 0 (initial sync)────────────────┤
      │                                                               │
      │                                                               │
      │                                            Check: Tail+1 != Head?
      │                                            (Queue not full)   │
      │                                                               │
      │                                            Prepare Buffer 0   │
      │                                            (Header: ID=0)     │
      │                                                               │
      │◄───────────RDMA WRITE: Data → receiver.buffer[0]──────────────┤
      │                                                               │
      │◄───────────RDMA WRITE: Tail = 1 ──────────────────────────────┤
      │                                                               │
 Read local Tail                                                      │
 (Tail=1 != Head=0)                                                   │
 Queue not empty!                                                     │
      │                                                               │
 Process buffer 0                                                     │
 Verify ID = 0                                                        │
      │                                                               │
 Update: Head = 1                                                     │
      │                                                               │
      │────────────RDMA WRITE: Head = 1 ─────────────────────────────►│
      │                                                               │
      │                                            Read remote Head   │
      │                                            (Head updated!)    │
      │                                                               │
      │                                            Check: Tail+1 != Head?
      │                                            (Queue not full)   │
      │                                                               │
      │                                            Prepare Buffer 1   │
      │                                                               │
      │◄──────────RDMA WRITE: Data → receiver.buffer[1]───────────────┤
      │◄──────────RDMA WRITE: Tail = 0 (wrapped)──────────────────────┤
      │                                                               │
 Read local Tail                                                      │
 (Tail=0 != Head=1)                                                   │
      │                                                               │
 Process buffer 1                                                     │
 Update: Head = 0                                                     │
      │                                                               │
      │─────────RDMA WRITE: Head = 0 ────────────────────────────────►│
      │                                                               │
      │  ... Continue circular queue operation ...                    │
      │                                                               │
```

---

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
