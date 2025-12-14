# NIXL Sender-Receiver Example

## Overview

A **high-throughput streaming pattern** using NIXL with notification-based backpressure. Demonstrates circular buffer management with RDMA WRITE operations and flow control to prevent buffer overruns.

**Key Features:**
- Notification-based backpressure (sender waits if too far ahead)
- Sequence number verification (detects buffer overruns)
- Circular buffer management
- Bandwidth measurement with detailed timing breakdown
- Reusable utility functions

---

## Quick Start

### Configuration

Edit constants at the top of `nixl_sender_receiver.py`:

```python
NUM_BUFFERS = 64                   # Number of buffer slots
BUFFER_SIZE = 16 * 1024 * 1024     # 16MB per buffer
NUM_TRANSFERS = 1000               # Number of transfers to perform
BACKPRESSURE_THRESHOLD = 60        # NUM_BUFFERS - 4 (leave margin)
PROGRESS_UPDATE_INTERVAL = 16      # Send progress every N transfers
```

### Usage

```bash
# Run the example (assumes NIXL is properly installed)
cd send_recv
python3 nixl_sender_receiver.py
```

**Expected Output:**
```
[main] Starting sender-receiver test...
[receiver] Starting
[sender] Starting
...
[receiver] Bandwidth: <varies> MB/s
[receiver] ✓ No buffer overrun (0 mismatches)
[sender] Bandwidth: <varies> MB/s
[sender] Backpressure: <N> checks, <N>ms wait, max ahead: 60/64
[main] ✓ Success!
```

> **Note:** Bandwidth values vary by platform. Expect ~1-2 GB/s on shared memory, ~10-25 GB/s on RDMA hardware.

---

## Architecture Summary

### Memory Layout

**Receiver:**
```
[Buffer0: Seq(8B) + Data][Buffer1: Seq(8B) + Data]...  ← Sender WRITES here
```

**Sender:**
```
[Buffer0: Seq(8B) + Data][Buffer1: Seq(8B) + Data]...  ← Local preparation
```

**Buffer Entry Format:**
```
[Sequence Number (8 bytes)][Data (BUFFER_SIZE bytes)]
```

### Flow Control (Notification-Based Backpressure)

**Sequence Numbers:**
- Each transfer has a sequence number (0, 1, 2, ...)
- Written to buffer header before RDMA WRITE
- Receiver verifies expected sequence to detect overruns

**Backpressure:**
- Receiver sends progress notifications every N transfers
- Sender tracks how far ahead it is from receiver
- If `(sent - receiver_progress) >= THRESHOLD`, sender waits

**Sender:** Check not too far ahead → prepare data with sequence → RDMA WRITE buffer

**Receiver:** Poll for expected sequence → verify → send progress notification periodically

---

## Code Structure

### Phase 1: Setup
```python
# Create NIXL agent
receiver_agent = nixl_agent("receiver", nixl_agent_config(backends=["UCX"]))

# Allocate and register memory
memory_addr = nixl_utils.malloc_passthru(TOTAL_MEMORY_SIZE)
memory_reg_descs = receiver_agent.get_reg_descs(memory_reg_desc, "DRAM")
receiver_agent.register_memory(memory_reg_descs)
```

### Phase 2: Metadata Exchange
```python
# Publish own metadata and descriptors
publish_agent_metadata(receiver_agent, "receiver_metadata")
publish_descriptors(receiver_agent, buffers_xfer_descs, "receiver_buffers_desc")

# Retrieve remote agent
sender_name = retrieve_agent_metadata(receiver_agent, "sender_metadata",
                                     role_name="receiver")
```

### Phase 3: Transfer Preparation
```python
# Prepare reusable transfer handles (sender side)
local_buffers_prep = sender_agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_buffer_list, "DRAM")
remote_buffers_prep = sender_agent.prep_xfer_dlist(receiver_name, receiver_buffers_descs, "DRAM")

# Pre-create transfer handles for each buffer slot
for i in range(NUM_BUFFERS):
    handle = sender_agent.make_prepped_xfer("WRITE", local_buffers_prep, [i],
                                            remote_buffers_prep, [i], f"BUF_{i}".encode())
```

### Phase 4: Main Loop

**Receiver:**
```python
while transfers_received < NUM_TRANSFERS:
    buffer_idx = transfers_received % NUM_BUFFERS
    
    # Poll until expected sequence number appears
    while read_uint64(buffer_addr) != transfers_received:
        pass
    
    # Verify and process
    transfers_received += 1
    
    # Send progress notification periodically
    if transfers_received % PROGRESS_INTERVAL == 0:
        receiver_agent.send_notif(sender_name, f"P:{transfers_received}".encode())
```

**Sender:**
```python
while transfers_sent < NUM_TRANSFERS:
    # Check backpressure
    if (transfers_sent - receiver_progress) >= THRESHOLD:
        # Wait for receiver to catch up via notifications
        while (transfers_sent - receiver_progress) >= THRESHOLD:
            notifs = sender_agent.get_new_notifs()
            # Update receiver_progress from notifications...
    
    # Prepare buffer with sequence number
    buffer_idx = transfers_sent % NUM_BUFFERS
    write_uint64(buffer_addr, transfers_sent)
    
    # RDMA WRITE buffer to receiver
    sender_agent.transfer(buffer_handles[buffer_idx])
    transfers_sent += 1
```

---

## Utility Functions

Located in `../utils/`:

### `metadata_utils.py`

- **`publish_agent_metadata(agent, key)`** - Publish agent metadata to TCP server
- **`retrieve_agent_metadata(agent, key, timeout=10.0, role_name)`** - Retrieve remote agent (customizable timeout)
- **`publish_descriptors(agent, xfer_descs, key)`** - Publish serialized descriptors
- **`retrieve_descriptors(agent, key)`** - Retrieve and deserialize descriptors

### `memory_utils.py`

- **`write_uint64(addr, value)`** - Write 64-bit integer
- **`read_uint64(addr)`** - Read 64-bit integer

---

## Key NIXL Concepts

1. **Memory Registration**: `agent.register_memory(reg_descs)` before transfers
2. **Descriptor Serialization**: Share memory regions via `get_serialized_descs()`/`deserialize_descs()`
3. **Prepared Transfers**: Pre-create handles with `prep_xfer_dlist()` + `make_prepped_xfer()` for reuse
4. **RDMA WRITE**: One-sided operation, direct remote memory write
5. **Asynchronous Transfers**: `transfer()` is non-blocking, poll with `check_xfer_state()`

---

## References

- **General Guide**: `../utils/NIXL_PYTHON_GUIDE.md` - Transfer modes, polling, notifications, backpressure
- **Simple Example**: `../2proc/` - Basic two-process transfers
- **Utility Functions**: `../utils/`

---

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

