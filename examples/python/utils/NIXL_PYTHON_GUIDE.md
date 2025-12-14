# NIXL Python API Guide

A practical guide to common NIXL Python API patterns and best practices.

---

## Table of Contents

1. [Transfer Modes](#transfer-modes)
2. [Polling for Transfer Status](#polling-for-transfer-status)
3. [Using Notifications](#using-notifications)
4. [Backpressure and Flow Control](#backpressure-and-flow-control)
5. [Metadata Exchange](#metadata-exchange)

---

## Transfer Modes

NIXL provides two ways to create and execute transfers:

### Mode 1: `initialize_xfer()` + `transfer()`

**Simple, one-shot transfers.**

```python
# Create transfer handle (combines local + remote descriptors)
xfer_handle = agent.initialize_xfer(
    "WRITE",           # or "READ"
    local_xfer_descs,  # List of (addr, size, dev_id)
    remote_xfer_descs, # List of (addr, size, dev_id)
    remote_agent_name, # Name of remote agent
    b"optional_tag"    # Notification tag (optional)
)

# Execute transfer
state = agent.transfer(xfer_handle)
```

**When to use:**
- One-time transfers
- Simple setup with minimal code
- Dynamic descriptors that change each time
- Prototyping and testing

### Mode 2: `prep_xfer_dlist()` + `make_prepped_xfer()` + `transfer()`

**Optimized, reusable transfers.**

```python
# Step 1: Prepare descriptor lists (do once)
local_prep = agent.prep_xfer_dlist(
    "NIXL_INIT_AGENT",                              # Special name for local agent
    [(addr1, size1, 0), (addr2, size2, 0)],         # List of descriptors (addr, size, device_id)
    "DRAM"                                          # Memory type
)
remote_prep = agent.prep_xfer_dlist(
    remote_agent_name,
    remote_xfer_descs,
    "DRAM"
)

# Step 2: Create transfer handle with index selection
xfer_handle = agent.make_prepped_xfer(
    "WRITE",           # or "READ"
    local_prep,        # Prepared local list
    [0],               # Indices to use from local list
    remote_prep,       # Prepared remote list
    [0],               # Indices to use from remote list
    b"optional_tag"    # Notification tag (optional)
)

# Step 3: Execute (can reuse handle multiple times)
state = agent.transfer(xfer_handle)
```

**When to use:**
- Repeated transfers to same memory regions
- High-throughput streaming
- Need to select specific descriptors per transfer
- Performance-critical applications

### Comparison Table

| Aspect | `initialize_xfer()` | `make_prepped_xfer()` |
|--------|---------------------|----------------------|
| **Setup complexity** | Simple, one call | Two-step setup |
| **Reusability** | One-time use | Reusable handles |
| **Performance** | Good for occasional | Better for repeated |
| **Index selection** | All descriptors | Flexible per-transfer |
| **Use case** | Simple transfers | High-frequency streaming |

---

## Polling for Transfer Status

Transfers in NIXL are **asynchronous**. After calling `transfer()`, you need to poll for completion.

### Transfer States

| State | Meaning |
|-------|---------|
| `"DONE"` | Transfer completed successfully |
| `"PROC"` | Transfer in progress (processing) |
| `"ERR"` | Transfer failed |

### Basic Polling Pattern

```python
# Start transfer (non-blocking)
state = agent.transfer(xfer_handle)

# Poll until completion
while True:
    state = agent.check_xfer_state(xfer_handle)
    if state == "DONE":
        break  # Success!
    elif state == "ERR":
        raise RuntimeError("Transfer failed")
    # state == "PROC" means still in progress
    time.sleep(0.001)  # Small sleep to reduce CPU usage
```

### Fire-and-Forget Pattern (High Throughput)

For streaming, don't wait after each transfer. Only wait when reusing a buffer:

```python
# Pre-create handles for each buffer slot
buffer_handles = [
    agent.make_prepped_xfer("WRITE", local_prep, [i], remote_prep, [i], f"BUF_{i}".encode())
    for i in range(NUM_BUFFERS)
]

# Main loop - overlap transfers
for i in range(NUM_TRANSFERS):
    buffer_idx = i % NUM_BUFFERS
    handle = buffer_handles[buffer_idx]
    
    # Only wait if this buffer's previous transfer is still active
    while agent.check_xfer_state(handle) == "PROC":
        pass  # Spin-wait (or use time.sleep for lower CPU)
    
    # Prepare data in buffer...
    
    # Start transfer (don't wait)
    agent.transfer(handle)
```

### Remote Transfer Completion Check

The **passive side** (target) can check if the active side completed a transfer:

```python
# Initiator: Create transfer with a notification tag
xfer_handle = initiator_agent.initialize_xfer(
    "READ", local_descs, remote_descs, "target", b"XFER_1"  # <-- b"XFER_1" is the notif_msg tag
)
initiator_agent.transfer(xfer_handle)

# Target: Wait for initiator to complete that tagged transfer
while not target_agent.check_remote_xfer_done("initiator", b"XFER_1"):
    time.sleep(0.001)
print("Initiator finished the transfer!")
```

The `notif_msg` tag (e.g., `b"XFER_1"`) is the last parameter in `initialize_xfer()` or `make_prepped_xfer()`.
When the transfer completes, this tag is automatically sent to the remote side.

---

## Using Notifications

Notifications are a **lightweight messaging system** between agents, separate from data transfers.

### Sending Notifications

```python
# Send a message to a remote agent
agent.send_notif(remote_agent_name, b"message_bytes")

# Examples
agent.send_notif("receiver", b"READY")
agent.send_notif("sender", f"PROGRESS:{count}".encode())
agent.send_notif("peer", b"SHUTDOWN")
```

### Receiving Notifications

```python
# Get all new notifications (returns dict: agent_name -> list of messages)
notifs = agent.get_new_notifs()

# Process notifications from a specific agent
if "remote_agent" in notifs:
    for msg in notifs["remote_agent"]:
        print(f"Received: {msg}")

# Example: Parse progress updates
if "receiver" in notifs:
    for msg in notifs["receiver"]:
        if msg.startswith(b"P:"):
            progress = int(msg[2:])
            print(f"Receiver at: {progress}")
```

### Check for Specific Notification

```python
# Check if a specific notification arrived (removes it if found)
if agent.check_notif("remote_agent", b"ACK"):
    print("ACK received!")

# Check with prefix matching
if agent.check_notif("remote_agent", b"ERROR", tag_is_prefix=True):
    print("Got an error notification")
```

### Notifications vs. Transfer Tags

| Feature | Notifications | Transfer Tags |
|---------|--------------|---------------|
| **API** | `send_notif()` / `get_new_notifs()` | `notif_msg` param in `transfer()` |
| **Timing** | Sent immediately | Sent after transfer completes |
| **Purpose** | General messaging, flow control | Transfer completion signal |
| **Receiver API** | `get_new_notifs()` | `check_remote_xfer_done()` |

---

## Backpressure and Flow Control

When streaming data, the sender may be faster than the receiver. **Backpressure** prevents buffer overruns.

### The Problem

```
Sender: [====>]  (fast)
Receiver: [=>]   (slow)

Without backpressure: Sender overwrites buffers before receiver reads them!
```

### Solution: Notification-Based Backpressure

**Receiver** periodically sends progress updates:

```python
PROGRESS_INTERVAL = 16  # Send update every 16 messages

while transfers_received < NUM_TRANSFERS:
    # ... receive and process data ...
    
    transfers_received += 1
    
    # Send progress update periodically
    if transfers_received % PROGRESS_INTERVAL == 0:
        agent.send_notif(sender_name, f"P:{transfers_received}".encode())
```

**Sender** checks progress and waits if too far ahead:

```python
BACKPRESSURE_THRESHOLD = NUM_BUFFERS - 4  # Leave 4 buffer margin
receiver_progress = 0

while transfers_sent < NUM_TRANSFERS:
    # Check for progress notifications
    notifs = agent.get_new_notifs()
    if receiver_name in notifs:
        for msg in notifs[receiver_name]:
            if msg.startswith(b"P:"):
                receiver_progress = int(msg[2:])
    
    # Apply backpressure if too far ahead
    if (transfers_sent - receiver_progress) >= BACKPRESSURE_THRESHOLD:
        # Wait for receiver to catch up
        while (transfers_sent - receiver_progress) >= BACKPRESSURE_THRESHOLD:
            notifs = agent.get_new_notifs()
            if receiver_name in notifs:
                for msg in notifs[receiver_name]:
                    if msg.startswith(b"P:"):
                        receiver_progress = int(msg[2:])
            time.sleep(0.0001)  # Brief sleep
    
    # Safe to send
    agent.transfer(buffer_handles[transfers_sent % NUM_BUFFERS])
    transfers_sent += 1
```

### Verifying No Buffer Overrun

Add sequence numbers to detect overwrites:

```python
# Sender: Write sequence number to buffer header
write_uint64(buffer_addr, sequence_number)

# Receiver: Verify sequence matches expected
seq = read_uint64(buffer_addr)
if seq != expected_sequence:
    print(f"OVERRUN! Expected {expected_sequence}, got {seq}")
```

### Backpressure Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `NUM_BUFFERS` | Total buffer slots | 64 |
| `BACKPRESSURE_THRESHOLD` | Max ahead before waiting | NUM_BUFFERS - 4 |
| `PROGRESS_INTERVAL` | How often receiver reports | NUM_BUFFERS / 4 |

---

## Metadata Exchange

NIXL examples support two methods for metadata exchange:

### Method 1: TCP Server (Default)

A simple TCP key-value server for local testing. Used by default in the examples.

```python
from utils import (
    publish_agent_metadata,
    retrieve_agent_metadata,
    publish_descriptors,
    retrieve_descriptors,
    start_server,
)

# Start TCP server (usually in main process)
start_server(9998)

# Publish metadata
publish_agent_metadata(agent, "my_metadata_key")
publish_descriptors(agent, xfer_descs, "my_descriptors_key")

# Retrieve from another process
remote_name = retrieve_agent_metadata(agent, "remote_metadata_key")
remote_descs = retrieve_descriptors(agent, "remote_descriptors_key")
```

### Method 2: NIXL Built-in (etcd)

NIXL has built-in support for metadata exchange via etcd. To use:

1. Set the `NIXL_ETCD_ENDPOINTS` environment variable:
   ```bash
   export NIXL_ETCD_ENDPOINTS="http://localhost:2379"
   ```

2. Use the `use_nixl_builtin=True` parameter:
   ```python
   # Publish
   publish_agent_metadata(agent, "my_label", use_nixl_builtin=True)
   
   # Retrieve (must specify remote agent name)
   retrieve_agent_metadata(
       agent, "remote_label",
       use_nixl_builtin=True,
       remote_agent_name="remote_agent"
   )
   ```

### Comparison

| Feature | TCP Server | NIXL Built-in (etcd) |
|---------|-----------|---------------------|
| **Setup** | Start server in code | External etcd service |
| **Scalability** | Single node | Distributed |
| **Use case** | Local testing | Production deployments |
| **Dependencies** | None | etcd cluster |

---

## Related Examples

| Example | Description | Location |
|---------|-------------|----------|
| `nixl_api_2proc.py` | Basic two-process transfers (both modes) | `../2proc/` |
| `nixl_sender_receiver.py` | High-throughput streaming with backpressure | `../send_recv/` |

---

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

