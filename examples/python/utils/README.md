# NIXL Python Examples - Shared Utilities

This package provides reusable utilities for NIXL Python examples.

## Overview

| Module | Purpose |
|--------|---------|
| `memory_utils.py` | Memory read/write helpers for raw memory operations |
| `metadata_utils.py` | Metadata exchange (TCP server and NIXL built-in etcd) |
| `tcp_server.py` | Simple TCP-based key-value store for local testing |

## Quick Start

```python
from utils import (
    # Memory utilities
    read_uint64, write_uint64,
    # Metadata utilities
    publish_agent_metadata, retrieve_agent_metadata,
    publish_descriptors, retrieve_descriptors,
    # TCP server
    start_server, set_metadata, get_metadata, clear_metadata,
)
```

## Module Details

### memory_utils.py

Low-level memory operations using NumPy for safe pointer access.

```python
from utils import write_uint64, read_uint64

# Write a 64-bit value to a memory address
write_uint64(addr, 0xDEADBEEF)

# Read a 64-bit value from a memory address
value = read_uint64(addr)
```

**Functions:**
- `write_uint64(addr, value)` - Write a 64-bit unsigned integer to memory
- `read_uint64(addr)` - Read a 64-bit unsigned integer from memory

### metadata_utils.py

Handles metadata exchange between NIXL agents. Supports two modes:

1. **TCP Server Mode** (default): Uses a simple local TCP server for development/testing
2. **NIXL Built-in Mode**: Uses NIXL's native etcd APIs for production/distributed deployments

```python
from utils import publish_agent_metadata, retrieve_agent_metadata

# --- TCP Server Mode (default) ---
# Publish metadata
publish_agent_metadata(agent, "my_agent_key")

# Retrieve remote metadata
remote_name = retrieve_agent_metadata(agent, "remote_agent_key")

# --- NIXL Built-in Mode (etcd) ---
# Requires: export NIXL_ETCD_ENDPOINTS="http://localhost:2379"

# Publish with etcd
publish_agent_metadata(
    agent, "my_label",
    use_nixl_builtin=True,
    reg_descs=my_registration_descs
)

# Retrieve with etcd
remote_name = retrieve_agent_metadata(
    agent, "remote_label",
    use_nixl_builtin=True,
    remote_agent_name="remote_agent"
)
```

**Functions:**
- `publish_agent_metadata(agent, key, ...)` - Publish agent metadata for discovery
- `retrieve_agent_metadata(agent, key, ...)` - Retrieve and add remote agent metadata
- `publish_descriptors(agent, xfer_descs, key, ...)` - Publish transfer descriptors (TCP only)
- `retrieve_descriptors(agent, key, ...)` - Retrieve transfer descriptors (TCP only)

### tcp_server.py

A lightweight TCP-based key-value store for metadata exchange in local testing scenarios.

```python
from utils import start_server, set_metadata, get_metadata, clear_metadata

# Start the server (singleton, called once per process)
start_server(port=9998)

# Store a value
set_metadata("key", "value")

# Retrieve a value
value = get_metadata("key")

# Clear all metadata
clear_metadata()
```

**Functions:**
- `start_server(port=9998)` - Start the metadata server (singleton pattern)
- `set_metadata(key, value, ...)` - Store a key-value pair
- `get_metadata(key, ...)` - Retrieve a value by key
- `clear_metadata(...)` - Clear all stored metadata

## Metadata Exchange Modes

### TCP Server Mode

Best for:
- Local development and testing
- Single-machine multi-process examples
- Quick prototyping

```bash
# No special setup required - server starts automatically
python3 my_example.py
```

### NIXL Built-in Mode (etcd)

Best for:
- Production deployments
- Distributed multi-node setups
- Persistent metadata storage

```bash
# Ensure etcd is running
sudo systemctl start etcd

# Set the etcd endpoint
export NIXL_ETCD_ENDPOINTS="http://localhost:2379"

# Run with --use-etcd flag
python3 my_example.py --use-etcd
```

## Additional Documentation

For comprehensive NIXL Python API patterns and best practices, see:

📖 **[NIXL_PYTHON_GUIDE.md](NIXL_PYTHON_GUIDE.md)** - Complete API guide with examples

Topics covered:
- Agent lifecycle management
- Memory registration patterns
- Transfer operations
- Notification system
- Backpressure handling
- Error handling

