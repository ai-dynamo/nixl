# ETCD Python Runtime

This directory contains the ETCD-based runtime implementation that replaces the MPI Python runtime with a new Python API that uses C++ etcd functions.

## Overview

The ETCD runtime provides distributed coordination and communication using etcd as the backend instead of MPI. This includes:

- **C++ Runtime** (`etcd_rt.cpp/h`): Core ETCD-based communication primitives
- **Python Bindings** (`python_bindings.cpp`): pybind11 bindings to expose C++ functions to Python
- **Python Runtime** (`../../../kvbench/runtime/etcd_rt.py`): High-level Python API that implements the `_RTUtils` interface

## Features

- Point-to-point communication (sendInt, recvInt, sendChar, recvChar)
- Collective operations (broadcast, reduce)
- Barrier synchronization
- Distributed rank management
- Object serialization for Python data structures

## Building

### Prerequisites

- etcd server running (default: `http://localhost:2379`)
- etcd-cpp-api library
- pybind11
- Python 3.6+

### Option 1: Using Meson (Recommended)

The Python bindings are automatically built as part of the main meson build if pybind11 and Python dependencies are found.

```bash
# From the main build directory
meson setup builddir
ninja -C builddir
```

### Option 2: Using setuptools

```bash
cd /path/to/etcd
pip install pybind11
python setup.py build_ext --inplace
```

## Usage

The Python runtime provides ETCD-based distributed coordination:

### Option 1: Direct instantiation
```python
from nixl.benchmark.kvbench.runtime.etcd_rt import _EtcdDistUtils

# Create runtime instance using ETCD
runtime = _EtcdDistUtils(etcd_endpoints="http://localhost:2379", size=4)

# Use the distributed runtime API
rank = runtime.get_rank()
world_size = runtime.get_world_size()

# Collective operations
data = runtime.allgather_obj({"rank": rank, "data": "hello"})
runtime.barrier()
```

### Option 2: Using the global instance (configured via environment variables)
```python
from nixl.benchmark.kvbench.runtime import dist_rt

# Runtime is automatically configured from ETCD_ENDPOINTS and WORLD_SIZE env vars
rank = dist_rt.get_rank()
world_size = dist_rt.get_world_size()

# Collective operations
data = dist_rt.allgather_obj({"rank": rank, "data": "hello"})
dist_rt.barrier()
```

## Configuration

Environment variables:
- `ETCD_ENDPOINTS`: Comma-separated list of etcd endpoints (default: "http://localhost:2379")
- `WORLD_SIZE`: Number of distributed processes (default: 1)

## Architecture

```
Python Application
       ↓
_EtcdDistUtils (Python)
       ↓
etcd_runtime (pybind11)
       ↓
xferBenchEtcdRT (C++)
       ↓
etcd-cpp-api
       ↓
ETCD Server
```

## Limitations

1. **Data Size**: Character data transfer has practical size limits
2. **Reduce Operations**: Only SUM and AVG are fully implemented (MIN/MAX use placeholders)
3. **Error Handling**: Basic error handling with timeouts and retries
4. **Performance**: May be slower than native MPI for high-frequency operations

## Future Improvements

- Implement proper MIN/MAX reduce operations
- Add support for custom serialization protocols
- Optimize data transfer for large objects
- Add connection pooling and reconnection logic
- Support for hierarchical barriers and reductions
