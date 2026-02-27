# Pure C++ Device API Example

This is a simplified, pure C++ version of the 2proc device API example. Unlike the Python version with multiple abstraction layers (pybind11 → device_host.cpp → device_buffer.py → example.py), this is a single self-contained C++ program.

## Architecture

```
simple_write.cpp        # Main program
├── nixlAgent           # Direct NIXL C++ API usage
├── CUDA                # Direct cudaMalloc/cudaMemcpy
├── Device kernels      # GPU-initiated RDMA (kernels.cu)
└── PyTorch TCPStore    # Distributed metadata exchange (same as EP)
```

**Key simplifications:**
- No Python layers
- No pybind11 bindings
- Direct nixlAgent C++ API (like EP framework uses)
- Direct CUDA runtime API
- Single executable with `--mode target|initiator` flag

## Build

From NIXL repository root:

```bash
# Configure and build
meson setup build/ -Ducx_path=/usr -Dlibfabric_path=/usr/local
ninja -C build
ninja -C build install
ldconfig

# Binary location
build/examples/device/2proc/simple_write
```

## Run

```bash
# Set UCX configuration for container compatibility
export UCX_POSIX_USE_PROC_LINK=n

# Terminal 1 (initiator - must start FIRST to create TCPStore server)
./build/examples/device/2proc/simple_write --mode initiator

# Terminal 2 (target - start SECOND after initiator's server is ready)
./build/examples/device/2proc/simple_write --mode target
```

**UCX Configuration:**

Key setting for container compatibility:

- **`UCX_POSIX_USE_PROC_LINK=n`** (Disable procfs mode - **important for containers!**)
  - **Without this**: UCX tries to access `/proc/<other_pid>/fd/<fd>` for shared memory
  - **Result**: "Permission denied" errors in containers (containers block cross-process /proc access)
  - **With this setting**: UCX uses file paths instead (e.g., `/dev/shm/ucx_shm_posix_...`)
  - **Benefit**: Clean output, no permission errors

**Optional settings:**
- `export UCX_TLS=all` - Explicitly enable all transports (default, but useful for documentation)
- `unset UCX_NET_DEVICES` - Required for CUDA IPC in single-process scenarios

**Running the example:**
- Initiator creates TCPStore server (port 9998), target connects as client
- Start initiator first, then target
- Metadata exchanged via PyTorch TCPStore (same pattern as EP framework)

## Command Line Options

```
--mode <initiator|target>  Run as initiator or target (default: initiator)
--size <bytes>             Buffer size in bytes (default: 1048576)
--help                     Show help
```

## How It Works

### Target Side

1. **Setup**: Create nixlAgent, allocate GPU memory, register with NIXL
2. **Publish**: Share metadata and transfer descriptors via TCP
3. **Connect**: Load initiator metadata, wait for connection ready
4. **Wait**: GPU kernel polls signal until data arrives
5. **Verify**: Check data correctness

### Initiator Side

1. **Setup**: Create nixlAgent, allocate GPU memory with data, register with NIXL
2. **Publish**: Share metadata via TCP
3. **Connect**: Load target metadata, fetch transfer descriptors
4. **Create**: Build GPU request handles for device API
5. **Transfer**: Launch GPU kernel that posts RDMA write + signal
6. **Measure**: Repeat and report performance

## Key Features

- **Direct API**: Shows NIXL C++ API usage without abstraction layers
- **Single file**: All logic in one place for easy understanding
- **Device API V2**: Uses modern `nixlPut()` and `nixlAtomicAdd()` GPU operations
- **Educational**: Clear example of how EP framework internally uses NIXL

## Further Reading

- **KERNEL_GUIDE.md** - GPU kernel patterns and Device API V2 details
- **UNIFIED_EXAMPLE.md** - Single-process variant with in-memory metadata exchange
- **EP framework** (`examples/device/ep/`) - Production-ready distributed training example
