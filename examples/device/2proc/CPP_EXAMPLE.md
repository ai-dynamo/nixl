# Pure C++ Device API Example

This is a simplified, pure C++ version of the 2proc device API example. Unlike the Python version with multiple abstraction layers (pybind11 → device_host.cpp → device_buffer.py → example.py), this is a single self-contained C++ program.

## Architecture

```
simple_write.cpp        # Main program (~500 lines)
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

From NIXL repository root, in DFW container:

```bash
# Configure and build with all required flags
rm -rf build && mkdir build
meson setup build/ \
    -Ducx_path=/usr \
    -Dlibfabric_path=/usr/local \
    -Dbuild_nixl_ep=true \
    --prefix=/usr/local/nixl \
    --buildtype=release

cd build && ninja && ninja install
ldconfig

# Binary location
build/examples/device/2proc/simple_write
```

## Run

```bash
# Set UCX configuration for optimal container compatibility
export UCX_TLS=all                    # Auto-select best transports
export UCX_POSIX_USE_PROC_LINK=n      # Avoid /proc/PID/fd access errors

# Terminal 1 (initiator - must start FIRST to create TCPStore server)
./build/examples/device/2proc/simple_write --mode initiator --size 1048576

# Terminal 2 (target - start SECOND after initiator's server is ready)
./build/examples/device/2proc/simple_write --mode target --size 1048576
```

**UCX Configuration:**

Two key settings for container compatibility:

1. **`UCX_TLS=all`** (Transport Layer Selection)
   - Lets UCX auto-select best available transports including CUDA IPC

2. **`UCX_POSIX_USE_PROC_LINK=n`** (Disable procfs mode - **important for containers!**)
   - **Without this**: UCX tries to access `/proc/<other_pid>/fd/<fd>` for shared memory
   - **Result**: "Permission denied" errors in containers (containers block cross-process /proc access)
   - **With this setting**: UCX uses file paths instead (e.g., `/dev/shm/ucx_shm_posix_...`)
   - **Benefit**: Clean output, no permission errors

**Optional settings:**
- If `UCX_NET_DEVICES` is set, unset it for optimal NVLink performance

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

## Code Flow

```cpp
// nixlAgent setup (like EP framework)
nixlAgent agent("name", config);
agent.getPluginParams("UCX", mems, init_params);
init_params["num_workers"] = "1";
init_params["ucx_error_handling_mode"] = "none";
agent.createBackend("UCX", init_params, backend);

// CUDA memory
cudaMalloc(&data_ptr, size);
cudaMemset(data_ptr, fill_value, size);

// Register with NIXL
nixl_reg_dlist_t reg(VRAM_SEG);
reg.addDesc(nixlBlobDesc((uintptr_t)data_ptr, size, dev_id, ""));
agent.registerMem(reg, &params);

// Metadata exchange
std::string local_meta;
agent.getLocalMD(local_meta);
// ... exchange via TCP ...
agent.loadRemoteMD(remote_meta, remote_name);

// Create GPU request handles
agent.createXferReq(NIXL_WRITE, local_descs, remote_descs,
                    remote_name, xfer_req, &params);
agent.createGpuXferReq(*xfer_req, gpu_req);

// Launch device kernel
launch_post_write_and_signal(
    gpu_req_handles,      // GPU handles
    signal_gpu_req,       // Signal handle
    signal_ptr,           // Signal memory
    size,                 // Transfer size
    ...
);
```

## Comparison: Python vs C++

### Python Version (Current)
```
example.py                 (~230 lines)
  └── device_buffer.py     (~320 lines)
      └── device_host.so   (pybind11, ~200 lines C++)
          └── nixlAgent    (C++ API)
```
**Total**: ~750 lines across 3 layers

### C++ Version (This)
```
simple_write.cpp           (~500 lines, all-in-one)
  └── nixlAgent            (C++ API, direct)
```
**Total**: ~500 lines, single layer

## Benefits of C++ Version

1. **Simpler**: No Python/C++ boundary, no pybind11
2. **Clearer**: Direct NIXL API usage visible in one file
3. **Faster build**: No Python module compilation
4. **Educational**: Shows exactly how EP framework uses NIXL
5. **Portable**: Just C++ and CUDA, no Python dependencies

## When to Use Which

**Use C++ version** when:
- Learning device API fundamentals
- Understanding how EP framework works internally
- Building C++ applications with device API
- Want minimal dependencies

**Use Python version** when:
- Rapid prototyping
- Integration with PyTorch workflows
- Python ecosystem tools needed

## Next Steps

After understanding this example, see:
- **KERNEL_GUIDE.md** - GPU kernel patterns explained
- **EP framework** (`examples/device/ep/`) - Production MoE use case
- **NIXL docs** - Full API reference
