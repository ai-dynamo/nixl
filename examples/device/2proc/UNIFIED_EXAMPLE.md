# Unified Single-File Device API Example

## Overview

`simple_write_unified.cu` is a simplified, single-file version of the device API example that demonstrates GPU-initiated RDMA transfers in ~350 lines of code.

## Key Differences from `simple_write.cpp`

### Simplified Version (`simple_write_unified.cu`)
```
✅ Single .cu file (~350 lines)
✅ Kernels inline (no separate header)
✅ NIXL native TCP (no PyTorch dependency)
✅ Notification-based metadata exchange
✅ Simpler to read and understand
```

### Original Version (`simple_write.cpp` + helpers)
```
- Multiple files (simple_write.cpp, kernels.cu, kernels.h, tcp_store.cpp/h)
- PyTorch TCPStore for metadata exchange
- More production-oriented structure
- ~500+ lines total
```

## Dependencies

**Unified version needs only:**
- CUDA
- NIXL

**No PyTorch required!**

## How It Works

### 1. Target (Receiver)
```cpp
// Listen for connections
nixlAgentConfig cfg(true);
cfg.listen_for_peers = true;
cfg.listen_port = 9998;
auto agent = nixlAgent("target", cfg);

// Wait for initiator to connect
// Exchange descriptors via NIXL notifications
// Launch GPU kernel to wait for signal
```

### 2. Initiator (Sender)
```cpp
// Connect to target
nixlAgentConfig cfg(true);
cfg.listen_for_peers = false;
auto agent = nixlAgent("initiator", cfg);
agent->connectToPeer(target_ip, 9998, "target", target_name);

// Get target descriptors via notifications
// Launch GPU kernel to write data + signal
```

### 3. Metadata Exchange

Uses a simple `SimpleStore` class that exchanges key-value pairs through NIXL notifications:

```cpp
class SimpleStore {
    void set(key, value) {
        // Send "SET:key:value" notification
    }

    string get(key) {
        // Poll notifications for "SET:key:value"
    }
};
```

**No external dependencies!** Everything uses NIXL's built-in notification system.

## Build

```bash
cd /shared/container/nixl_ep
rm -rf build && mkdir build
meson setup build/ \
    -Ducx_path=/usr \
    -Dbuild_nixl_ep=true \
    --prefix=/usr/local/nixl

cd build && ninja
```

Binary location: `build/examples/device/2proc/simple_write_unified`

## Run

```bash
# Terminal 1 - Target (start first)
./build/examples/device/2proc/simple_write_unified --mode target --size 1048576

# Terminal 2 - Initiator
./build/examples/device/2proc/simple_write_unified --mode initiator --size 1048576
```

### Options
```
--mode <initiator|target>   Role (default: initiator)
--size <bytes>              Buffer size (default: 1MB)
--peer-ip <ip>              Peer IP address (default: 127.0.0.1)
```

## Expected Output

**Target:**
```
[target] Starting...
[target] Waiting for initiator to connect...
[target] Connected to: initiator
[target] Waiting for data...
[target] Data received! Verifying...
[target] ✓ Checksum OK: 44040192
[target] Complete!
```

**Initiator:**
```
[initiator] Starting...
[initiator] Connecting to target at 127.0.0.1:9998
[initiator] Connected to: target
[initiator] Launching GPU kernel to write 1048576 bytes...
[initiator] Transfer complete!
[initiator] Complete!
```

## Code Structure

```
simple_write_unified.cu (350 lines)
├── CUDA Kernels (~30 lines)
│   ├── write_and_signal_kernel()  - Post RDMA write + signal
│   └── wait_for_signal_kernel()   - Poll for signal
│
├── SimpleStore (~40 lines)
│   ├── set() - Publish via notification
│   └── get() - Poll for notification
│
├── run_target() (~100 lines)
│   ├── Setup agent with listening
│   ├── Register memory
│   ├── Accept connection
│   ├── Exchange descriptors
│   └── Wait for GPU signal
│
└── run_initiator() (~100 lines)
    ├── Setup agent
    ├── Connect to target
    ├── Register memory
    ├── Get remote descriptors
    └── Launch write+signal kernel
```

## Learning Path

1. **Start here** - `simple_write_unified.cu` (single file, easy to follow)
2. **Then see** - `simple_write.cpp` (production structure, PyTorch TCPStore)
3. **Deep dive** - `KERNEL_GUIDE.md` (THREAD vs WARP cooperation)

## When to Use Which

### Use `simple_write_unified.cu` when:
- Learning device API basics
- Quick prototyping
- Minimal dependencies needed
- Single-file simplicity preferred

### Use `simple_write.cpp` when:
- Building production applications
- Already using PyTorch
- Need modular structure
- Prefer separated concerns

## Performance

Same performance as the original version:
- **60-70 GB/s** on same-node H100 GPUs (NVLink)
- Both use identical device API calls
- Both use UCX with CUDA IPC transport

## Limitations

- Single buffer transfer (not pipelined)
- Single GPU device (GPU 0)
- Local testing focus (127.0.0.1 default)
- Basic error handling

These are intentional to keep the example simple and focused on core concepts.

## Next Steps

Once you understand this example:
- Add pipelined transfers (multiple writes per signal)
- Use WARP-level cooperation (see `KERNEL_GUIDE.md`)
- Scale to multiple GPU blocks
- Handle multi-GPU systems
- Add distributed testing across nodes

## See Also

- `CPP_EXAMPLE.md` - Detailed usage and UCX configuration
- `KERNEL_GUIDE.md` - GPU kernel patterns explained
- `simple_write.cpp` - Full-featured version with PyTorch TCPStore
