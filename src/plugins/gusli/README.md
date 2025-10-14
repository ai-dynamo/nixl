# NIXL Gusli Plugin

This plugin utilizes `gusli_clnt.so` as an I/O backend for NIXL. The GUSLI client communicates with a GUSLI server to perform high-performance block device I/O operations.

## GUSLI Client-Server Architecture

GUSLI supports multiple connection modes:

1. **Local Shared Memory** (recommended): GUSLI client communicates with a GUSLI server on the local machine via shared memory, achieving the highest performance. The server is integrated into user-space block device storage providers like SPDK, NVMeshUM, etc. I/O completely bypasses the kernel.

2. **Networked Server** (device type `N`): GUSLI client can connect to a remote GUSLI server over the network, enabling distributed storage access while maintaining user-space performance benefits.

3. **Direct Local Access** (fallback): GUSLI client can work without a server by directly accessing local block devices/files using standard kernel APIs, but this is inefficient and not recommended for production use.

## Usage Guide
1. Build and install [Gusli](https://github.com/nvidia/gusli).
2. Do it via: git clone git clone https://github.com/nvidia/gusli.git
3. cd gusli; `make all BUILD_RELEASE=1 BUILD_FOR_UNITEST=0 VERBOSE=1 ALLOW_USE_URING=0`
4. Ensure that libraries: `libgusli_clnt.so`, are installed under `/usr/lib/`.
5. Ensure that headers are installed under `/usr/include/gusli_*.hpp`.
6. Build NIXL. [!IMPORTANT] You must build gusli before building NIXL
7. Once the Gusli Backend is built, you can use it in your data transfer task by specifying the backend name as "GUSLI".
8. See example in nixl_gusli_test.cpp file. In short:

```cpp
nixlAgent agent("your_client_name", nixlAgentConfig(true));
nixl_b_params_t params = gen_gusli_plugin_params(agent);	// Insert list of your block devices here, grep this function to see how it is used
nixlBackendH* gusli_ptr = nullptr;		// Backend gusli plugin (typically dont need to access this pointer)
nixl_status_t status = agent.createBackend("GUSLI", params, n_backend);
...
```

## Sample config

The config file can be generated using the GUSLI API:

```cpp
gusli::client_config_file conf(1 /*Version*/);
using gsc = gusli::bdev_config_params;
conf.bdev_add(gsc(__stringify (UUID_LOCAL_FILE_0), gsc::bdev_type::DEV_FS_FILE,    "./store0.bin", "sec=0x03", 0, gsc::connect_how::SHARED_RW));
conf.bdev_add(gsc(__stringify (UUID_K_DEV_ZERO_1), gsc::bdev_type::DEV_BLK_KERNEL, "/dev/zero",    "sec=0x71", 0, gsc::connect_how::EXCLUSIVE_RW));
conf.bdev_add(gsc(__stringify (UUID_NVME_DISK__0), gsc::bdev_type::DEV_BLK_KERNEL, "/dev/nvme0n1", "sec=0x07", 1, gsc::connect_how::EXCLUSIVE_RW));
params["config_file"] = conf.get();
```

See [gusli_client_api.hpp](https://github.com/NVIDIA/GUSLI/blob/main/gusli_client_api.hpp) for more details.

## GUSLI Configuration File

The GUSLI plugin uses a configuration file to define block devices. The config file format is:

```
# Config file
version=1
<device_id> <type> <access_mode> <shared> <path> <security_flags>
```

### Configuration Fields:
- `device_id`: Numeric identifier for the device
- `type`: Device type
  - `F`: File-backed storage
  - `K`: Kernel block device (e.g., `/dev/nvme0n1`, `/dev/zero`)
  - `N`: Networked GUSLI server
- `access_mode`: `W` (read/write) or `R` (read-only)
- `shared`: `N` (exclusive) or `Y` (shared)
- `path`: Device path or server address
  - **For `F` and `K` types:** Local file or device path (e.g., `./store0.bin`, `/dev/nvme0n1`)
  - **For `N` type:** Remote server address prefixed with protocol: `t` for TCP or `u` for UDP (e.g., `t127.0.0.1`, `u10.0.0.5`)
- `security_flags`: Security cookie (e.g., `sec=0x3`)

### Example Configurations:

**Local devices:**
```
# Config file
version=1
11 F W N ./store0.bin sec=0x3
14 K W N /dev/nvme0n1 sec=0x7
```

**Networked server:**
```
# Config file
version=1
20 N W N t192.168.1.100 sec=0x10
21 N W N u10.0.0.5 sec=0x20
```

**Mixed local and networked:**
```
# Config file
version=1
11 F W N ./store0.bin sec=0x3
20 N W N t192.168.1.100 sec=0x10
27 K W N /dev/nvme0n1 sec=0x7
```

When using NIXLBench, this configuration is auto-generated from the `--device_list` parameter, but you can also provide a custom config file using `--gusli_config_file`.

## Running gusli unit test
1. build NIXL in a directory. Example: be it `/root/NNN`
2. `clear; ninja -C /root/NNN install`
3. Run gusli unit-test via: `clear; /root/NNN/test/unit/plugins/gusli/nixl_gusli_test; GUSLI show`
4. Run in unit-test framework via `rm /root/NNN/meson-logs/testlog.txt; meson test gusli_plugin_test -C /root/NNN; cat /root/NNN/meson-logs/testlog.txt`

## Using GUSLI with NIXLBench

NIXLBench supports GUSLI backend for storage performance benchmarking. The benchmark tool allows configurable device specifications and automatically generates GUSLI configuration.

### Basic Usage

GUSLI backend requires explicit device specification using the `--device_list` parameter in format `id:type:path`.

**Important Notes**:
- For storage backends, the local memory is always DRAM (or VRAM), and the remote side is the storage backend (BLK for GUSLI)
- The operation type (`--op_type`) determines the data direction:
  - `WRITE`: DRAM → Block device (initiator has DRAM, target is BLK)
  - `READ`: Block device → DRAM (initiator has DRAM, target is BLK)
- Direct I/O (`storage_enable_direct`) is **automatically enabled** when GUSLI backend is selected for optimal performance

```bash
# Write to block device (DRAM → BLK)
./nixlbench --backend=GUSLI \
           --device_list="11:F:./store0.bin" \
           --op_type=WRITE \
           --num_initiator_dev=1 \
           --num_target_dev=1

# Read from block device (BLK → DRAM)
./nixlbench --backend=GUSLI \
           --device_list="11:F:./store0.bin" \
           --op_type=READ \
           --num_initiator_dev=1 \
           --num_target_dev=1

# Two devices with default security
./nixlbench --backend=GUSLI \
           --device_list="11:F:./store0.bin,14:K:/dev/zero" \
           --num_initiator_dev=2 \
           --num_target_dev=2 \
           --op_type=WRITE
```

### Device List Format

Devices are specified using the standard `--device_list` parameter with GUSLI-specific format `id:type:path`:
- `id`: Numeric device identifier (e.g., 11, 14, 20, 27)
- `type`: Device type
  - `F`: File-backed storage
  - `K`: Kernel block device
  - `N`: Networked GUSLI server
- `path`: Device path or server address
  - **For `F` type:** Local file path (e.g., `./store0.bin`)
  - **For `K` type:** Kernel device path (e.g., `/dev/nvme0n1`, `/dev/zero`)
  - **For `N` type:** Remote server address prefixed with protocol: `t` for TCP or `u` for UDP (e.g., `t127.0.0.1`, `u10.0.0.5`)

**Important**: The number of devices in `--device_list` must match `--num_initiator_dev` and `--num_target_dev`.

Security flags can be specified per-device using `--gusli_device_security`:

```bash
# Custom devices with per-device security
./nixlbench --backend=GUSLI \
           --device_list="11:F:./mystore.bin,27:K:/dev/nvme0n1" \
           --gusli_device_security="sec=0x3,sec=0x7" \
           --num_initiator_dev=2 \
           --num_target_dev=2 \
           --op_type=WRITE

# Partial security flags (missing entries use default sec=0x3)
./nixlbench --backend=GUSLI \
           --device_list="11:F:./mystore.bin,27:K:/dev/nvme0n1,14:K:/dev/zero" \
           --gusli_device_security="sec=0x3,sec=0x7" \
           --num_initiator_dev=3 \
           --num_target_dev=3 \
           --op_type=WRITE
```

### GUSLI-Specific Parameters

**Required**:
- `--device_list`: Device specs in format `id:type:path` (e.g., `11:F:./store0.bin,14:K:/dev/zero`)
- `--num_initiator_dev`: Must match number of devices in `--device_list`
- `--num_target_dev`: Must match number of devices in `--device_list`

**Optional**:
- `--gusli_client_name`: Client identifier (default: "NIXLBench")
- `--gusli_max_simultaneous_requests`: Concurrent request limit (default: 32)
- `--gusli_device_security`: Comma-separated security flags per device (e.g., `sec=0x3,sec=0x71`). Defaults to `sec=0x3` for missing entries
- `--gusli_bdev_byte_offset`: Starting LBA offset in bytes (default: 1048576 = 1MB)
- `--gusli_config_file`: Override with custom config file content (auto-generated if not provided)

### Advanced Examples

```bash
# Multi-device setup with NVMe and per-device security
./nixlbench --backend=GUSLI \
           --device_list="11:F:./store0.bin,14:K:/dev/zero,27:K:/dev/nvme0n1" \
           --gusli_device_security="sec=0x3,sec=0x71,sec=0x7" \
           --num_initiator_dev=3 \
           --num_target_dev=3 \
           --total_buffer_size=$((8*1024*1024*1024)) \
           --op_type=WRITE

# High concurrency NVMe benchmark
./nixlbench --backend=GUSLI \
           --device_list="27:K:/dev/nvme0n1" \
           --gusli_device_security="sec=0x7" \
           --num_initiator_dev=1 \
           --num_target_dev=1 \
           --gusli_max_simultaneous_requests=128 \
           --num_threads=8 \
           --op_type=READ

# Multiple file devices for load distribution
./nixlbench --backend=GUSLI \
           --device_list="11:F:./store0.bin,12:F:./store1.bin,13:F:./store2.bin" \
           --gusli_device_security="sec=0x3,sec=0x3,sec=0x3" \
           --num_initiator_dev=3 \
           --num_target_dev=3 \
           --op_type=WRITE

# Networked GUSLI server (TCP connection)
./nixlbench --backend=GUSLI \
           --device_list="20:N:t192.168.1.100" \
           --gusli_device_security="sec=0x10" \
           --num_initiator_dev=1 \
           --num_target_dev=1 \
           --op_type=WRITE

# Mixed local and networked devices
./nixlbench --backend=GUSLI \
           --device_list="11:F:./store0.bin,20:N:t192.168.1.100,27:K:/dev/nvme0n1" \
           --gusli_device_security="sec=0x3,sec=0x10,sec=0x7" \
           --num_initiator_dev=3 \
           --num_target_dev=3 \
           --op_type=WRITE
```

The benchmark will automatically distribute I/O operations across configured devices (local and/or networked) and generate appropriate GUSLI configuration.

## Known Issues
1. The `Notif[ication]` and `ProgTh[read]` features are not supported.
2. GUSLI backend requires per-iteration request creation/release due to a library limitation (handled automatically in NIXLBench).
