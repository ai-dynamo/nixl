# NIXL Infinia Plugin

This backend provides high-performance object storage using DDN's Infinia storage system with C++20 coroutine-based async API.

## Dependencies

This backend requires the Infinia Client and Infinia Async libraries. The Infinia installation should include:

- **libred_client.so** - Core Infinia storage client functionality
- **libred_async.so** - C++20 coroutine-based async API
- **Headers**: `<red/red_async.hpp>`, `<red/red_status.h>`

A C++20 compiler (GCC 10+ or Clang 14+) is required for coroutine support.

### Build Configuration

```bash
# Configure with Infinia support
meson setup build -Dinfinia_path=/path/to/infinia/installation

# Build
cd build
ninja
```

The `infinia_path` should point to the Infinia installation directory containing `lib/` and `include/` subdirectories.

## Configuration

The Infinia backend supports configuration through backend parameter maps and environment variables.
Backend parameters take precedence over environment variables.

### Backend Parameters

Backend parameters are passed as a key-value map (`nixl_b_params_t`) when creating the backend instance:

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `cluster` | Infinia cluster name | `cluster1` | No |
| `tenant` | Tenant name | `red` | No |
| `subtenant` | Subtenant name | `red` | No |
| `dataset` | Dataset/bucket name | `nixl` | No |
| `sthreads` | Number of service threads | `8` | No |
| `num_buffers` | Number of buffers for operations | `512` | No |
| `num_ring_entries` | Number of ring buffer entries | `512` | No |
| `coremasks` | CPU core affinity mask (hexadecimal) | `0x2` | No |
| `max_retries` | Maximum retries for failed operations | (library default) | No |
| `config_file` | Path to configuration file (key=value format) | - | No |

### Environment Variables

The following environment variables are supported:

| Variable | Description | Example |
|----------|-------------|---------|
| `RED_CLUSTER` | Infinia cluster name | `mycluster` |
| `RED_TENANT` | Tenant name (can include subtenant as `tenant/subtenant`) | `mytenant/mysubtenant` |
| `RED_DATASET` | Dataset name | `mydataset` |

### Configuration Priority

Configuration values are resolved in the following priority order (highest to lowest):

1. **Environment Variables**: Infinia environment variables
2. **Backend Parameters**: Values passed directly in the backend parameter map
3. **Configuration File**: Values from config file (if `config_file` parameter is provided)
4. **Built-in Defaults**: Default values

### Configuration Examples

#### Minimal Configuration

```cpp
nixl_b_params_t params = {{"cluster", "mycluster"}, {"dataset", "mydataset"}};
agent.createBackend("INFINIA", params);
```

#### Environment Variable Configuration

```bash
export RED_CLUSTER=mycluster
export RED_TENANT=mytenant/mysubtenant
export RED_DATASET=mydataset
```

```cpp
agent.createBackend("INFINIA", {});
```

#### Configuration File

```bash
# infinia.conf
cluster=mycluster
tenant=mytenant
subtenant=mysubtenant
dataset=mydataset
sthreads=16
num_buffers=1024
coremasks=0xff
```

```cpp
nixl_b_params_t params = {{"config_file", "/path/to/infinia.conf"}};
agent.createBackend("INFINIA", params);
```

## Transfer Operations

The Infinia backend supports read and write operations between local memory and Infinia storage. Key aspects:

### Supported Memory Types

- **DRAM_SEG**: Host memory (CPU RAM) - pre-registered for zero-copy transfers
- **VRAM_SEG**: Device memory (GPU VRAM) - requires CUDA, pre-registered for zero-copy transfers
- **OBJ_SEG**: Object storage (no physical memory backing)

### Device ID to Object Key Mapping

- Each object in Infinia storage is identified by a unique key
- The backend maintains a mapping between device IDs (`devId`) and object keys
- When registering OBJ_SEG memory:
  - If `metaInfo` is provided in the blob descriptor, it is used as the object key
  - Otherwise, the device ID is converted to a string and used as the object key
- This mapping is used during transfer operations to locate the correct Infinia storage object

### Memory Registration

- **DRAM_SEG/VRAM_SEG**: Memory is pre-registered with Infinia using `red_config_t::register_user_memory()` to obtain a handle for zero-copy transfers
- **OBJ_SEG**: No physical memory registration; only creates devId-to-key mapping
- Transfer buffers must be fully contained within registered memory regions

### Asynchronous Operations

- All transfer operations are asynchronous using C++20 coroutines
- The backend uses `red_async::BatchTask` for parallel batch execution
- Operations are executed in the background with automatic polling
- Transfer handles can be prepared once and posted multiple times for efficient repeated operations
- The `checkXfer` function polls for operation completion (non-blocking)
- Request handles must be released using `releaseReqH` after operations complete
