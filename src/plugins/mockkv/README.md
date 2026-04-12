# MOCKKV Plugin — Simple In-Memory Key-Value Store

## Introduction

MOCKKV is a minimal NIXL backend plugin that implements an in-memory key-value store with C++ `std::unordered_map`. It helps learners understand how NIXL plugins work without external services or async complexity.

## Features

- **Simple**: Standard C++ only, no extra dependencies
- **Synchronous**: All operations complete in-line for easier reasoning
- **Threading**: Same assumption as typical NIXL backends — the agent serializes calls to one engine (no internal mutex, same as Obj)
- **Complete**: Implements the required NIXL backend surface
- **Commented**: Heavily annotated for learning

## Architecture

### Storage

```cpp
std::unordered_map<std::string, std::vector<uint8_t>> kv_store_;
```

- **Key**: `std::string` from `metaInfo` or stringified `devId`
- **Value**: `std::vector<uint8_t>` (opaque bytes)

### Flow

1. **registerMem**: Register descriptors; build devId → key mapping
2. **prepXfer**: Validate parameters
3. **postXfer**: Run PUT/GET (synchronous)
   - **WRITE (PUT)**: Copy from local buffer into the map
   - **READ (GET)**: Copy from the map into local buffer
4. **checkXfer**: Status (always SUCCESS here — work finished in postXfer)
5. **releaseReqH**: Free the request handle

## Build

### Static plugin

```bash
meson setup build -Dstatic_plugins=MOCKKV --prefix=/usr/local/nixl
ninja -C build
```

### Shared plugin

```bash
meson setup build --prefix=/usr/local/nixl
ninja -C build
```

## nixlbench example

```bash
cd /workspace/nixl/benchmark/nixlbench
./build/nixlbench \
    --backend MOCKKV \
    --op_type WRITE \
    --total_buffer_size 1000000 \
    --start_block_size 1024 \
    --max_block_size 4096 \
    --start_batch_size 1 \
    --max_batch_size 1 \
    --warmup_iter 2 \
    --num_iter 5
```

## Learning checklist

### 1. Plugin registration

- `mockkv_plugin.cpp`: entry points
- `createStaticMOCKKVPlugin()`: static registration
- `nixl_plugin_init()`: dynamic load

### 2. Backend implementation

- Subclass `nixlBackendEngine`
- Implement required virtuals
- Handle registration and transfers

### 3. Descriptors

- `nixlBlobDesc`: addresses, devId, metaInfo, …
- `nixlBackendMD`: backend metadata (key mapping)
- `nixl_meta_dlist_t`: metadata descriptor lists

### 4. Transfer

- `prepXfer`: validation
- `postXfer`: actual data movement
- `checkXfer`: status (polling for async backends)

## MOCKKV vs Redis plugin

| Aspect | MOCKKV | REDIS |
|--------|--------|-------|
| Store | In-process map | Redis server |
| Ops | Sync | Async |
| Deps | None | hiredis, libevent |
| Complexity | Low | Higher |
| Typical use | Learning | Production-style |

## Layout

```
mockkv/
├── mockkv_backend.h
├── mockkv_backend.cpp
├── mockkv_plugin.cpp
├── meson.build
├── MOCKKV_ARCHITECTURE.md
└── README.md
```

## Debugging

```bash
NIXL_LOG_LEVEL=DEBUG ./build/nixlbench --backend MOCKKV ...
```

## Next steps

1. Read the Redis plugin for async patterns
2. Read the OBJ plugin for object storage
3. Prototype your own plugin

## FAQ

**Q: Why no async?**  
A: MOCKKV is intentionally minimal. Async adds promise/futures, thread pools, etc., which obscures the NIXL API for beginners.

**Q: Is data persisted?**  
A: No. In-memory only; process exit drops all data — by design for teaching.

**Q: Multi-process sharing?**  
A: No. Each process has its own map. Use the Redis plugin for shared remote storage.
