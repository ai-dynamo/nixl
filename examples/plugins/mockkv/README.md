# MOCKKV Example Plugin

MOCKKV is a small dynamic NIXL backend plugin that implements an in-process key-value store. It is meant to be read as an example of how to write a backend plugin, not as a production storage backend.

## Introduction

The plugin stores bytes in:

```cpp
std::unordered_map<std::string, std::vector<uint8_t>> kv_store_;
```

Each registered descriptor names a key through `metaInfo`; if `metaInfo` is empty, MOCKKV falls back to a string form of `devId`. A `NIXL_WRITE` copies bytes from the user buffer into the map. A `NIXL_READ` copies bytes from the map back into the user buffer.

## What It Demonstrates

- Implementing a minimal `nixlBackendEngine`
    - Registering and deregistering backend metadata
    - Preparing, posting, checking, and releasing a transfer request
    - Supporting a local-only backend with no service, remote connection, or notification path
    - Exporting the plugin entry points in `mockkv_plugin.cpp`
- Building a NIXL backend as a dynamic plugin: `libplugin_MOCKKV.so`

MOCKKV intentionally keeps the backend simple:

- Standard C++ only
- No daemon or external service
- No persistence
- No cross-process sharing
- Synchronous completion in `postXfer`
- `DRAM_SEG` only
- `supportsLocal() == true`
- `supportsRemote() == false`
- `supportsNotif() == false`

Its architecture is detailed in [MOCKKV architecture](MOCKKV_ARCHITECTURE.md).

## Build

MOCKKV lives under `examples/plugins/mockkv`. It

- is built only when the root NIXL build enables examples, and
- is not installed by `ninja install`.

1. Build NIXL plugin example

From the NIXL repository root:

```bash
meson setup build -Dbuild_examples=true --prefix=/usr/local/nixl
ninja -C build MOCKKV
```

The plugin is produced in the build tree:

```bash
build/examples/plugins/mockkv/libplugin_MOCKKV.so
```

2. Export `NIXL_PLUGIN_DIR`

NIXL discovers dynamic plugins from `NIXL_PLUGIN_DIR`, so export the build-tree plugin directory before starting the process that creates the `nixlAgent`:

```bash
export NIXL_PLUGIN_DIR=/path/to/nixl/build/examples/plugins/mockkv
```

The `export` matters: a plain shell assignment is not inherited by child processes.

## Use With nixlbench

MOCKKV support in nixlbench is gated separately so the benchmark does not depend on this example by default.

Build nixlbench with:

```bash
cd benchmark/nixlbench
meson setup build -Denable_mockkv=true -Dnixl_path=/usr/local/nixl
ninja -C build
```

Run with the plugin directory exported and the backend set to `MOCKKV`:

```bash
export NIXL_PLUGIN_DIR=/path/to/nixl/build/examples/plugins/mockkv
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

nixlbench has a small amount of MOCKKV-specific code because MOCKKV behaves like a storage backend from the benchmark's point of view, but uses `DRAM_SEG` descriptors with keys in `metaInfo` instead of file descriptors and `FILE_SEG` descriptors.

## Code Tour

- `meson.build`: builds `libplugin_MOCKKV.so` as a build-tree-only dynamic plugin.
- `mockkv_plugin.cpp`: exports `nixl_plugin_init` and `nixl_plugin_fini`, and registers the backend name `MOCKKV`.
- `mockkv_backend.h`: declares the `nixlMockKVEngine` class and the backend methods it implements.
- `mockkv_backend.cpp`: implements registration, transfer preparation, PUT/GET, completion, cleanup, and local metadata handling.
- `MOCKKV_ARCHITECTURE.md`: explains the plugin lifecycle and the design choices in more detail.

A good reading order is:

1. `mockkv_plugin.cpp`
2. `mockkv_backend.h`
3. `mockkv_backend.cpp`, starting with `registerMem`, `prepXfer`, `postXfer`, and `deregisterMem`
4. `MOCKKV_ARCHITECTURE.md`

## Lifecycle Summary

1. NIXL loads `libplugin_MOCKKV.so` from `NIXL_PLUGIN_DIR`.
2. `nixl_plugin_init()` returns plugin metadata and the engine factory.
3. The application creates a `nixlAgent` and asks for backend `MOCKKV`.
4. `registerMem()` records a key and returns backend metadata.
5. `prepXfer()` validates a `NIXL_WRITE` or `NIXL_READ` and creates a placeholder request handle.
6. `postXfer()` performs the synchronous PUT or GET.
7. `checkXfer()` returns `NIXL_SUCCESS` because the work already completed.
8. `releaseReqH()` frees the placeholder request handle.
9. `deregisterMem()` removes the key mapping and frees backend metadata.

## Limitations

MOCKKV is deliberately narrow:

- Dynamic example plugin only
- Build-tree only, not installed
- Process-local map only
- No persistence
- No remote metadata exchange protocol
- No notifications
- No internal locking; it assumes serialized backend calls, like other simple backend examples
