# INMEMKV Example Plugin

INMEMKV is a small dynamic NIXL backend plugin that implements an in-process key-value store. It is meant to be read as an example of how to write a backend plugin, not as a production storage backend.

## Introduction

The plugin uses a layered design:

```text
nixlInMemKVEngine          Plugin entry (extends nixlKVEngine)
    -> nixlKVEngine         Shared thin wrapper (src/plugins/kv/kv_engine.h)
        -> nixlInMemKVEngineImpl   Plugin logic (inmemkv_engine_impl.*)
            -> InMemKVStore (iKVStore)   Storage (inmemkv_store.*)
```

Shared headers under `src/plugins/kv/`:

- `kv_store.h` — `iKVStore` put/get/exists storage interface
- `kv_engine_impl.h` — `nixlKVEngineImpl` abstract backend logic interface
- `kv_engine.h` / `kv_engine.cpp` — `nixlKVEngine` delegation wrapper

`InMemKVStore` in this directory provides the in-memory `iKVStore` implementation.
Each registered descriptor names a key through `metaInfo`; if `metaInfo` is empty,
INMEMKV falls back to a string form of `devId`. A `NIXL_WRITE` calls `store_->put(...)`.
A `NIXL_READ` calls `store_->get(...)`.

## What It Demonstrates

- Building a NIXL backend as a dynamic plugin: `libplugin_INMEMKV.so`
- Exporting the plugin entry points in `inmemkv_plugin.cpp`
- Implementing a minimal `nixlBackendEngine`
- Registering and deregistering backend metadata
- Preparing, posting, checking, and releasing a transfer request
- Supporting a local-only backend with no service, remote connection, or notification path

INMEMKV intentionally keeps the backend simple:

- Standard C++ only
- No daemon or external service
- No persistence
- No cross-process sharing
- Synchronous completion in `postXfer`
- `DRAM_SEG` only
- `supportsLocal() == true`
- `supportsRemote() == false`
- `supportsNotif() == false`

Its architecture is detailed in [INMEMKV architecture](INMEMKV_ARCHITECTURE.md).

## Build

INMEMKV lives under `examples/plugins/inmemkv`. It is built from the **NIXL repository root** (not from `benchmark/nixlbench`). It is gated by `build_examples` (default `true`) and is not installed by `ninja install`.

### Step 1: Configure and build the plugin (NIXL root)

Run all commands from the NIXL repository root, for example `/workspace/nixl`:

```bash
cd /workspace/nixl

# First-time setup, or reconfigure an existing build directory
meson setup --reconfigure build \
    -Dbuild_examples=true \
    --prefix=/usr/local/nixl \
    --buildtype=release

# Build the plugin (use the output path as the ninja target)
ninja -C build examples/plugins/inmemkv/libplugin_INMEMKV.so
```

After a `meson setup --reconfigure`, you can also use the convenience alias:

```bash
ninja -C build INMEMKV
```

Verify the plugin was produced:

```bash
ls -l build/examples/plugins/inmemkv/libplugin_INMEMKV.so
```

`ninja: no work to do` means the plugin is already up to date.

### Step 2: Export `NIXL_PLUGIN_DIR`

NIXL discovers dynamic plugins from `NIXL_PLUGIN_DIR`. Export the build-tree plugin directory before starting any process that creates a `nixlAgent`:

```bash
export NIXL_PLUGIN_DIR=/workspace/nixl/build/examples/plugins/inmemkv
```

The `export` matters: a plain shell assignment is not inherited by child processes.

## Use With nixlbench

nixlbench is a **separate Meson project** under `benchmark/nixlbench`. Do not pass `-Dbuild_examples` there; use `-Denable_inmemkv` instead.

### Step 3: Build nixlbench

```bash
cd /workspace/nixl/benchmark/nixlbench
meson setup build -Denable_inmemkv=true -Dnixl_path=/usr/local/nixl
ninja -C build
```

### Step 4: Run

```bash
export NIXL_PLUGIN_DIR=/workspace/nixl/build/examples/plugins/inmemkv
./build/nixlbench \
    --backend INMEMKV \
    --op_type WRITE \
    --total_buffer_size 1000000 \
    --start_block_size 1024 \
    --max_block_size 4096 \
    --start_batch_size 1 \
    --max_batch_size 1 \
    --warmup_iter 2 \
    --num_iter 5
```

nixlbench has a small amount of INMEMKV-specific code because INMEMKV behaves like a storage backend from the benchmark's point of view, but uses `DRAM_SEG` descriptors with keys in `metaInfo` instead of file descriptors and `FILE_SEG` descriptors.

## Code Tour

- `meson.build`: builds `libplugin_INMEMKV.so` as a build-tree-only dynamic plugin.
- `inmemkv_plugin.cpp`: exports `nixl_plugin_init` and `nixl_plugin_fini`, and registers the backend name `INMEMKV`.
- `inmemkv_backend.h/.cpp`: thin `nixlInMemKVEngine` wrapper and impl factory.
- `inmemkv_engine_impl.h/.cpp`: `nixlInMemKVEngineImpl` (registerMem, prep/postXfer, ...).
- `inmemkv_store.h/.cpp`: `InMemKVStore` implementing shared `iKVStore`.
- `src/plugins/kv/kv_store.h`: shared storage interface.
- `src/plugins/kv/kv_engine_impl.h`: shared backend logic interface.
- `src/plugins/kv/kv_engine.h/.cpp`: shared `nixlKVEngine` delegation layer.
- `INMEMKV_ARCHITECTURE.md`: explains the plugin lifecycle and the design choices in more detail.

A good reading order is:

1. `inmemkv_plugin.cpp`
2. `src/plugins/kv/kv_engine_impl.h` and `kv_engine.h`
3. `inmemkv_backend.h` and `inmemkv_backend.cpp`
4. `inmemkv_engine_impl.cpp` (`registerMem`, `prepXfer`, `postXfer`, `deregisterMem`)
5. `INMEMKV_ARCHITECTURE.md`

## Lifecycle Summary

1. NIXL loads `libplugin_INMEMKV.so` from `NIXL_PLUGIN_DIR`.
2. `nixl_plugin_init()` returns plugin metadata and the engine factory.
3. The application creates a `nixlAgent` and asks for backend `INMEMKV`.
4. `registerMem()` records a key and returns backend metadata.
5. `prepXfer()` validates a `NIXL_WRITE` or `NIXL_READ` and creates a placeholder request handle.
6. `postXfer()` performs the synchronous PUT or GET.
7. `checkXfer()` returns `NIXL_SUCCESS` because the work already completed.
8. `releaseReqH()` frees the placeholder request handle.
9. `deregisterMem()` removes the key mapping and frees backend metadata.

## Limitations

INMEMKV is deliberately narrow:

- Dynamic example plugin only
- Build-tree only, not installed
- Process-local map only
- No persistence
- No remote metadata exchange protocol
- No notifications
- No internal locking; it assumes serialized backend calls, like other simple backend examples
