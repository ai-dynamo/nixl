# INMEMKV Architecture Guide

This guide explains how the INMEMKV example plugin fits into NIXL and how its layered KV backend design works.

INMEMKV is intentionally small. It demonstrates a KV-style NIXL backend with the fewest moving pieces.

## Big Picture

A NIXL application does not call a backend plugin directly. It creates a `nixlAgent`, asks the agent to use a backend by name, and then uses normal NIXL APIs such as `registerMem`, `createXferReq`, `postXferReq`, and `deregisterMem`.

For INMEMKV, the layers look like this:

```text
Application (e.g. nixlbench)
  selects backend "INMEMKV"
  builds descriptor lists
  calls nixlAgent APIs
        |
        v
nixlAgent
  loads/discovers backend plugins
  creates the backend engine
  owns local sections and transfer requests
  routes calls to the backend engine
        |
        v
nixlInMemKVEngine          Plugin entry (extends nixlKVEngine)
        |
        v
nixlKVEngine               Shared thin wrapper (src/plugins/kv/)
  delegates all data-plane calls to impl_
        |
        v
nixlInMemKVEngineImpl      Plugin-specific logic (inmemkv_engine_impl.*)
  registerMem, prepXfer, postXfer, ...
        |
        v
InMemKVStore (iKVStore)    Storage put/get/exists (inmemkv_store.*)
```

### Layer Responsibilities

| Layer | Class | Role |
|-------|-------|------|
| NIXL-facing engine | `nixlKVEngine` | Thin wrapper; implements `nixlBackendEngine` |
| Abstract impl interface | `nixlKVEngineImpl` | Backend-specific logic contract |
| Concrete impl | `nixlInMemKVEngineImpl` | INMEMKV register/transfer implementation |
| Storage abstraction | `iKVStore` | put/get/exists storage operations |
| Shared headers | `src/plugins/kv/` | Reusable across KV backends |

The important separation is:

- The **application** decides what descriptors it wants to register and transfer.
- The **NIXL core** manages plugin loading, backend instances, metadata lists, and request flow.
- **`nixlKVEngine`** implements the `nixlBackendEngine` contract and forwards calls.
- **`nixlKVEngineImpl`** subclasses implement backend-specific behavior.
- **`iKVStore`** isolates raw put/get/exists storage from NIXL descriptor mapping.

## Shared KV Headers (`src/plugins/kv/`)

### `kv_store.h` — `iKVStore`

Minimal storage interface shared by all KV backends:

```cpp
class iKVStore {
    virtual bool put(key, data, len) = 0;
    virtual bool get(key, data, len, out_len) = 0;
    virtual bool exists(key) = 0;
};
```

INMEMKV uses `InMemKVStore` (`std::unordered_map`). A future REDIS plugin would use `RedisKVStore` (hiredis).

### `kv_engine_impl.h` — `nixlKVEngineImpl`

Abstract interface for KV backend logic. Each KV plugin provides a concrete subclass implementing:

- `getSupportedMems()`
- `registerMem` / `deregisterMem` / `queryMem`
- `prepXfer` / `postXfer` / `checkXfer` / `releaseReqH`

Note: `prepXfer` receives `local_agent` explicitly so impl classes do not depend on `nixlBackendEngine` protected members.

### `kv_engine.h` / `kv_engine.cpp` — `nixlKVEngine`

Thin wrapper extending `nixlBackendEngine`. Declares common KV capabilities:

```cpp
supportsLocal()  == true
supportsRemote() == false
supportsNotif()  == false
```

All data-plane methods delegate to `impl_` in `kv_engine.cpp`.

## Dynamic Plugin Entry Point

`inmemkv_plugin.cpp` is the plugin boundary. NIXL discovers `libplugin_INMEMKV.so` and calls:

```cpp
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *nixl_plugin_init();
extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini();
```

`nixl_plugin_init()` returns plugin metadata and an engine factory through `nixlBackendPluginCreator<nixlInMemKVEngine>::create(...)`.

## Plugin Wiring

`inmemkv_backend.cpp` is minimal — it only wires the factory:

```cpp
nixlInMemKVEngine::nixlInMemKVEngine(const nixlBackendInitParams *init_params)
    : nixlKVEngine(init_params, createInMemKVEngineImpl(init_params)) {}
```

The factory function `createInMemKVEngineImpl()` returns a `std::unique_ptr<nixlKVEngineImpl>`.

## Data Model

Each registered memory descriptor is interpreted as a KV key:

1. `mem.metaInfo`, when provided
2. `std::to_string(mem.devId)`, as a fallback

`nixlInMemKVEngineImpl` keeps:

```cpp
std::unordered_map<uint64_t, std::string> devIdToKey_;
std::unique_ptr<iKVStore> store_;
```

`postXfer()` resolves the key from remote metadata or `devIdToKey_`, then calls `store_->put()` or `store_->get()`.

## Lifecycle

```text
1. create backend          nixlInMemKVEngine -> nixlKVEngine -> nixlInMemKVEngineImpl
2. register memory         registerMem() -> nixlInMemKVMetadata + devIdToKey_
3. prepare transfer        prepXfer() -> allocate request handle
4. post transfer           postXfer() -> store_->put/get (synchronous)
5. check transfer          checkXfer() -> NIXL_SUCCESS (already done)
6. release request handle  releaseReqH()
7. deregister memory       deregisterMem() -> free metadata
```

## Step-by-Step Implementation Guide

When adding a new KV backend (e.g. REDIS), follow these steps:

### Step 1: Implement `iKVStore`

Create a storage class in your plugin directory:

```cpp
class RedisKVStore : public iKVStore { ... };
```

### Step 2: Implement `nixlKVEngineImpl`

Create `redis_engine_impl.h/.cpp`:

```cpp
class nixlRedisKVEngineImpl : public nixlKVEngineImpl {
    std::unique_ptr<iKVStore> store_;
    // registerMem, prepXfer, postXfer, ...
};
```

Copy the descriptor/key mapping logic from `nixlInMemKVEngineImpl` and adapt storage calls.

### Step 3: Create thin engine wrapper

```cpp
class nixlRedisKVEngine : public nixlKVEngine {
public:
    explicit nixlRedisKVEngine(const nixlBackendInitParams *p)
        : nixlKVEngine(p, std::make_unique<nixlRedisKVEngineImpl>(p)) {}
};
```

### Step 4: Register plugin

Use `nixlBackendPluginCreator<nixlRedisKVEngine>` in `redis_plugin.cpp`.

### Step 5: Build

Link `../../../src/plugins/kv/kv_engine.cpp` in your plugin's `meson.build`.

## File Map

```text
src/plugins/kv/
  kv_store.h           iKVStore storage interface
  kv_engine_impl.h     nixlKVEngineImpl abstract interface
  kv_engine.h          nixlKVEngine thin wrapper declaration
  kv_engine.cpp        nixlKVEngine delegation implementation

examples/plugins/inmemkv/
  meson.build               build-tree-only dynamic plugin target
  inmemkv_plugin.cpp         plugin entry points and backend registration
  inmemkv_backend.h/.cpp     thin nixlInMemKVEngine wrapper + factory
  inmemkv_engine_impl.h/.cpp nixlInMemKVEngineImpl (backend logic)
  inmemkv_store.h/.cpp       InMemKVStore (iKVStore implementation)
  README.md                  quick start and usage notes
  INMEMKV_ARCHITECTURE.md    this guide
```

## Reading Checklist

When learning the code, read in this order:

1. `src/plugins/kv/kv_engine_impl.h` — abstract impl contract
2. `src/plugins/kv/kv_engine.h` + `kv_engine.cpp` — delegation layer
3. `inmemkv_plugin.cpp` — how NIXL learns the backend name and factory
4. `inmemkv_backend.cpp` — factory wiring (3 lines of logic)
5. `inmemkv_engine_impl.cpp` — `registerMem`, `prepXfer`, `postXfer`
6. `inmemkv_store.cpp` — raw put/get/exists
7. Guarded INMEMKV code in nixlbench — how an application prepares descriptors
