# MOCKKV Architecture Guide

This guide explains how the MOCKKV example plugin fits into NIXL and what parts are useful when writing a new backend plugin.

MOCKKV is intentionally small. It does not try to be a complete storage system. Its job is to show the shape of a NIXL backend with the fewest moving pieces:

- one dynamic plugin entry point file
- one backend engine class
- local registration and deregistration
- synchronous transfer execution
- simple metadata ownership

## Big Picture

A NIXL application does not call a backend plugin directly. It creates a `nixlAgent`, asks the agent to use a backend by name, and then uses normal NIXL APIs such as `registerMem`, `createXferReq`, `postXferReq`, and `deregisterMem`.

For MOCKKV, the layers look like this:

```text
Application, for example nixlbench
  selects backend "MOCKKV"
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
nixlMockKVEngine
  registers DRAM descriptors as KV keys
  prepares synchronous requests
  copies bytes into/out of an in-process map
  releases request handles and metadata
```

The important separation is:

- The application decides what descriptors it wants to register and transfer.
- The NIXL core manages plugin loading, backend instances, metadata lists, and request flow.
- The backend implements the storage-specific behavior behind the `nixlBackendEngine` interface.

## Dynamic Plugin Entry Point

`mockkv_plugin.cpp` is the plugin boundary. NIXL discovers `libplugin_MOCKKV.so`, opens it dynamically, and looks for:

```cpp
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *nixl_plugin_init();
extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini();
```

`nixl_plugin_init()` returns plugin metadata and an engine factory through `nixlBackendPluginCreator<nixlMockKVEngine>::create(...)`.

For MOCKKV, the registration says:

```cpp
mockkv_plugin_t::create(
    NIXL_PLUGIN_API_VERSION,
    "MOCKKV",
    "0.1.0",
    {},
    {DRAM_SEG}
);
```

That means:

- The backend name is `MOCKKV`.
- It uses the current NIXL plugin API version.
- It supports `DRAM_SEG` descriptors.
- NIXL should construct `nixlMockKVEngine` when an agent creates this backend.

## Backend Engine Contract

`nixlMockKVEngine` derives from `nixlBackendEngine`. A real backend does not need to implement every possible NIXL feature, but it must describe what it supports and implement the methods needed for that support.

MOCKKV advertises:

```cpp
supportsLocal()  == true
supportsRemote() == false
supportsNotif()  == false
```

This makes MOCKKV a local-only backend. There is no remote connection protocol, no cross-process metadata exchange, and no notification mechanism. That is deliberate: it keeps the example focused on the core lifecycle.

The key methods to study are:

- `registerMem`
- `deregisterMem`
- `prepXfer`
- `postXfer`
- `checkXfer`
- `releaseReqH`
- `getSupportedMems`
- `loadLocalMD`
- `unloadMD`

## Data Model

MOCKKV stores opaque bytes in a process-local map:

```cpp
std::unordered_map<std::string, std::vector<uint8_t>> kv_store_;
```

Each registered memory descriptor is interpreted as a key. The key comes from:

1. `mem.metaInfo`, when provided
2. `std::to_string(mem.devId)`, as a fallback

The backend also keeps a lookup table:

```cpp
std::unordered_map<uint64_t, std::string> devIdToKey_;
```

That lets `postXfer()` resolve a key from the remote descriptor's `devId` when metadata is not available directly.

## Lifecycle

A typical MOCKKV flow is:

```text
1. create backend
2. register memory descriptors
3. prepare transfer
4. post transfer
5. check transfer status
6. release transfer request handle
7. deregister memory descriptors
8. unload metadata during cleanup, if core asks for it
```

The same flow with method names:

```text
nixlAgent::createBackend("MOCKKV", ...)
  -> nixlMockKVEngine(...)

nixlAgent::registerMem(...)
  -> nixlMockKVEngine::registerMem(...)

nixlAgent::createXferReq(...) or related transfer setup
  -> nixlMockKVEngine::prepXfer(...)

nixlAgent::postXferReq(...)
  -> nixlMockKVEngine::postXfer(...)

nixlAgent::getXferStatus(...)
  -> nixlMockKVEngine::checkXfer(...)

nixlAgent::releaseXferReq(...)
  -> nixlMockKVEngine::releaseReqH(...)

nixlAgent::deregisterMem(...)
  -> nixlMockKVEngine::deregisterMem(...)
```

## Step-by-Step

### 1. Register Memory

`registerMem()` is where MOCKKV turns a NIXL descriptor into backend metadata.

It does four things:

1. Verifies that the memory type is `DRAM_SEG`.
2. Chooses a key from `metaInfo` or `devId`.
3. Allocates `nixlMockKVMetadata` containing `devId` and key.
4. Records `devId -> key` in `devIdToKey_`.

The returned `nixlBackendMD *` is owned by the NIXL local section path and is freed later by `deregisterMem()`.

### 2. Prepare Transfer

`prepXfer()` checks that the request is something MOCKKV understands:

- operation is `NIXL_WRITE` or `NIXL_READ`
- local descriptor list uses `DRAM_SEG`
- remote descriptor list uses `DRAM_SEG`

MOCKKV is synchronous, so it does not need a real asynchronous work object. It still allocates a small request handle because the NIXL backend interface expects one.

### 3. Post Transfer

`postXfer()` is where the data moves.

For `NIXL_WRITE`:

```text
local user buffer -> kv_store_[key]
```

For `NIXL_READ`:

```text
kv_store_[key] -> local user buffer
```

The key is resolved from remote metadata when possible. If metadata is not present, the backend uses `remote_desc.devId` and `devIdToKey_`.

### 4. Check Transfer

`checkXfer()` always returns `NIXL_SUCCESS` because `postXfer()` already completed the operation.

An asynchronous backend would usually poll hardware, a queue, a service, or an internal state machine here.

### 5. Release Request Handle

`releaseReqH()` deletes the placeholder request handle allocated by `prepXfer()`.

### 6. Deregister Memory

`deregisterMem()` removes the `devId -> key` mapping and frees the metadata object.

This is the ownership point to pay attention to when writing a backend: if `registerMem()` allocates backend metadata, `deregisterMem()` should release it.

### 7. Unload Metadata

`unloadMD()` is a no-op in MOCKKV. This mirrors simple local/storage-style backends where metadata is released through deregistration rather than remote-section teardown.

Do not use `unloadMD()` to free the same metadata that `deregisterMem()` owns, or you risk double-free behavior.

## Why nixlbench Has MOCKKV-Specific Handling

NIXL can discover and load MOCKKV dynamically through `NIXL_PLUGIN_DIR`, but nixlbench still has to construct descriptor lists that match the backend.

Most nixlbench storage backends use file-like descriptors:

```text
FILE_SEG + file descriptor + file offset
```

MOCKKV uses:

```text
DRAM_SEG + key in metaInfo
```

So nixlbench needs guarded `NIXLBENCH_ENABLE_MOCKKV` code in a few places:

- allocate/register MOCKKV descriptors as `DRAM_SEG`
- deregister MOCKKV descriptors as `DRAM_SEG`
- exchange IOVs without file descriptors
- prepare remote transfer descriptors as `DRAM_SEG`

That benchmark glue is not part of the plugin API itself. It is only needed because nixlbench has backend-specific descriptor setup logic.

## What To Copy For A New Plugin

Useful patterns to copy:

- Keep plugin registration small and explicit in a `*_plugin.cpp` file.
- Put backend behavior in a `nixlBackendEngine` subclass.
- Clearly state supported memory types in `getSupportedMems()` and plugin registration.
- Allocate backend metadata in `registerMem()` and free it in `deregisterMem()`.
- Validate descriptor types and operation kinds in `prepXfer()`.
- Keep request-handle ownership clear: allocate in `prepXfer()`, release in `releaseReqH()`.
- Return `NIXL_SUCCESS` from `checkXfer()` only if work is actually complete.

Things that are example-specific:

- The in-process `unordered_map`
- The `metaInfo` key convention
- The synchronous `postXfer()` implementation
- The lack of locking
- The lack of remote support and notifications

## File Map

```text
examples/plugins/mockkv/
  meson.build               build-tree-only dynamic plugin target
  mockkv_plugin.cpp         plugin entry points and backend registration
  mockkv_backend.h          backend class declaration and API summary
  mockkv_backend.cpp        backend implementation
  README.md                 quick start and usage notes
  MOCKKV_ARCHITECTURE.md    this guide
```

## Reading Checklist

When learning the code, read in this order:

1. `mockkv_plugin.cpp`: how NIXL learns the backend name and factory.
2. `mockkv_backend.h`: what interface the engine implements.
3. `registerMem()` in `mockkv_backend.cpp`: how descriptors become backend metadata.
4. `prepXfer()` and `postXfer()`: how a NIXL request becomes a PUT or GET.
5. `releaseReqH()` and `deregisterMem()`: how ownership is cleaned up.
6. The guarded MOCKKV code in nixlbench, if you want to see how an application prepares descriptors for this backend.
