---
title: Building a Backend Plugin
description: Step-by-step guide to implementing a custom NIXL backend plug-in using the POSIX plug-in as a reference.
---

Build a custom NIXL backend plug-in from scratch, using the POSIX plug-in as a teaching example. A backend plug-in implements the Southbound API (SB API) to add support for a new transfer mechanism -- whether that is a networking transport, a GPU-direct storage interface, or a local file I/O system.

Plug-in development is C++ only. All backend plug-ins inherit from `nixlBackendEngine` (defined in `src/api/cpp/backend/backend_engine.h`) and override the required virtual methods. The Plugin Manager handles loading your plug-in at runtime, discovering it from a shared library that exports a standard entry point. For the complete method-by-method SB API reference, see the [SB API Reference](./sb-api-reference). For the user-facing API that triggers these backend calls, see the [C++ API Reference](../api-reference/cpp-api).

First, understand how user-facing (NB) API calls map to the backend (SB) API methods that your plug-in implements.

## NB-to-SB API Mapping

When a user interacts with a `nixlAgent`, the Transfer Agent translates those high-level operations into SB API calls on the appropriate backend plug-in. The following table shows this mapping:

| User Action (NB API) | SB API Call(s) |
|---|---|
| `agent.createBackend("POSIX", params)` | `nixl_plugin_init()` then `create_engine(init_params)` |
| `agent.registerMem(descs, backend)` | `backend.registerMem(mem, nixl_mem, out)` per descriptor |
| `agent.getLocalMD(blob)` | `backend.getPublicData(meta, str)` per registered memory |
| `agent.loadRemoteMD(blob)` | `backend.loadRemoteConnInfo(...)` + `backend.loadRemoteMD(...)` |
| `agent.postXferReq(req)` | `backend.prepXfer(...)` then `backend.postXfer(...)` |
| `agent.getXferStatus(req)` | `backend.checkXfer(handle)` |
| `agent.releaseXferReq(req)` | `backend.releaseReqH(handle)` |
| `agent.deregisterMem(descs, backend)` | `backend.deregisterMem(meta)` per descriptor |
| `agent.getNotifs(notifs)` | `backend.getNotifs(notif_list)` per backend |
| `agent.connect(remote)` | `backend.connect(remote_agent)` per common backend |

<Tip>
See the [C++ API Reference](../api-reference/cpp-api) for full NB API documentation and the [Quick Start](../getting-started/quick-start) guide for the user workflow.
</Tip>

Users call the NB API, the Transfer Agent routes calls to the appropriate backend, and your plug-in handles the data transfer through SB API methods.

## The POSIX Plugin

The POSIX plug-in is the simplest backend in NIXL, making it an ideal learning example. It provides local file I/O using asynchronous POSIX interfaces (io_uring, Linux AIO, or POSIX AIO depending on system availability).

**Capabilities:**

| Property | Value |
|----------|-------|
| `supportsLocal` | `true` -- file I/O is local to the agent |
| `supportsRemote` | `false` -- no inter-agent communication |
| `supportsNotif` | `false` -- no notification support |
| Supported memory types | `FILE_SEG`, `DRAM_SEG` |
| Plugin name | `"POSIX"` |
| Plugin version | `"0.1.0"` |

The POSIX plug-in is ideal for learning because it is a storage backend that only needs local transfer methods. Methods like `connect()`, `disconnect()`, and `getPublicData()` simply return `NIXL_SUCCESS` or are not needed at all, letting you focus on the core transfer lifecycle: registering memory, preparing transfers, submitting I/O, checking completion, and cleaning up. The POSIX plug-in has only 6-7 substantive method implementations, compared to a full network plug-in like UCX which implements all methods including remote metadata, notifications, and inter-agent connections.

## Key Method Implementations

This section walks through the 7 key methods of the POSIX plug-in. For each method, the actual source code is shown with annotations explaining the patterns and decisions. These are the methods you would customize when building your own backend.

### Constructor

The constructor initializes the backend engine with user-provided parameters. For the POSIX plug-in, this means selecting the async I/O queue type based on custom parameters.

```cpp
nixlPosixEngine::nixlPosixEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      io_queue_type_(getIoQueueType(init_params->customParams)),
      io_queue_(nixlPosixIOQueue::instantiate(io_queue_type_,
                                              getIOSPoolSize(init_params->customParams),
                                              getKernelQueueSize(init_params->customParams))),
      io_queue_lock_(init_params->syncMode) {
    if (io_queue_type_.empty()) {
        initErr = true;
        NIXL_ERROR << "Failed to initialize POSIX backend - no supported io queue type found";
        return;
    }
    NIXL_INFO << absl::StrFormat("POSIX backend initialized using io queue type: %s",
                                 io_queue_type_);
}
```

**Key patterns:**
- Always call the base class constructor `nixlBackendEngine(init_params)` first -- it initializes `localAgent`, `backendType`, `customParams`, and `enableTelemetry_`.
- Read custom parameters from `init_params->customParams` (a `std::map<std::string, std::string>`) to configure backend-specific behavior. Here the POSIX plug-in reads `use_aio`, `use_uring`, and `use_posix_aio` parameters.
- Set `initErr = true` if initialization fails. The Plugin Manager checks this via `getInitErr()` and will reject the backend if it reports an error.
- Allocate internal resources (here, the I/O queue) during construction so they are ready for transfer operations.

### registerMem

Registers a memory region with the backend. The agent calls this once per descriptor when the user registers memory.

```cpp
nixl_status_t
nixlPosixEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) != supported_mems.end())
        return NIXL_SUCCESS;

    return NIXL_ERR_NOT_SUPPORTED;
}
```

**Key patterns:**
- This is the simplest possible registration implementation -- it validates that the memory type is supported and does nothing else. The `out` parameter is left as `nullptr` since the POSIX plug-in does not need backend-specific metadata for registered memory.
- More complex backends (like UCX) would create a metadata object, store it in `out`, and later use it during transfer operations for memory keys, handles, or registration tokens.
- Return `NIXL_SUCCESS` to accept the memory or `NIXL_ERR_NOT_SUPPORTED` to reject it.

### loadLocalMD

Produces target-side metadata for local (within-agent) transfers. Some backends need different metadata for the initiator and target sides of a transfer.

```cpp
nixl_status_t
loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
    output = input;
    return NIXL_SUCCESS;
}
```

**Key patterns:**
- For local-only storage backends, this can be trivial: just return the input pointer as the output. This means the same metadata object is used for both initiator and target sides.
- Network backends that support local transfers (like UCX) might create a separate metadata object here for target-side buffer access.
- This method is only called when `supportsLocal()` returns `true`.

### prepXfer

Prepares a transfer by validating parameters and creating a backend-specific request handle.

```cpp
nixl_status_t
nixlPosixEngine::prepXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    try {
        auto posix_handle =
            std::make_unique<nixlPosixBackendReqH>(operation, local, remote, opt_args, io_queue_);
        NIXL_LOCK_GUARD(io_queue_lock_);
        nixl_status_t status = posix_handle->prepXfer();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        handle = posix_handle.release();
        return NIXL_SUCCESS;
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << absl::StrFormat("Error: %s", e.what());
        return e.code();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << absl::StrFormat("Unexpected error: %s", e.what());
        return NIXL_ERR_BACKEND;
    }
}
```

**Key patterns:**
- Validate parameters before doing any work. The POSIX plug-in checks that local descriptors are DRAM, remote descriptors are FILE, counts match, and the remote agent is actually the local agent (since POSIX is local-only).
- Create a backend-specific request handle class (here `nixlPosixBackendReqH`) that inherits from `nixlBackendReqH`. This handle stores all state needed for the transfer.
- Use `std::make_unique` for exception safety, then call `release()` to transfer ownership to the caller via the `handle` output parameter.
- Wrap handle creation in try/catch to convert exceptions into `nixl_status_t` error codes.

### postXfer

Submits the transfer for execution. The agent calls this after `prepXfer` to actually initiate the I/O operations.

```cpp
nixl_status_t
nixlPosixEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    try {
        auto &posix_handle = castPosixHandle(handle);
        NIXL_LOCK_GUARD(io_queue_lock_);
        nixl_status_t status = posix_handle.postXfer();
        if (status != NIXL_IN_PROG) {
            NIXL_ERROR << "Error in submitting queue";
        }
        return status;
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    return NIXL_ERR_BACKEND;
}
```

**Key patterns:**
- Cast the generic `nixlBackendReqH*` handle back to your backend-specific type using `dynamic_cast` (the `castPosixHandle` helper validates and casts).
- Delegate the actual I/O submission to the request handle object. The POSIX handle iterates over descriptors and enqueues each I/O operation to the async I/O queue.
- Return `NIXL_IN_PROG` on successful submission to indicate the transfer is now in progress and the caller should poll with `checkXfer`.
- Thread safety is managed with `NIXL_LOCK_GUARD` since multiple transfers might share the same I/O queue.

### checkXfer

Polls for transfer completion. The agent calls this repeatedly until the transfer finishes.

```cpp
nixl_status_t
nixlPosixEngine::checkXfer(nixlBackendReqH *handle) const {
    try {
        auto &posix_handle = castPosixHandle(handle);
        NIXL_LOCK_GUARD(io_queue_lock_);
        return posix_handle.checkXfer();
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    return NIXL_ERR_BACKEND;
}
```

**Key patterns:**
- Return `NIXL_SUCCESS` when all I/O operations have completed, `NIXL_IN_PROG` when the transfer is still running, or a negative error code on failure.
- The POSIX handle internally tracks the number of confirmed I/O completions against the total queue depth. Each call to `checkXfer` polls the I/O queue for new completions.
- This method should be lightweight since the agent may call it in a tight polling loop.

### releaseReqH

Releases the backend-specific request handle and frees associated resources.

```cpp
nixl_status_t
nixlPosixEngine::releaseReqH(nixlBackendReqH *handle) const {
    try {
        auto &posix_handle = castPosixHandle(handle);
        posix_handle.~nixlPosixBackendReqH();
        return NIXL_SUCCESS;
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    return NIXL_ERR_BACKEND;
}
```

**Key patterns:**
- Cast the handle and explicitly call its destructor to clean up resources. The agent allocated the handle in `prepXfer`, and this is where it gets freed.
- Always return `NIXL_SUCCESS` on successful cleanup, even if there is nothing to free. The agent relies on this to know it can safely discard the handle pointer.
- This is called after the transfer completes (or is aborted), so the handle should not have any pending I/O at this point.

<Note>
Methods like `connect()`, `disconnect()`, and `deregisterMem()` simply return `NIXL_SUCCESS` in the POSIX plug-in since file I/O does not require remote connections or special deregistration. When building a network backend (like UCX), these methods would manage transport-level connections and memory deregistration with the networking stack.
</Note>

## Plugin Manager API

Every backend plug-in must expose a set of entry points so the Plugin Manager can discover its name, version, supported memory types, and how to create and destroy engine instances. These are defined in the `nixlBackendPlugin` struct (from `src/api/cpp/backend/backend_plugin.h`).

The `nixlBackendPluginCreator<EngineType>` template handles all of this automatically. You provide your engine class as the template parameter, and the template generates the required function pointers.

### Entry Point Methods

| Method | Description | How `nixlBackendPluginCreator` handles it |
|--------|-------------|-------------------------------------------|
| `get_plugin_name` | Returns the plug-in's display name (e.g., `"POSIX"`) | Stores the name string passed to `create()` and returns it via a lambda |
| `get_plugin_version` | Returns the plug-in's version string (e.g., `"0.1.0"`) | Stores the version string passed to `create()` and returns it via a lambda |
| `create_engine` | Creates a new engine instance from init params | Calls `new EngineType(init_params)` (or uses a factory for UCX) |
| `destroy_engine` | Destroys an engine instance | Calls `delete engine` |
| `get_backend_mems` | Returns the list of supported memory types | Stores the `nixl_mem_list_t` passed to `create()` and returns it via a lambda |
| `get_backend_options` | Returns the custom parameter names the backend accepts | Stores the `nixl_b_params_t` passed to `create()` and returns it via a lambda |

### API Version

The `nixlBackendPlugin` struct includes an `api_version` field, currently set to `NIXL_PLUGIN_API_VERSION` (value: `1`). This version number ensures forward and backward compatibility: the Plugin Manager can check the API version before calling any function pointers, allowing NIXL to evolve the plug-in interface without breaking existing plug-ins.

<Warning>
Always pass `NIXL_PLUGIN_API_VERSION` when creating your plug-in. Do not hardcode a numeric value -- the macro ensures your plug-in tracks the current API version at compile time.
</Warning>

## Plugin Entry Points

Your plug-in shared library must export two C-linkage functions that the Plugin Manager calls to initialize and clean up the plug-in. The POSIX plug-in's entry point file (`posix_plugin.cpp`) shows the complete pattern:

```cpp
#include <memory>
#include "posix_backend.h"
#include "backend/backend_plugin.h"

// Plugin type alias for convenience
using posix_plugin_t = nixlBackendPluginCreator<nixlPosixEngine>;

#ifdef STATIC_PLUGIN_POSIX
nixlBackendPlugin *
createStaticPOSIXPlugin() {
    return posix_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "POSIX", "0.1.0", {}, {DRAM_SEG, FILE_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return posix_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "POSIX", "0.1.0", {}, {DRAM_SEG, FILE_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
```

**Key elements:**

- **Type alias:** `using posix_plugin_t = nixlBackendPluginCreator<nixlPosixEngine>` creates a convenience alias for the plug-in creator template specialized with your engine class.

- **`nixl_plugin_init()`:** This is the main entry point called by the Plugin Manager when loading the plug-in. It must return a `nixlBackendPlugin*` created via the `nixlBackendPluginCreator::create()` method. Pass your plug-in name, version, backend options (empty `{}` if none), and supported memory types. Both `extern "C"` (to prevent C++ name mangling) and `NIXL_PLUGIN_EXPORT` (to set symbol visibility) are required.

- **`nixl_plugin_fini()`:** Called when the plug-in is unloaded. For most plug-ins this is empty, but you can use it to clean up global resources if your plug-in allocates any during `nixl_plugin_init()`.

- **Static plug-in variant:** The `#ifdef STATIC_PLUGIN_POSIX` block provides `createStaticPOSIXPlugin()` for compile-time plug-in inclusion. Static plug-ins are linked directly into the NIXL binary rather than loaded from disk, providing slightly better performance at the expense of a larger binary. The function signature differs (no `extern "C"` or `NIXL_PLUGIN_EXPORT` needed) but the creation call is identical.

<Tip>
For your custom plug-in, replace `nixlPosixEngine` with your engine class, update the plug-in name and version strings, and list your supported memory types. The `nixlBackendPluginCreator` template handles all the boilerplate.
</Tip>

## Build Integration

NIXL uses Meson as its build system. Each plug-in is built as either a shared library (for dynamic loading) or a static library (for compile-time inclusion). The POSIX plug-in's `meson.build` shows the standard pattern:

```meson
plugin_deps = [nixl_infra, nixl_common_dep, file_utils_interface]

# Define base source files
posix_sources = [
    'posix_backend.cpp',
    'posix_backend.h',
    'posix_plugin.cpp',
    'io_queue.h',
    'io_queue.cpp'
]

compile_defs = []
plugin_link_args = []

# Conditional dependencies (plugin-specific)
if has_linux_aio
    compile_defs += ['-DHAVE_LINUXAIO']
    posix_sources += ['linux_aio_io_queue.cpp']
    plugin_deps += [ linux_aio_dep ]
endif

if has_io_uring
    compile_defs += ['-DHAVE_LIBURING']
    posix_sources += ['io_uring_io_queue.cpp']
    plugin_deps += [ io_uring_dep ]
endif

if 'POSIX' in static_plugins
    posix_backend_lib = static_library('POSIX',
        posix_sources,
        dependencies: plugin_deps,
        link_args: plugin_link_args,
        cpp_args: compile_defs + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')
else
    posix_backend_lib = shared_library('POSIX',
        posix_sources,
        dependencies: plugin_deps,
        link_args: plugin_link_args,
        cpp_args: compile_defs + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
endif

posix_backend_interface = declare_dependency(link_with: posix_backend_lib)
```

**Key settings for your plug-in:**

- **`shared_library('POSIX', ...)`** -- The first argument is the library's base name. Combined with `name_prefix: 'libplugin_'`, this produces `libplugin_POSIX.so` on Linux. Replace `'POSIX'` with your plug-in name.

- **`static_library('POSIX', ...)`** -- Built when your plug-in name appears in the `static_plugins` build option. Static plug-ins are linked into the NIXL binary at compile time. Note `install: false` since they are embedded in the main library.

- **`name_prefix: 'libplugin_'`** -- All NIXL plug-ins use this prefix convention. The Plugin Manager searches for libraries matching this pattern when discovering plug-ins.

- **`install_dir: plugin_install_dir`** -- Resolves to `{libdir}/plugins` (typically `/usr/local/lib/plugins` or similar). This is where the Plugin Manager looks for dynamic plug-ins at runtime.

- **`install_rpath: '$ORIGIN/..'`** -- Sets the runtime library search path relative to the plug-in's location, allowing the plug-in to find the main NIXL shared library without requiring `LD_LIBRARY_PATH`.

- **Dependencies:** Every plug-in needs `nixl_infra` and `nixl_common_dep` at minimum. Add your plug-in-specific dependencies (e.g., `io_uring_dep` for the POSIX plug-in, `ucx_dep` for the UCX plug-in).

<Note>
When creating a new plug-in, add a `meson.build` file in your plug-in's directory under `src/plugins/your_plugin/`, and register it from the parent `src/plugins/meson.build` with a `subdir('your_plugin')` call.
</Note>

## Plugin Discovery

The Plugin Manager finds and loads backend plug-ins at runtime through the following process:

1. **Search directories:** The Plugin Manager scans configurable directories for shared libraries. The default search path is `{libdir}/plugins` (the `plugin_install_dir` from the build system).

2. **Symbol verification:** For each shared library found, the Plugin Manager attempts to resolve the `nixl_plugin_init` symbol. Only libraries that export this symbol with the correct signature are considered valid plug-ins.

3. **Plug-in initialization:** The Plugin Manager calls `nixl_plugin_init()` to get the `nixlBackendPlugin` struct, which provides function pointers for creating engines, querying capabilities, and getting the plug-in name/version.

4. **Lifetime management:** Valid plug-ins remain loaded in memory for the application's lifetime. The `nixl_plugin_fini()` function is called during cleanup.

5. **Static plug-ins:** Plug-ins compiled with the `STATIC_PLUGIN_*` preprocessor flag are registered at build time rather than discovered at runtime. They use a dedicated creator function (e.g., `createStaticPOSIXPlugin()`) instead of dynamic symbol resolution. Static plug-ins provide slightly better performance (no dynamic loading overhead) at the expense of a larger binary and less flexibility.

<Warning>
Dynamic plug-ins must be compiled as position-independent code (`-fPIC`) and must export `nixl_plugin_init` and `nixl_plugin_fini` with `extern "C"` linkage and `NIXL_PLUGIN_EXPORT` visibility. Missing any of these will cause the Plugin Manager to silently skip your plug-in.
</Warning>

## Comparing Backends

Different backends implement different subsets of the SB API based on their capabilities. Understanding these patterns helps when deciding which methods your plug-in needs to implement substantively versus which can simply return `NIXL_SUCCESS`.

### Network Backends

Network plug-ins like UCX set all three capability flags (`supportsRemote`, `supportsLocal`, `supportsNotif`). They implement all SB API methods because they need to:
- Manage inter-agent connections (`connect`, `disconnect`)
- Exchange remote memory identifiers (`getPublicData`, `loadRemoteMD`)
- Handle cross-agent notifications (`getNotifs`, `genNotif`)
- Support both local and remote transfers

### Storage Backends

Storage plug-ins like GDS and POSIX only set `supportsLocal` to `true`. From the Transfer Agent's perspective, all storage access is local -- even for remote distributed storage, a local client on the agent handles the communication with the storage system. This means:

- No remote agent communication is needed, so `connect()` and `disconnect()` return `NIXL_SUCCESS` immediately
- No remote metadata exchange, so `getPublicData()`, `loadRemoteConnInfo()`, and `loadRemoteMD()` are not required
- No notification support, so `getNotifs()` and `genNotif()` are not required
- `loadLocalMD()` can return the input pointer unchanged (identity return)

The only methods that need substantive implementation are the core transfer lifecycle: `registerMem`, `deregisterMem`, `prepXfer`, `postXfer`, `checkXfer`, and `releaseReqH`.

<Tip>
If you are building a storage backend, the POSIX plug-in is your closest reference. If you are building a network backend, look at the UCX plug-in in `src/plugins/ucx/` for a complete example of remote metadata exchange, connection management, and notification handling.
</Tip>

### Capability Summary

| Capability | Network | Storage |
|------------|:---:|:---:|
| `supportsLocal` | Yes | Yes |
| `supportsRemote` | Yes | No |
| `supportsNotif` | Yes | No |
| Substantive methods | All | 6-7 core methods |
| Remote metadata | Full implementation | Not needed |
| Connection management | Full implementation | Returns `NIXL_SUCCESS` |

For the complete list of which methods are required for each capability flag combination, see the [Capability Matrix](./sb-api-reference#capability-matrix) in the SB API Reference.
