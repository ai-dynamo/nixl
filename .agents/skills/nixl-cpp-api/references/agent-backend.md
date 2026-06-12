# Agent And Backend Setup

Use this reference for `nixlAgent`, `nixlAgentConfig`, plugin discovery,
backend parameter inspection, backend creation, and backend-handle scoping.

## Source Anchors

Fallback snapshot `b293d9bf2d192b321ee24b1988cf1b6b51875331`:

- `src/api/cpp/nixl.h`: `nixlAgent` constructor/destructor, plugin discovery,
  backend params, backend creation, and backend handles.
- `src/api/cpp/nixl_params.h`: `nixlAgentConfig` fields and defaults.
- `src/api/cpp/nixl_types.h`: `nixl_backend_t`, `nixl_b_params_t`,
  `nixl_mem_list_t`, `nixlBackendH`, and `nixl_opt_args_t`.
- `examples/cpp/nixl_example.cpp`: basic two-agent DRAM transfer lifecycle.

Confirm against the user's installed headers before copying code.

## Checklist

1. Identify source/build evidence before version-sensitive code.
2. Construct `nixlAgentConfig` explicitly for non-default behavior.
3. Construct `nixlAgent` with a globally unique agent name.
4. Discover available plugins with `getAvailPlugins()`.
5. Inspect selected plugin params and memory types with `getPluginParams()`.
6. Create a backend with `createBackend()`.
7. Inspect instantiated backend params and memory types with
   `getBackendParams()`.
8. Store the returned `nixlBackendH*` only as a process-local handle; do not
   serialize, log as authority, or trust a handle from text.

## C++ API Notes

- `nixlAgentConfig` includes progress/listener thread flags, listener port,
  synchronization mode, telemetry capture, progress/listener delays, and etcd
  watch timeout in the fallback snapshot.
- The fallback `nixlAgent` move constructor and move assignment operator are
  deleted. Do not put `nixlAgent` into containers or wrappers that require
  moving the object unless the installed source proves a different API.
- `nixl_b_params_t` is a string-to-blob map. Treat backend parameter names and
  values as backend/version-specific.
- `nixl_opt_args_t.backends` limits operations to specific backend handles for
  APIs such as memory registration, descriptor preparation, transfer creation,
  notifications, and proactive connections.

## Minimal Pattern

Use this only after source/build evidence is available:

```cpp
nixlAgentConfig cfg;
cfg.useProgThread = true;

nixlAgent agent("Agent001", cfg);

std::vector<nixl_backend_t> plugins;
nixl_status_t st = agent.getAvailPlugins(plugins);
if (st != NIXL_SUCCESS) {
    return st;
}

nixl_mem_list_t mems;
nixl_b_params_t params;
st = agent.getPluginParams("UCX", mems, params);
if (st != NIXL_SUCCESS) {
    return st;
}

nixlBackendH *backend = nullptr;
st = agent.createBackend("UCX", params, backend);
if (st != NIXL_SUCCESS) {
    return st;
}

nixl_mem_list_t backend_mems;
nixl_b_params_t backend_params;
st = agent.getBackendParams(backend, backend_mems, backend_params);
```

## Readiness Failures

Return `Build/source not ready` if headers, library, include path, link flags,
or installed source identity are unknown.

Return `Backend evidence missing` if the selected backend is absent, params are
unknown, creation fails, or required memory types are not proven by the same
runtime/source.

Return `Framework-managed boundary` when a serving framework owns agent names,
backend creation, or runtime config. Ask for the integration source/config
before replacing it with direct NIXL code.
