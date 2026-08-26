# Memory Views

Use this reference for `nixlMemViewH`, local and remote memory-view
preparation, and release. This is an advanced path: require matching source and
device/backend evidence before giving device-side code.

## Source Anchors

Fallback snapshot `b293d9bf2d192b321ee24b1988cf1b6b51875331`:

- `src/api/cpp/nixl.h`: `prepMemView()` overloads and `releaseMemView()`.
- `src/api/cpp/nixl_types.h`: `nixlMemViewH`.
- `src/api/cpp/nixl_descriptors.h`: `nixl_remote_dlist_t`,
  `nixl_local_dlist_t`, and `nixlRemoteDesc`.
- `src/api/gpu/ucx/nixl_device.cuh`: device API references that expect memory
  views prepared on the host.

Confirm against the user's installed headers and backend/device API before
copying code.

## Readiness Requirements

Before writing memory-view code, verify:

- The user's installed source exposes `prepMemView()` and `releaseMemView()`.
- The selected backend supports the requested memory-view path.
- Remote metadata is loaded for remote descriptors.
- Local and remote descriptors refer to registered memory.
- Device-side API calls and headers are version-matched.
- The `nixlMemViewH` lifetime is owned and released by the same agent/source
  path that prepared it.

## Host-Side Shape

Remote memory-view preparation uses `nixl_remote_dlist_t`:

```cpp
nixl_remote_dlist_t remote(VRAM_SEG);
remote.addDesc(nixlRemoteDesc(remote_addr, bytes, remote_dev_id, remote_agent));

nixlMemViewH mvh = nullptr;
nixl_status_t st = agent.prepMemView(remote, mvh, &args);
if (st != NIXL_SUCCESS) {
    return st;
}

agent.releaseMemView(mvh);
```

Local memory-view preparation uses `nixl_local_dlist_t`:

```cpp
nixl_local_dlist_t local(VRAM_SEG);
local.addDesc(nixlBasicDesc(local_addr, bytes, local_dev_id));

nixlMemViewH mvh = nullptr;
nixl_status_t st = agent.prepMemView(local, mvh, &args);
```

These snippets are shape guidance only until verified against the user's
installed headers and backend/device API.

## Stop Conditions

Stop and ask for source evidence if the user asks for:

- Exact `nixlPut`, `nixlAtomicAdd`, or device-kernel code.
- Memory views over storage, object, block, or mixed memory types.
- Cross-process lifetime rules for `nixlMemViewH`.
- Atomic ordering, cancellation, retry, or synchronization semantics.
