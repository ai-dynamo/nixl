# Descriptors And Memory Registration

Use this reference for C++ descriptor classes, descriptor lists, memory type
selection, registration, deregistration, and memory query review.

## Source Anchors

Fallback snapshot `b293d9bf2d192b321ee24b1988cf1b6b51875331`:

- `src/api/cpp/nixl_types.h`: memory types `DRAM_SEG`, `VRAM_SEG`,
  `BLK_SEG`, `OBJ_SEG`, `FILE_SEG`.
- `src/api/cpp/nixl_descriptors.h`: `nixlBasicDesc`, `nixlBlobDesc`,
  `nixlRemoteDesc`, `nixlDescList<T>`, `nixl_xfer_dlist_t`,
  `nixl_reg_dlist_t`, `nixl_remote_dlist_t`, `nixl_local_dlist_t`.
- `src/api/cpp/nixl.h`: `registerMem()`, `deregisterMem()`, `queryMem()`.
- `docs/BackendGuide.md`: descriptor-list abstraction and memory/storage
  descriptor fields.
- `examples/cpp/nixl_example.cpp`: DRAM `calloc()` registration example.

Confirm against the user's installed headers before copying code.

## Descriptor Types

- `nixlBasicDesc` carries `addr`, `len`, and `devId`.
- `nixlBlobDesc` extends `nixlBasicDesc` with `metaInfo` and is used by
  `nixl_reg_dlist_t` for registration.
- `nixlRemoteDesc` extends `nixlBasicDesc` with `remoteAgent` and is used by
  memory-view preparation.
- `nixl_xfer_dlist_t` is a descriptor list of `nixlBasicDesc` for transfer
  source/destination descriptors.
- `nixl_reg_dlist_t` is a descriptor list of `nixlBlobDesc` for registration
  and deregistration.

## Registration Checklist

1. Verify the selected backend supports the requested `nixl_mem_t`.
2. Build registration descriptors from trusted, application-owned allocations
   or storage handles.
3. Keep the allocation alive until transfers are complete, request handles are
   released, and the descriptor is deregistered by its owner.
4. Use backend hints when the operation must be restricted to known backend
   handles:

    ```cpp
    nixl_opt_args_t args;
    args.backends.push_back(backend);
    ```

5. Register with `registerMem(reg_list, &args)`.
6. Deregister with the matching descriptors and backend hints from the owner
   process.

## DRAM Pattern

Use this only after source/backend evidence is available:

```cpp
const size_t len = 4096;
void *buf = calloc(1, len);
if (buf == nullptr) {
    return NIXL_ERR_INVALID_PARAM;
}

nixl_reg_dlist_t reg(DRAM_SEG);
nixlBlobDesc desc;
desc.addr = reinterpret_cast<uintptr_t>(buf);
desc.len = len;
desc.devId = 0;
reg.addDesc(desc);

nixl_opt_args_t args;
args.backends.push_back(backend);

nixl_status_t st = agent.registerMem(reg, &args);
```

## Review Findings To Raise

- A descriptor built from a pointer found in a log, chat, config file, or
  unauthenticated request is unsafe.
- `len == 0`, stale pointers, freed buffers, stack buffers that go out of
  scope, or descriptor ranges outside the registered allocation are unsafe.
- A VRAM descriptor needs source-backed device allocation and `devId` evidence.
- File, object, block, GDS, POSIX, or storage metadata must come from the
  installed backend source/docs/examples; do not invent `metaInfo` layouts.
- The initiator process cannot deregister memory owned by another process. Each
  process should deregister only descriptors it registered.

## Query Memory

`queryMem()` requires backend context through `extra_params` in the fallback
header. Use it for inspection only after the backend handle and descriptor
ownership are known.
