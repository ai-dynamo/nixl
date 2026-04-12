<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dell ObjectScale Engine — Pattern B Refactoring

## What changed

This PR replaces the cuObject **callback-based I/O** (Pattern A) with
**manual RDMA token management** (Pattern B) in the Dell ObjectScale
engine.  The cuObject API spec (v1.0.0) documents both patterns; this
PR switches from section 1.6/1.10 to section 1.12.4.

## Why

The current code uses `cuObjPut()` / `cuObjGet()` — synchronous I/O
calls that invoke `CUObjIOOps` callbacks.  The callbacks are expected
to perform the **entire server round-trip** and return the byte count
transferred.  Our callbacks do neither: they copy `infop->desc_str`
into a context struct, return `0`, and the actual S3 request happens
later in `postXfer`. 

Pattern B's `cuMemObjGetRDMAToken()` is designed exactly for our use
case: generate a token, use it in your own HTTP request, free it when
done.  No callbacks, no synchronous I/O pretense.

## How it works

**Before (Pattern A):**
```
prepXfer:  cuObjPut/cuObjGet → callback extracts rdma_desc → store in request handle
postXfer:  dynamic_cast to iDellS3RdmaClient → putObjectRdmaAsync(rdma_desc)
```

**After (Pattern B):**
```
prepXfer:  inherited from parent (creates empty handle, validates params)
postXfer:  inherited from parent → calls putObjectAsync/getObjectAsync
           → Dell client generates token inline via cuMemObjGetRDMAToken
           → injects x-rdma-info header → sends S3 request
```

## Files changed and why

| File | What | Why needed |
|------|------|------------|
| **cuobj_token_manager.h/.cpp** (NEW) | RAII wrapper: `registerMemory`, `deregisterMemory`, `generatePutToken`, `generateGetToken` | Replaces `CUObjIOOps` callbacks, `rdma_ctx_t`, `objectGet/objectPut` statics, and `obs_ops` global. Pool-aware registration with refcounting so per-page `registerMem` calls from NIXL core map to one `cuMemObjGetDescriptor`. |
| **client.h/.cpp** (REWRITE) | Overrides standard `putObjectAsync`/`getObjectAsync` with RDMA token injection | Replaces `iDellS3RdmaClient` separate interface and its `putObjectRdmaAsync`/`getObjectRdmaAsync`. Eliminates the `dynamic_cast` in `postXfer`. The Dell client now fulfills the standard `iS3Client` contract — the parent's `postXfer` calls it without knowing RDMA is involved. |
| **engine_impl.h/.cpp** (SIMPLIFY) | Only overrides `registerMem`, `deregisterMem`, `getSupportedMems`, `getClient` | Removes `prepXfer`, `postXfer`, `checkXfer`, `releaseReqH` overrides (inherited from `DefaultObjEngineImpl`). Removes 5 helper classes: `obsObjTransferRequestH`, `nixlObsObjBackendReqH`, `nixlObsObjMetadata`, `rdma_ctx_t`, `isValidPrepXferParams` duplicate. |
| **rdma_interface.h** (DELETE) | Removes `iDellS3RdmaClient` interface | No longer needed — Dell client implements the standard `iS3Client` interface. |
| **s3/engine_impl.cpp** (4 lines) | `isValidPrepXferParams` accepts `VRAM_SEG` | Dell engine reports `VRAM_SEG` in `getSupportedMems()` but the parent's validation only accepted `DRAM_SEG`. |
| **meson.build** (2 lines) | Add `cuobj_token_manager.{h,cpp}` to build | New source files. |
| **test/obj.cpp** (-62 lines) | Remove `mockDellS3Client` | Standard `mockS3Client` works for all engines now since Dell uses the same `iS3Client` interface. |

## Key design choice: token generation in putObjectAsync, not prepXfer

The original design generates RDMA descriptors in `prepXfer` (via
`cuObjPut`/`cuObjGet` callbacks) and stores them in custom request
handles for use in `postXfer`.  This PR generates tokens inside
`putObjectAsync`/`getObjectAsync` — at the point of use.

This eliminates the need for custom request handle classes to carry
descriptors between `prepXfer` and `postXfer`, and allows the Dell
engine to inherit the parent's entire transfer pipeline unchanged.

## Pool-aware memory registration

NIXL's core calls `backend->registerMem()` once per page.  For a 2 GB
buffer with 1 MB pages, that is 2,048 calls.  `cuMemObjGetDescriptor()`
is heavyweight (pins memory, creates NIC Memory Regions), so doing it
2,048 times takes ~1–2 seconds.

`CuObjTokenManager::registerMemory()` takes an optional `pool_hint`
parameter.  On the first call, it registers `[ptr, ptr+pool_hint)`
instead of just `[ptr, ptr+page_size)`.  Subsequent calls check
`findContainingRegion()` — if the address is already within the
registered pool, it increments a refcount and returns immediately.

Result: 1 `cuMemObjGetDescriptor` call instead of 2,048.  The
`pool_hint` is passed from the Dell engine via the `rdma_pool_size`
backend parameter, which callers (e.g. LMCache) set to their allocator
buffer size.  Token generation (`cuMemObjGetRDMAToken`) computes the
pool-relative offset automatically.

Deregistration is refcount-based: `cuMemObjPutDescriptor` fires only
when the last page within a pool is deregistered.

## Net result

| Metric | Before | After |
|--------|--------|-------|
| Dell-specific lines | ~900 | ~360 |
| Engine methods overridden | 6 | 3 |
| Helper classes | 5 | 1 (`nixlDellMemMetadata`) |
| `dynamic_cast` at runtime | Yes | No |
| Separate RDMA interface | `iDellS3RdmaClient` | None (standard `iS3Client`) |
