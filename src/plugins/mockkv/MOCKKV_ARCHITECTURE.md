# MOCKKV Plugin Architecture and NIXL Backend Overview

This document explains the NIXL plugin architecture, lifecycle, and how MOCKKV fits in.

---

## 1. Overall NIXL plugin architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Application (e.g. nixlbench)                                            │
│    - Parse CLI / config                                                  │
│    - Create nixlAgent, select backend (e.g. "MOCKKV")                    │
│    - Call agent->registerMem / prepXfer / postXfer / deregisterMem ...   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  nixlAgent (core)                                                        │
│    - Owns backend instances (nixlBackendEngine*)                         │
│    - Manages memory sections (local/remote descriptor lists)           │
│    - Routes by memory type (DRAM_SEG / FILE_SEG / OBJ_SEG, ...)         │
│    - Invokes backend registerMem / prepXfer / postXfer / deregisterMem   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  nixlBackendEngine (plugin impl., e.g. nixlMockKVEngine)                 │
│    - Implements nixlBackendEngine                                      │
│    - Exposes supportsRemote / supportsLocal / supportsNotif              │
│    - Implements registerMem, deregisterMem, prepXfer, postXfer, ...     │
│    - If supportsRemote: getConnInfo, loadRemoteConnInfo, getPublicData,  │
│      loadRemoteMD, connect, disconnect, getNotifs, genNotif, ...        │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Agent** chooses which backend to use and when to register, transfer, and deregister.
- **Backend** implements storage, PUT/GET, and how local/remote metadata is represented.

---

## 2. Backend lifecycle (call order)

Typical order when using MOCKKV with nixlbench:

| Step | Call | Notes |
|------|------|--------|
| 1. Create | `agent->createBackend("MOCKKV", ...)` | Loads plugin; constructs engine. If supportsRemote: getConnInfo; if supportsLocal: connect(local_agent). |
| 2. Register | `agent->registerMem(desc_list, ...)` | App registers buffers; backend allocates metadata (`nixlBackendMD*`), devId↔key mapping. |
| 3. Prep | `backend->prepXfer(op, local, remote, ...)` | Validates; creates `nixlBackendReqH*`; MOCKKV is sync — handle is a placeholder. |
| 4. Post | `backend->postXfer(op, local, remote, ...)` | **Actual PUT/GET**: WRITE copies local memory to store; READ copies from store to local. |
| 5. Check | `backend->checkXfer(handle)` | Async backends poll here; MOCKKV returns SUCCESS immediately. |
| 6. Release handle | `backend->releaseReqH(handle)` | Frees handle from prepXfer. |
| 7. Deregister | `agent->deregisterMem(desc_list, ...)` | Backend frees metadata and devId mapping (same as Obj/S3: `unique_ptr` in `deregisterMem`). |
| 8. Teardown | `nixlRemoteSection::~nixlRemoteSection()` etc. | Core may call `unloadMD(meta)`; **same as nixlObjEngine: `unloadMD` does not free**; freeing is on `nixlLocalSection` path via `deregisterMem`. |

Important:

- **`registerMem`**: Caller owns `nixlBackendMD*`; **free in `deregisterMem`** (Obj-style; `unloadMD` is no-op).
- **`prepXfer` / `postXfer` / `checkXfer` / `releaseReqH`**: One transfer cycle; MOCKKV completes PUT/GET inside `postXfer` synchronously.

---

## 3. Local-only (aligned with Obj plugin)

MOCKKV: `supportsRemote() == false`, `supportsLocal() == true`. Same class of behavior as **nixlObjEngine (S3)**: local section register/deregister, `loadLocalMD` passes through, `connect`/`disconnect` may be no-op. **No cross-process object protocol here**; the goal is to teach PUT/GET with an in-process map.

---

## 4. MOCKKV data-flow sketch

```
nixlbench creates key (metaInfo)
        │
        ▼
  registerMem(mem, DRAM_SEG)  ──►  nixlMockKVMetadata(devId, key)
        │                              devIdToKey_[devId]=key
        ▼
  prepXfer(WRITE/READ, local, remote)  ──►  validate + placeholder ReqH
        ▼
  postXfer(WRITE, local, remote)  ──►  kv_store_[key] (in-process)
        ▼
  checkXfer  ──►  NIXL_SUCCESS (sync)
  releaseReqH  ──►  delete handle
        ▼
  deregisterMem(meta)  ──►  devIdToKey_.erase(devId); unique_ptr deletes meta (same as Obj)
        ▼
  (if core calls) unloadMD(meta)  ──►  no-op, no delete (same as nixlObjEngine::unloadMD)
```

---

## 5. Log messages (troubleshooting)

| Log snippet | Meaning |
|-------------|---------|
| `MOCKKV backend initialized (in-memory only)` | Engine constructed |
| `MOCKKV: Local agent = ...` | Local agent name |
| `registerMem: type=DRAM_SEG, devId=..., metaInfo=...` | Segment type, device id, key |
| `registerMem: registered devId=... -> key=...` | devId→key mapping stored |
| `prepXfer: operation=..., local_count=..., remote_count=...` | WRITE/READ and counts |
| `postXfer: Starting WRITE (PUT) operation with N descriptor(s)` | Starting N writes |
| `postXfer: Local operation - using in-memory store` | Using kv_store_ |
| `postXfer: Found key in metadata: <key>` | Key from remote metadata |
| `postXfer: WRITE: Copying X bytes from buffer to key=...` | Bytes and target key |
| `postXfer: All N operations completed successfully` | Batch done |
| `releaseReqH: Releasing request handle` | Releasing xfer handle |
| `deregisterMem: removing devId=..., key=...` | Deregistering metadata |
| (none) | `unloadMD` logs nothing and frees nothing (Obj-style) |

Calling `deregisterMem` twice on the same raw pointer is undefined behavior, as with Obj.

---

## 6. MOCKKV-specific notes

1. **Storage**: `kv_store_[key]` = `vector<uint8_t>` in-process; no locking (same assumption as Obj: serialized backend calls).
2. **Flags**: `supportsRemote=false`, `supportsLocal=true`, `supportsNotif=false`.
3. **Memory types**: `getSupportedMems()` returns only `DRAM_SEG`, matching nixlbench.

---

## 7. Files

- **mockkv_backend.h**: Class declaration and API summaries.
- **mockkv_backend.cpp**: ctor, registerMem/deregisterMem, prepXfer/postXfer/checkXfer/releaseReqH, loadLocalMD, connect/disconnect, unloadMD (no-op). Compare with Obj plugin.
- **mockkv_plugin.cpp**: Registers "MOCKKV" with NIXL.

**Suggested reading**: this doc → mockkv_backend.h → mockkv_backend.cpp (ctor → registerMem → postXfer → deregisterMem / unloadMD) → run with logs from section 5.
