<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL RESP KV plugins (`respkv`)

`respkv` is the source family for local NIXL storage backends whose command plane is RESP. Public backend names stay `REDIS` and `VALKEY`. Each product is its own plugin, with its own client, config, and transfer path. They share only the thin base `nixlRespKvEngine`.

The directory name means RESP-command KV backends. For `VALKEY`, RESP carries control (`BLOB.*`) and the payload moves by server-initiated RDMA.

Design: [RESP KV engine spec](../../../docs/superpowers/specs/2026-11-06-redis-valkey-kv-engine-design.md).

## When to use which plugin

| Backend | Directory | Transport | Client |
|---|---|---|---|
| `REDIS` | [`redis/`](redis/README.md) | TCP RESP `SET` / `GET` / `EXISTS` | hiredis + libevent (`iRedisClient`) |
| `VALKEY` | [`valkey/`](valkey/README.md) | Server-initiated RDMA via `BLOB.RDMA_SET` / `BLOB.RDMA_GET` | Valkey GLIDE FFI (`--features rdma`) |

Use `REDIS` when a Redis server and TCP byte copies are enough. Use `VALKEY` when objects move through the [valkey-large-object](https://github.com/valkey-io/valkey-large-object) module (AWS EFA, or libfabric `tcp` for development). Key rules, error codes, and `queryMem` stay per plugin; see the child READMEs.

Later RESP-family backends (for example Lavik) belong here as their own subdirectory and plugin name, subclassing `nixlRespKvEngine` when the NIXL surface matches. A backend outside this family gets its own plugin directory.

## Base: `nixlRespKvEngine`

`respkv_engine.h` and `respkv_engine.cpp` hold shared capabilities and no-ops:

- local only (`supportsLocal`; no remote peer, no notifications)
- `getSupportedMems()` returns `{OBJ_SEG, DRAM_SEG}`
- `connect`, `disconnect`, and `unloadMD` succeed
- `loadLocalMD` is an identity map; a null input is invalid
- `validatePrepCommon()` checks read/write, non-empty equal descriptor counts, and local `DRAM_SEG`, and warns when the remote agent differs. Remote segment type stays with the subclass (`REDIS` allows DRAM or OBJ; `VALKEY` requires OBJ).

`nixlRedisKVEngine` and `nixlValkeyEngine` subclass this base. The static library behind `respkv_engine_dep` has no hiredis or GLIDE dependency. Each child links its own client libraries.

## Layout

```text
src/plugins/respkv/
  respkv_engine.h/.cpp   nixlRespKvEngine
  meson.build            respkv_engine_dep; subdir('redis'); subdir('valkey')
  README.md              this file
  redis/                 plugin "REDIS" — redis/README.md
  valkey/                plugin "VALKEY" — valkey/README.md
```

## Build

Both names are in the default `all_plugins` list. `src/plugins/meson.build` enters this directory once; each child meson file decides whether that plugin is built.

- **REDIS** builds on a default configure when hiredis, libevent, and libevent_pthreads are present. Missing dependencies skip REDIS with a warning. `-Denable_plugins=REDIS`, or listing REDIS in `static_plugins`, is an error if those libraries are missing. Parameters, pool behavior, and tests: [`redis/README.md`](redis/README.md).
- **VALKEY** needs `-Dglide_path=<prefix>`: GLIDE `libglide_ffi.a` built with `--features rdma`, `glide_ffi.h`, and `connection_request.proto`, plus `protoc` and libprotobuf at build time. A default configure skips VALKEY with a warning if any of those are missing. An explicit `-Denable_plugins=VALKEY` (or static VALKEY) errors on the same missing dependency. GLIDE build steps stay in [`valkey/README.md`](valkey/README.md).

```bash
meson setup build                        # REDIS if hiredis and libevent are present
meson setup build -Dglide_path=<prefix>  # REDIS as above, plus VALKEY
```

nixlbench: `--backend=REDIS` or `--backend=VALKEY`. Container image: pass
`--enable-valkey` to `benchmark/nixlbench/contrib/build.sh` (GLIDE built in-image).
Smoke steps: [`valkey/README.md`](valkey/README.md), [`redis/README.md`](redis/README.md).
