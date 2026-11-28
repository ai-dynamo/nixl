<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL RESP KV plugins (`respkv`)

`respkv` is the source family for local NIXL storage backends whose command plane is RESP. Each product is its own plugin, with its own client, config, and transfer path. They share only the thin base `nixlRespKvEngine`.

## Plugins

| Backend | Directory | Transport | Client |
|---|---|---|---|
| `REDIS` | [`redis/`](redis/README.md) | TCP RESP `SET` / `GET` / `EXISTS` | hiredis + libevent (`iRedisClient`) |

Key rules, error codes, and `queryMem` stay per plugin; see the child README.

A later RESP-family backend belongs here as its own subdirectory and plugin name, subclassing `nixlRespKvEngine` when the NIXL surface matches. A backend outside this family gets its own plugin directory.

## Base: `nixlRespKvEngine`

`respkv_engine.h` and `respkv_engine.cpp` hold shared capabilities and no-ops:

- local only (constructed with `nixl_backend_capability_t::LOCAL`; no remote peer, no notifications)
- `getSupportedMems()` returns `{OBJ_SEG, DRAM_SEG}`
- `connect`, `disconnect`, and `unloadMD` succeed
- `loadLocalMD` is an identity map; a null input is invalid
- `validatePrepCommon()` checks read/write, non-empty equal descriptor counts, and local `DRAM_SEG`, and warns when the remote agent differs. Remote segment type stays with the subclass.

`nixlRedisKVEngine` subclasses this base. The static library behind `respkv_engine_dep` has no hiredis dependency.

## Layout

```text
src/plugins/respkv/
  respkv_engine.h/.cpp   nixlRespKvEngine
  meson.build            respkv_engine_dep; subdir('redis')
  README.md              this file
  redis/                 plugin "REDIS" — redis/README.md
```

## Build

`src/plugins/meson.build` enters this directory once; each child meson file decides whether that plugin is built.

- **REDIS** builds on a default configure when hiredis, libevent, and libevent_pthreads are present. Missing dependencies skip REDIS with a warning. `-Denable_plugins=REDIS`, or listing REDIS in `static_plugins`, is an error if those libraries are missing. Parameters, pool behavior, and tests: [`redis/README.md`](redis/README.md).

```bash
meson setup build
```
