<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL Redis Plugin

The `REDIS` plugin exposes Redis as a local NIXL storage backend. It supports asynchronous
`NIXL_WRITE`/`NIXL_READ` transfers and synchronous `queryMem()` checks.

## Architecture

The plugin follows the same direct-backend layout as `src/plugins/infinia`:

```text
src/plugins/redis/
  redis_backend.h/.cpp  nixlRedisKVEngine and NIXL transfer logic
  redis_client.h/.cpp   hiredis client and its unit-test interface
  redis_plugin.cpp      NIXL plugin registration
  meson.build           REDIS plugin target
  README.md
```

`nixlRedisKVEngine` derives directly from `nixlBackendEngine`. The internal `iRedisClient`
interface isolates hiredis/libevent and permits Redis-free unit tests; it is not a generic KV
extension API.

The production client uses:

- One hiredis/libevent async connection for `SET` and `GET`.
- One blocking hiredis connection for `EXISTS`.
- One libevent thread. Hiredis callbacks complete NIXL request promises directly.

## Dependencies

- `hiredis` with async API support
- `libevent`
- `libevent_pthreads`

If REDIS is explicitly enabled or selected as a static plugin, missing hiredis/libevent is a
configuration error. During a default all-plugin build, missing dependencies cause REDIS to be
skipped with a warning.

## Configuration

Backend parameters take precedence over environment variables.

| Parameter | Environment fallback | Default | Description |
|-----------|----------------------|---------|-------------|
| `host` | `REDIS_HOST` | `localhost` | Redis hostname or IP address |
| `port` | `REDIS_PORT` | `6379` | Redis TCP port |
| `password` | `REDIS_PASSWORD` | empty | Redis AUTH password |
| `db` | none | `0` | Redis logical database |

```cpp
nixl_b_params_t params = {
    {"host", "127.0.0.1"},
    {"port", "6379"},
    {"db", "2"},
};
agent.createBackend("REDIS", params);
```

## Transfer behavior

| NIXL operation | Redis operation | Result |
|----------------|-----------------|--------|
| `NIXL_WRITE` | `SET key bytes` | Stores local DRAM bytes |
| `NIXL_READ` | `GET key` | Copies the exact Redis value into local DRAM |
| `queryMem` | `EXISTS key` | Reports whether the key exists |

Local descriptors must be `DRAM_SEG`; remote descriptors may be `DRAM_SEG` or `OBJ_SEG`. A Redis
key comes from descriptor `metaInfo` when present, otherwise from the decimal `devId`. `postXfer()`
resolves every remote key before dispatching commands, so an invalid descriptor cannot produce a
partially submitted transfer.

`postXfer()` returns `NIXL_IN_PROG`; `checkXfer()` polls the request futures and returns success or
the first backend error once all completed work has been observed.

## Build and test

```bash
meson setup build -Denable_plugins=REDIS
meson compile -C build REDIS
meson test -C build unit --print-errorlogs
```

The dynamic plugin is produced at:

```text
build/src/plugins/redis/libplugin_REDIS.so
```

For a static build:

```bash
meson setup build -Denable_plugins=REDIS -Dstatic_plugins=REDIS
meson compile -C build
```

For a live smoke test, start Redis and run one write and one read:

```bash
export REDIS_HOST=127.0.0.1 REDIS_PORT=6379
nixlbench --backend REDIS --start_block_size 4096 --max_block_size 4096 \
  --num_iter 10 --warmup_iter 1 --op_type WRITE
nixlbench --backend REDIS --start_block_size 4096 --max_block_size 4096 \
  --num_iter 10 --warmup_iter 1 --op_type READ
```
