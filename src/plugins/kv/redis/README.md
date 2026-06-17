<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL Redis Plugin (KV)

The REDIS implementation lives in `src/plugins/kv/redis/`. Shared KV abstractions and the plugin
entrypoint stay in `src/plugins/kv/`.

```
kv/                           # shared KV layer (PR1 interface)
  kv_backend.h/.cpp           nixlKVEngine, nixlKVEngineImpl, iKVStore
  kv_plugin.cpp               registers "REDIS"
  meson.build                 delegates to redis/ when REDIS enabled

kv/redis/                     # REDIS plugin (this directory)
  redis_engine.h/.cpp         iRedisClient + nixlRedisKVEngine
  redis_executor.h            Redis-specific ASIO thread pool
  client.h/.cpp               hiredisAsyncClient
  engine_impl.h/.cpp          nixlRedisKVEngineImpl
  meson.build
  README.md
```

## Class hierarchy

| Layer | REDIS plugin |
|-------|--------------|
| Shared KV wrapper | `nixlKVEngine` (`../kv_backend.h`) |
| Shared KV impl base | `nixlKVEngineImpl` (`../kv_backend.h`) |
| Plugin entry | `nixlRedisKVEngine` (`redis_engine.h`) |
| Async storage interface | `iRedisClient` (`redis_engine.h`) |
| Client implementation | `hiredisAsyncClient` (`client.h`) |
| Engine implementation | `nixlRedisKVEngineImpl` (`engine_impl.h`) |

## iKVStore vs iRedisClient

| | `iKVStore` (`../kv_backend.h`) | `iRedisClient` (`redis_engine.h`) |
|--|---------------------------|-----------------------------------|
| Used by | INMEMKV example plugin | REDIS plugin |
| API style | Sync `put` / `get` / `exists` | Async SET/GET + sync EXISTS (separate connections) |
| `postXfer` | Returns `NIXL_SUCCESS` immediately | Returns `NIXL_IN_PROG`; `checkXfer` polls futures |
| Why separate | Blocking on hiredis event loop would deadlock | Async promise path for postXfer/checkXfer |

## iRedisClient API

```cpp
class iRedisClient {
    void setExecutor(std::shared_ptr<redisThreadPoolExecutor> executor);

    // postXfer path (async connection)
    void putKeyAsync(..., std::shared_ptr<std::promise<nixl_status_t>> promise);
    void getKeyAsync(..., std::shared_ptr<std::promise<nixl_status_t>> promise);

    // queryMem path (sync connection)
    std::optional<bool> checkKeyExistsSync(std::string_view key);
};
```

Connection settings (constructor): `host`, `port`, `password`, `db` via NIXL custom params or env vars `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`.

## Transfer call path

```
nixlbench / NIXL API
  -> nixlRedisKVEngine::postXfer
  -> nixlRedisKVEngineImpl::postXfer
       creates std::promise per descriptor
       -> iRedisClient::putKeyAsync / getKeyAsync (async connection)
       -> hiredisAsyncClient (libevent thread + redisAsyncCommand)
       promise completed on ASIO executor thread
  -> checkXfer polls statusFutures_

queryMem (sync path):
  -> nixlRedisKVEngineImpl::queryMem
       -> iRedisClient::checkKeyExistsSync (separate blocking hiredis connection)
       -> redisCommand EXISTS, returns before queryMem exits
```

## Build

### Native (plugin only)

```bash
meson setup build -Denable_plugins=REDIS
ninja -C build
# -> build/src/plugins/kv/redis/libplugin_REDIS.so
```

### Docker image (nixlbench + REDIS plugin)

On the **host**, from `benchmark/nixlbench/contrib/`:

```bash
cd benchmark/nixlbench/contrib

# Build image; tag defaults to nixlbench:v<version>.dev.<git-sha>
./build.sh --nixl <path-to-nixl> --redis-only

# Or pin an explicit tag:
./build.sh --nixl <path-to-nixl> --redis-only --tag nixlbench:v0.1.0.dev.9da7bd1
```

`--redis-only` passes `-Denable_plugins=REDIS -Dstatic_plugins=REDIS` into the image build.
The image installs `nixlbench` to `/usr/local/nixlbench/bin` and sets the default
working directory to `/workspace/nixl/benchmark/kvbench`.

## Test (Docker container, step by step)

This matches the workflow: build image on the host, then run `nixlbench` **inside** the container.

### Step 1 — Start Redis on the host

```bash
docker run -d --name nixlbench-redis -p 6379:6379 redis:7-alpine
```

Redis listens on host port `6379`. Use `--network host` when starting the bench container
(step 2) so `127.0.0.1:6379` inside the container reaches this Redis instance.

### Step 2 — Start the nixlbench container

```bash
docker run -it --rm --network host \
  nixlbench:v0.1.0.dev.9da7bd1 \
  bash
```

Replace the image tag with the one printed by `build.sh`.

### Step 3 — Inside the container: set Redis env and run bench

The image default working directory is `/workspace/nixl/benchmark/kvbench`.
`nixlbench` is already on `PATH`.

```bash
cd /workspace/nixl/benchmark/kvbench
export REDIS_HOST=127.0.0.1 REDIS_PORT=6379

# READ — nixlbench seeds the Redis key automatically before the transfer loop
nixlbench --backend REDIS --start_block_size 4096 --max_block_size 4096 \
  --num_iter 50 --warmup_iter 5 --op_type READ

# WRITE
nixlbench --backend REDIS --start_block_size 4096 --max_block_size 4096 \
  --num_iter 100 --warmup_iter 10 --op_type WRITE
```

Expected log lines:

- `REDIS backend configured (using defaults or environment variables)`
- READ only: `Seeded Redis key for READ: nixlbench_redis0_0_...`
- Stats table with bandwidth and latency (example: ~0.03 GB/s, ~120 µs at 4096 B)

**Notes:**

- `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` are read by `hiredisAsyncClient` at plugin init.
- nixlbench may adjust `--num_iter` / `--warmup_iter` for thread alignment (warnings are normal).
- For a one-shot run without an interactive shell, pass `nixlbench ...` as the `docker run` command
  instead of `bash`.

### Step 4 — Native bench (no Docker)

If `nixlbench` is built locally with REDIS support:

```bash
export REDIS_HOST=127.0.0.1 REDIS_PORT=6379

nixlbench --backend REDIS --start_block_size 4096 --max_block_size 4096 \
  --num_iter 100 --warmup_iter 10 --op_type WRITE

nixlbench --backend REDIS --start_block_size 4096 --max_block_size 4096 \
  --num_iter 50 --warmup_iter 5 --op_type READ
```
