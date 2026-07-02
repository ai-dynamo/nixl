<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NIXL Key-Value Storage Plugin

The KV plugin layer provides a NIXL backend facade for key-value storage. The shared code in this
directory follows the same facade/implementation split as the object backend, while concrete
implementations live in backend-specific subdirectories.

The current production implementation is the `REDIS` plugin under `redis/`.

## Architecture

The shared KV layer owns the NIXL-facing backend wrapper and implementation interface:

- **KV backend wrapper** (`nixlKVEngine`): The class registered with the NIXL backend plugin loader.
- **KV implementation base** (`nixlKVEngineImpl`): The backend-facing interface implemented by concrete KV stores.
- **Simple KV store interface** (`iKVStore`): A synchronous put/get/exists helper interface for simple KV implementations.

The Redis implementation uses a richer client interface because `postXfer` is asynchronous:

- **Redis engine** (`nixlRedisKVEngine`): The REDIS plugin engine registered by `kv_plugin.cpp`.
- **Redis implementation** (`nixlRedisKVEngineImpl`): Maps NIXL transfers to Redis SET/GET/EXISTS operations.
- **Redis client** (`iRedisClient`, `hiredisAsyncClient`): Uses an async hiredis connection for SET/GET and a
  separate synchronous connection for EXISTS.
- **Redis executor** (`redisThreadPoolExecutor`): Completes async promises away from the hiredis/libevent callback path.

```
kv/
  kv_backend.h/.cpp           shared nixlKVEngine, nixlKVEngineImpl, iKVStore
  kv_plugin.cpp               registers the REDIS backend with the NIXL plugin loader
  meson.build                 includes backend subdirectories

kv/redis/
  redis_engine.h/.cpp         iRedisClient and nixlRedisKVEngine
  engine_impl.h/.cpp          nixlRedisKVEngineImpl transfer logic
  client.h/.cpp               hiredisAsyncClient
  redis_executor.h            Redis-specific ASIO executor
  meson.build                 REDIS plugin target
  README.md                   Redis-specific build and benchmark notes
```

## Dependencies

The shared KV facade has no storage-system dependency. The REDIS backend requires:

| Dependency | Used for |
|------------|----------|
| `hiredis` / `hiredis_async` | Redis command encoding, sync EXISTS, async SET/GET |
| `libevent` | Event loop integration for hiredis async |
| `libevent_pthreads` | Thread-safe libevent setup |
| `asio` | Thread pool used by the Redis executor |

If REDIS is explicitly enabled or requested as a static plugin, missing hiredis/libevent is treated
as a configuration error. When building all plugins by default, a missing Redis dependency skips the
dynamic REDIS plugin with a warning.

## Configuration

Backend parameters are passed as a `nixl_b_params_t` map when creating the backend. Backend
parameters take precedence over environment variables.

| Parameter | Environment fallback | Default | Description |
|-----------|----------------------|---------|-------------|
| `host` | `REDIS_HOST` | `localhost` | Redis server hostname or IP address |
| `port` | `REDIS_PORT` | `6379` | Redis server TCP port |
| `password` | `REDIS_PASSWORD` | empty | Redis AUTH password |
| `db` | - | `0` | Redis logical database number |
| `num_threads` | - | half of hardware concurrency, minimum 1 | ASIO worker threads for async completion |

### Configuration Examples

#### Local Redis

```cpp
nixl_b_params_t params = {
    {"host", "127.0.0.1"},
    {"port", "6379"}
};
agent.createBackend("REDIS", params);
```

#### Authenticated Redis

```cpp
nixl_b_params_t params = {
    {"host", "redis.internal"},
    {"port", "6379"},
    {"password", "example-password"},
    {"db", "2"},
    {"num_threads", "4"}
};
agent.createBackend("REDIS", params);
```

#### Environment Variable Configuration

```bash
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export REDIS_PASSWORD=example-password
```

```cpp
nixl_b_params_t params = {};
agent.createBackend("REDIS", params);
```

## Transfer Operations

The REDIS backend supports `NIXL_WRITE` and `NIXL_READ`.

| NIXL operation | Redis operation | Behavior |
|----------------|-----------------|----------|
| `NIXL_WRITE` | `SET key bytes` | Stores local DRAM bytes under the remote Redis key |
| `NIXL_READ` | `GET key` | Reads the Redis value into the local DRAM descriptor |
| `queryMem` | `EXISTS key` | Reports whether a Redis key exists |

Local descriptors must be `DRAM_SEG`. Remote descriptors may be `DRAM_SEG` or `OBJ_SEG`, matching
the segment list exposed by the plugin. The Redis key is chosen from descriptor metadata:

1. Use `metaInfo` when it is present.
2. Otherwise use the descriptor `devId` converted to a string.

`prepXfer` validates operation type, local/remote descriptor counts, and supported segment types.
`postXfer` creates one promise/future pair per descriptor and returns `NIXL_IN_PROG`. `checkXfer`
polls those futures until all Redis async operations complete.

## Build

Build only the REDIS plugin:

```bash
meson setup build -Denable_plugins=REDIS
ninja -C build src/plugins/kv/redis/libplugin_REDIS.so
```

For static plugin builds:

```bash
meson setup build -Denable_plugins=REDIS -Dstatic_plugins=REDIS
ninja -C build
```

The dynamic plugin output is:

```text
build/src/plugins/kv/redis/libplugin_REDIS.so
```

## Extending KV Backends

New KV implementations should follow the same split used by Redis:

1. Add a backend-specific subdirectory under `src/plugins/kv/`.
2. Implement `nixlKVEngineImpl` for NIXL registration, query, prep, post, check, and request-handle cleanup.
3. Provide a small client/store interface for backend-specific storage operations.
4. Add a Meson subdirectory and plugin source list.
5. Register the plugin from a top-level `kv_plugin.cpp`-style entrypoint when the implementation should be exposed as a NIXL plugin.

There is no generic `kv_executor.h` today. Redis keeps `redis_executor.h` because the executor is
only used by the Redis async client, and OBJ's executor inherits from an AWS SDK executor interface.
If another KV implementation needs the same ASIO-only executor, it would be reasonable to promote
that type into a shared `kv_executor.h` at that point.
