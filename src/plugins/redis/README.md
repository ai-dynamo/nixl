<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL Redis Plugin

The `REDIS` plugin exposes Redis as a local NIXL storage backend. It supports

- asynchronous `NIXL_WRITE`/`NIXL_READ` transfers and
- synchronous `queryMem()` checks.

## Architecture

The plugin follows the direct-backend layout similar to existing backends:

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

`RedisConfig` is the Redis client's resolved, internal configuration value. The backend calls
`RedisConfig::fromBackendParams()` once during construction and passes the result to
`hiredisAsyncClient`; the client does not repeatedly read backend parameters or environment
variables.

```cpp
struct RedisConfig {
    std::string host = "localhost";
    int port = 6379;
    std::string username;
    std::string password;
    int db = 0;
};
```

Each setting is resolved in this precedence order:

1. A valid value in the NIXL backend parameter map.
2. The corresponding `REDIS_*` environment variable, when one exists.
3. The built-in default.

An explicitly provided string value, including an empty username or password, takes precedence
over its environment fallback. Port values that fail validation fall through to `REDIS_PORT` and
then to `6379`; database parse failures fall back to `0`. The logical database currently has no
environment fallback.

| Parameter | Environment fallback | Default | Description |
|-----------|----------------------|---------|-------------|
| `host` | `REDIS_HOST` | `localhost` | Redis hostname or IP address |
| `port` | `REDIS_PORT` | `6379` | Redis TCP port |
| `username` | `REDIS_USERNAME` | empty | Redis ACL username |
| `password` | `REDIS_PASSWORD` | empty | Redis AUTH password |
| `db` | none | `0` | Redis logical database |

Authentication behavior is determined by the resolved credentials:

- When `password` is empty, the client does not send `AUTH`. This is correct for a Redis server
  that allows unauthenticated access. A password-protected server will reject subsequent commands.
- When only `password` is set, the client sends legacy/default-user `AUTH password`.
- When both `username` and `password` are set, the client sends ACL-style
  `AUTH username password`. A username without a password is rejected during backend creation.

```cpp
nixl_b_params_t params = {
    {"host", "127.0.0.1"},
    {"port", "6379"},
    {"username", "nixl"},
    {"password", "example-password"},
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

Build the plugin:

```bash
meson setup build -Denable_plugins=REDIS
meson compile -C build REDIS
```

NIXL does not add tests to a `release` build, even when `build_tests` is enabled. To build and run
the Redis unit tests, install GoogleTest and GoogleMock and configure a separate non-release build:

```bash
# Debian/Ubuntu
sudo apt-get install libgtest-dev libgmock-dev

meson setup build-redis-tests \
  --buildtype=debug \
  -Dbuild_tests=true \
  -Denable_plugins=REDIS
meson compile -C build-redis-tests unit
meson devenv -C build-redis-tests \
  ./test/gtest/unit/unit \
  '--gtest_filter=redisConfigTest.*:redisEngineTest.*'
```

The filter runs only the Redis configuration and backend suites. A successful run currently
reports 11 tests from 2 test suites. These tests inject `mockRedisClient` and do not require a
running Redis server; use the live smoke test below to exercise hiredis and a real server.

The dynamic plugin is produced at:

```text
build/src/plugins/redis/libplugin_REDIS.so
```

For a static build:

```bash
meson setup build -Denable_plugins=REDIS -Dstatic_plugins=REDIS
meson compile -C build
```

## Live nixlbench smoke test

The unit tests use an injected Redis client and do not connect to a server. Use this smoke test to
exercise plugin creation, hiredis/libevent, and an actual Redis write/read data path. The example
uses a static REDIS plugin so the standalone nixlbench binary does not depend on dynamic plugin
discovery.

Start an isolated Redis server on the loopback interface and verify that it is ready:

```bash
docker run --detach --rm --name nixl-redis-smoke \
  --publish 127.0.0.1:6379:6379 redis:7-alpine
docker exec nixl-redis-smoke redis-cli PING
```

The readiness command must print `PONG`.

Install a static REDIS-enabled NIXL build into a temporary prefix, then build nixlbench against
that installation:

```bash
export NIXL_PREFIX=/tmp/nixl-redis-smoke-install

meson setup build-redis \
  --prefix="$NIXL_PREFIX" \
  -Denable_plugins=REDIS \
  -Dstatic_plugins=REDIS
meson compile -C build-redis
meson install -C build-redis

meson setup build-nixlbench benchmark/nixlbench \
  -Dnixl_path="$NIXL_PREFIX"
meson compile -C build-nixlbench
build-nixlbench/nixlbench --help | grep REDIS
```

The NIXL build requires the plugin dependencies listed above. The standalone nixlbench build also
requires the hiredis development package because it seeds Redis before a READ and independently
checks transferred data when consistency checking is enabled.

Run fixed-size 4 KiB WRITE and READ benchmarks. These commands use one thread, one descriptor per
batch, one in-flight request, 32 warm-up iterations, and 208 measured iterations:

```bash
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export LD_LIBRARY_PATH="$NIXL_PREFIX/lib/$(uname -m)-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

build-nixlbench/nixlbench \
  --backend=REDIS \
  --runtime_type=ASIO \
  --op_type=WRITE \
  --start_block_size=4096 \
  --max_block_size=4096 \
  --start_batch_size=1 \
  --max_batch_size=1 \
  --pipeline_depth=1 \
  --num_threads=1 \
  --total_buffer_size=67108864 \
  --warmup_iter=32 \
  --num_iter=208 \
  --check_consistency=true

build-nixlbench/nixlbench \
  --backend=REDIS \
  --runtime_type=ASIO \
  --op_type=READ \
  --start_block_size=4096 \
  --max_block_size=4096 \
  --start_batch_size=1 \
  --max_batch_size=1 \
  --pipeline_depth=1 \
  --num_threads=1 \
  --total_buffer_size=67108864 \
  --warmup_iter=32 \
  --num_iter=208 \
  --check_consistency=true
```

Each command must exit successfully and print a result row for block size `4096`. The READ run
must also print `Seeded Redis key for READ`; with `--check_consistency=true`, nixlbench reports an
error and exits unsuccessfully if the transferred data differs. The reported bandwidth and latency
are loopback smoke-test measurements, not production performance results.

The example uses Redis database 0 without authentication. `REDIS_PASSWORD` may be set for a
password-protected default user. nixlbench's direct seed/consistency helper does not currently
support `REDIS_USERNAME` or selecting a nonzero `db`, even though the plugin itself supports both.

Inspect the created keys if desired, then stop the server:

```bash
docker exec nixl-redis-smoke redis-cli DBSIZE
docker stop nixl-redis-smoke
```
