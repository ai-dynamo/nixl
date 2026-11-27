---
title: Redis
description: Key-value storage backend for transfers between DRAM and a Redis server.
---

## Overview

The Redis backend exposes a Redis server as local NIXL storage. `WRITE` stores
local DRAM bytes with `SET`, `READ` copies a value into DRAM with `GET`, and
`queryMem` checks key existence with `EXISTS`. For a read, the stored value
must match the local descriptor's length.

| Property | Value |
|----------|-------|
| **Transfer Type** | DRAM ↔ Key-Value |
| **Protocol** | Redis (hiredis async) |
| **Best For** | Caching and KV offload alongside NIXL agents |

The local transfer descriptor supplies the DRAM buffer (`DRAM_SEG`). The remote
descriptor identifies the Redis key according to its registration type.

- `OBJ_SEG` requires the key in `metaInfo`. The Redis engine returns
  `NIXL_ERR_INVALID_PARAM` if `metaInfo` is empty. When Redis is the only
  backend tried, `nixlAgent::registerMem` reports this as `NIXL_ERR_BACKEND`.
- `DRAM_SEG` uses a nonempty `metaInfo` as the key. When it is empty, the key
  is the decimal string of the registered `addr`.

The backend resolves all remote keys before issuing the first `SET` or `GET`.

## Installation

The backend requires hiredis 1.0.0 or later with async API support, `libevent`,
and `libevent_pthreads`. In a default all-plugin build, Meson warns and skips
Redis if any dependency is missing. Selecting `REDIS` in `enable_plugins` or
`static_plugins` makes a missing dependency a configuration error.

```bash
meson setup build -Denable_plugins=REDIS
meson compile -C build
```

## Configuration

The backend looks up each setting in the parameter map, then in the matching
`REDIS_*` environment variable, then under the same key in the NIXL TOML
file. It uses the default only when all three omit the setting.

| Parameter | Environment / TOML key | Default | Description |
|-----------|----------------------|---------|-------------|
| `host` | `REDIS_HOST` | `localhost` | Redis hostname or IP address |
| `port` | `REDIS_PORT` | `6379` | TCP port, `1` through `65535` inclusive |
| `username` | `REDIS_USERNAME` | *(empty)* | Redis ACL username. A nonempty username requires a nonempty password. |
| `password` | `REDIS_PASSWORD` | *(empty)* | Redis AUTH password |
| `db` | `REDIS_DB` | `0` | Logical database index, `0` through `INT_MAX`. The server must accept `SELECT` for a nonzero index. |
| `pool_size` | `REDIS_POOL_SIZE` | `8` | Number of asynchronous transfer connections, `1` through `INT_MAX` |

The three numeric settings must fit a C++ `int`. In the parameter map and
environment, their strings must parse in full as decimal integers. The Redis
parser throws `std::invalid_argument` if a parsed value is outside the range
above or a username has no password. These errors and integer conversion
errors make `nixlAgent::createBackend` return `NIXL_ERR_BACKEND`.

With both credentials empty the client skips `AUTH`. A password alone selects
`AUTH password`. Supplying both credentials selects `AUTH username password`.

`GET` and `SET` use the healthy async connection with the fewest in-flight
operations. One event-loop thread serves all of those connections. The pool
also starts `pool_size` workers to copy `GET` replies into DRAM, for
`pool_size + 1` threads in total. `queryMem` uses a separate shared synchronous
connection. On a memory-constrained client, `pool_size=1` uses one async
connection and two pool threads, and copies one reply at a time.

## When to Use

- **KV offload** — stage tensors or KV cache pages in Redis with the same
  transfer API used for RDMA and storage backends.
- **Existence checks** — pass the Redis backend as the sole backend to
  `queryMem` to test whether a key exists before issuing a transfer.

For unit-test commands and a `nixlbench` smoke test against a Redis server, see
the [plugin README](https://github.com/ai-dynamo/nixl/blob/main/src/plugins/redis/README.md#build-and-test).
