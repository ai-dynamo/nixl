---
title: Redis
description: Key-value storage backend for transfers between DRAM and a Redis server.
---

## Overview

The Redis backend exposes a Redis server as local NIXL storage: `WRITE`
transfers store local DRAM bytes with `SET`, `READ` transfers copy values into
local DRAM with `GET`, and `queryMem` checks key existence with `EXISTS`.

| Property | Value |
|----------|-------|
| **Transfer Type** | DRAM ↔ Key-Value |
| **Protocol** | Redis (hiredis async) |
| **Best For** | Caching and KV offload alongside NIXL agents |

Local descriptors must be `DRAM_SEG`; remote descriptors may be `DRAM_SEG` or
`OBJ_SEG`. A Redis key comes from the descriptor `metaInfo` when present,
otherwise from the decimal `addr`. All keys are resolved before a transfer is
submitted, so an invalid descriptor cannot produce a partially submitted
transfer.

## Installation

The backend requires `hiredis` (with async API support), `libevent`, and
`libevent_pthreads`. In a default all-plugin build the backend is skipped with
a warning when these are missing; enabling it explicitly
(`-Denable_plugins=REDIS`) turns missing dependencies into a configure error.

```bash
meson setup build -Denable_plugins=REDIS
meson compile -C build
```

## Configuration

Each setting is resolved from the backend parameter map first, then the
corresponding `REDIS_*` environment variable, then the built-in default.

| Parameter | Environment fallback | Default | Description |
|-----------|----------------------|---------|-------------|
| `host` | `REDIS_HOST` | `localhost` | Redis hostname or IP address |
| `port` | `REDIS_PORT` | `6379` | Redis TCP port |
| `username` | `REDIS_USERNAME` | *(empty)* | Redis ACL username |
| `password` | `REDIS_PASSWORD` | *(empty)* | Redis AUTH password |
| `db` | `REDIS_DB` | `0` | Redis logical database |
| `pool_size` | `REDIS_POOL_SIZE` | `8` | Number of concurrent Redis connections |

Commands are dispatched to the healthy connection with the fewest in-flight
operations (least-connection routing). A single shared event-loop thread serves
the whole pool, so the backend costs one OS thread regardless of pool size.
Increase the pool only when the server runs with multiple I/O threads;
decrease it to `1` for memory-constrained environments.

## When to Use

- **KV offload** — stage tensors or KV cache pages in Redis with the same
  transfer API used for RDMA and storage backends.
- **Existence checks** — use `queryMem` to test whether a key exists before
  issuing a transfer.

See the [plugin README](https://github.com/ai-dynamo/nixl/blob/main/src/plugins/redis/README.md)
for the connection-pool architecture, authentication behavior, unit-test
instructions, and an nixlbench smoke test against a real server.
