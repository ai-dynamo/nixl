<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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

# NIXL RADOS Plugin

## Overview

The `RADOS` backend transfers data between local `DRAM_SEG` memory and
RADOS objects (`OBJ_SEG`) in a single Ceph pool through the librados
asynchronous I/O interface.

- **Local-only:** it serves `NIXL_WRITE` from local `DRAM_SEG` to
  `OBJ_SEG` descriptors of the same agent, and `NIXL_READ` in the
  reverse direction. Remote agents and notifications are not supported.
- **Direct data path:** reads target caller-provided destination memory,
  reconciling replacement output before completion; writes reference
  caller memory through a non-owning view (the Ceph client and network
  stack may still copy).
- **Object ranges:** an `OBJ_SEG` address is a byte offset inside the
  object. A write creates a missing object, overwrites the selected
  range, may extend the object, and preserves bytes outside the written
  range (native RADOS range-write semantics).
- Each descriptor pair is an independent RADOS operation; there is no
  atomicity across pairs within a request.

## Dependencies

- **librados** (pkg-config name `rados`): headers and shared library
  from a Ceph client distribution. Meson probes the required API surface
  during setup and builds the plugin only when the probe succeeds. The
  runtime librados must be ABI/link compatible with the build headers
  (no runtime version negotiation).
- **A reachable Ceph cluster** with a readable Ceph configuration file
  and a client entity authorized to read and write objects in the
  configured pool.

## Build and Test

`RADOS` is part of the default plugin set, built when its dependencies
are found:

```bash
meson setup build
ninja -C build
meson install -C build
```

- Explicitly enable with `-Denable_plugins=RADOS` (configuration fails
  if librados is not found); otherwise the plugin is silently skipped
  when librados is missing. Exclude it with `-Ddisable_plugins=RADOS` or
  a plugin list that omits it.
- Adding `RADOS` to `-Dstatic_plugins` builds a static library linked
  into the core NIXL library instead of a loadable plugin.
- **Tests:** deterministic unit tests run with no Ceph access. A
  separate, environment-gated live test performs a real-cluster round
  trip only when `NIXL_RADOS_CONF`, `NIXL_RADOS_POOL`, and
  `NIXL_RADOS_NAME` are exported (optionally `NIXL_RADOS_NS` to run
  under a namespace), and skips otherwise. These variables configure
  that live test only; runtime backend configuration comes solely from
  the custom parameters below.

Run the hermetic RADOS tests from a debug build with:

```bash
build/test/gtest/unit/unit --gtest_filter='rados*'
```

Run the environment-gated live test with:

```bash
meson test -C build rados_live --print-errorlogs
```

## Configuration

Custom parameters, passed as the `nixl_b_params_t` map to
`createBackend("RADOS", ...)`:

| Parameter | Description | Required |
|-----------|-------------|----------|
| `conf` | Path to a readable Ceph configuration file | Yes |
| `name` | Full Ceph entity name (normally `client.<id>`) | Yes |
| `pool` | RADOS pool opened at backend initialization | Yes |
| `namespace` | Object namespace for the pool context; absent keeps the default (unnamed) namespace; must be non-empty, NUL-free, and not the all-namespaces sentinel | No |
| `status_wait_us` | Bounded status wait for `getXferStatus()` polling: decimal integer in `[0, 1000000]` microseconds; absent or `0` keeps polling fully nonblocking; malformed or above-ceiling values fail backend creation | No |
| `ceph.<option>` | Any other parameter; the prefix is stripped and the option/value is applied verbatim to the Ceph configuration | No |

Backend creation is synchronous and may block on Ceph configuration,
authentication, connection, or pool-open work; it fails on a missing or
empty `conf`/`name`/`pool`, an invalid `namespace` or `status_wait_us`,
an unknown parameter key, or a connection/pool-open failure. Client
options such as `rados_mon_op_timeout` and `rados_osd_op_timeout` are
set through `ceph.<option>` overrides, which always take effect after
the configuration file is read and before connecting. The opened pool
is fixed for the backend's lifetime; several pools mean several backend
instances.

### `status_wait_us` summary

Absent or `0` keeps `getXferStatus()` strictly
nonblocking. Above `0`, a status poll of an in-progress request may
sleep once for the configured budget before reporting
`NIXL_IN_PROG`; completions that land during the sleep are harvested on
the next poll. The budget is an intended sleep, not a wall-clock
guarantee.

### `namespace` behavior

When set, every operation of the backend instance resolves within that
namespace; other namespaces require separate backend instances. External
listing or cleanup needs the namespace selected (`rados -N
<namespace>`), and isolation holds only where cephx caps enforce it.

## Supported Operations and Limitations

Registration:

- Only `DRAM_SEG` and `OBJ_SEG` can be registered with this backend.
- `DRAM_SEG` registration is metadata-free: it does not pin, own, or
  modify the caller's buffer.
- `OBJ_SEG` registration records only an in-memory `devId` -> object-key
  mapping (the non-empty `metaInfo` verbatim, else the decimal
  `devId`); it does not create, reserve, lock, or stat anything in Ceph.
  Duplicate `devId` registrations are rejected, while different `devId`s
  may map to the same object key. A `metaInfo` containing an embedded NUL
  byte is rejected.
- In transfer descriptors, an `OBJ_SEG` `addr` is the byte offset inside
  the object; registration ranges do not constrain transfer ranges.

Transfers:

- Only `NIXL_WRITE` (`DRAM_SEG` -> `OBJ_SEG`) and `NIXL_READ`
  (`OBJ_SEG` -> `DRAM_SEG`) against the local agent itself; no remote
  agents, no notifications.
- Each pair has a nonzero length of at most `INT_MAX` bytes.
- Pairs are posted as independent asynchronous range operations in
  descriptor order, best effort with no rollback; posting is not
  guaranteed nonblocking (librados may briefly wait on its in-flight
  admission budgets).
- Status polling reports in-progress while operations run; the terminal
  status is the first failure in pair order: `NIXL_ERR_NOT_FOUND` for a
  read of a missing object, `NIXL_ERR_MISMATCH` for a short read (the
  delivered prefix has already landed in the destination), and
  `NIXL_ERR_BACKEND` for other Ceph I/O failures.
- A request may be re-posted only from a terminal status.
- Release (`releaseXferReq`) is cleanup, not result reporting: it is
  synchronous and blocking, waits for every still-in-flight operation,
  cancels nothing, and imposes no deadlines of its own - bound
  in-flight time via Ceph client timeout options. Every created request
  must be released exactly once (including never-posted or
  submission-failed requests), and a released handle must never be used
  again.
- Agent destruction blocks until all registered requests drain; release
  all transfer requests before destroying the agent.
- Handles are for one caller at a time: do not overlap posting,
  polling, or releasing on the same handle.
- The backend coordinates nothing across requests (overlapping write
  ranges within one request are rejected); serialize concurrent use of
  overlapping buffers or object ranges across requests yourself.
- **No automatic object deletion:** deregistration, release, and
  teardown never delete or truncate objects. Pool cleanup is an
  out-of-band operator task.

## Buffer Lifetime (Critical)

The plugin borrows caller memory and does not own it:

- Keep write sources allocated, readable, and **immutable**, and read
  destinations allocated and writable, from posting until the transfer
  reaches a terminal status or the request is successfully released.
- Reusing, freeing, or mutating them earlier races the transfer and can
  corrupt in-flight data.
- Writes are the unforgiving side: the non-owning buffer view lets
  librados keep referencing the source across redirects and retries,
  with no copy fallback and no notification when it finally drops the
  reference.
- Terminal status and successful release are the only reuse points;
  memory deregistration alone does not end the obligation.

## `queryMem` Summary

`queryMem` supports `OBJ_SEG` only, is synchronous to the caller, and
stats each descriptor's object independently of registration:

- Found object: a decimal `size` parameter (including zero).
- Missing object: a null entry under `NIXL_SUCCESS`.
- Any other stat failure: null entry for that descriptor, remaining
  descriptors still queried, overall `NIXL_ERR_BACKEND`.
- A non-`OBJ_SEG` list yields all-null entries and
  `NIXL_ERR_NOT_SUPPORTED`; a `metaInfo` with an embedded NUL byte
  fails the whole list with `NIXL_ERR_INVALID_PARAM`.

Results are point-in-time: a query guarantees neither object
completeness nor reservation for a later read.

## Example Usage

A local write-then-read round trip against pool `nixl` as
`client.nixl`. Each request is posted, polled to a terminal status, and
released exactly once; every step's result is checked and the
round-tripped bytes are compared.

```cpp
#include <chrono>
#include <thread>

#include "nixl.h"

// Returns true only if the write/read round trip completed and the bytes match.
bool rados_round_trip() {
    nixl_b_params_t params = {
        {"conf", "/etc/ceph/ceph.conf"},
        {"name", "client.nixl"},
        {"pool", "nixl"},
    };

    const std::string agent_name = "agent1";
    nixlAgent agent(agent_name, nixlAgentConfig{});
    nixlBackendH *backend = nullptr;
    if (agent.createBackend("RADOS", params, backend) != NIXL_SUCCESS) {
        return false;
    }

    nixl_opt_args_t args;
    args.backends = {backend};

    std::vector<unsigned char> payload(64 * 1024, 0x5a);
    std::vector<unsigned char> readback(payload.size(), 0x00);

    nixl_reg_dlist_t dram_reg(DRAM_SEG);
    dram_reg.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(payload.data()), payload.size(), 0, {}));
    dram_reg.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(readback.data()), readback.size(), 0, {}));
    if (agent.registerMem(dram_reg, &args) != NIXL_SUCCESS) {
        return false;
    }

    nixl_reg_dlist_t obj_reg(OBJ_SEG);
    obj_reg.addDesc(nixlBlobDesc(/*addr (object offset)*/ 0, payload.size(),
                                 /*devId*/ 1, "model-checkpoint-0007"));
    if (agent.registerMem(obj_reg, &args) != NIXL_SUCCESS) {
        agent.deregisterMem(dram_reg, &args);
        return false;
    }

    // Post, poll to a terminal status, then release exactly once.
    auto run = [&](nixl_xfer_op_t op, const nixl_xfer_dlist_t &local,
                   const nixl_xfer_dlist_t &remote) {
        nixlXferReqH *req = nullptr;
        if (agent.createXferReq(op, local, remote, agent_name, req, &args) != NIXL_SUCCESS ||
            req == nullptr) {
            return false;
        }
        nixl_status_t status = agent.postXferReq(req);
        while (status == NIXL_IN_PROG) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            status = agent.getXferStatus(req);
        }
        const bool complete = (status == NIXL_SUCCESS);
        const bool released = (agent.releaseXferReq(req) == NIXL_SUCCESS);
        return complete && released;
    };

    nixl_xfer_dlist_t write_local(DRAM_SEG);
    write_local.addDesc(
        nixlBasicDesc(reinterpret_cast<uintptr_t>(payload.data()), payload.size(), 0));
    nixl_xfer_dlist_t read_local(DRAM_SEG);
    read_local.addDesc(
        nixlBasicDesc(reinterpret_cast<uintptr_t>(readback.data()), readback.size(), 0));
    nixl_xfer_dlist_t object_remote(OBJ_SEG);
    object_remote.addDesc(nixlBasicDesc(/*object offset*/ 0, payload.size(), /*devId*/ 1));

    bool ok = run(NIXL_WRITE, write_local, object_remote) &&
              run(NIXL_READ, read_local, object_remote);
    ok = ok && (payload == readback);

    if (agent.deregisterMem(obj_reg, &args) != NIXL_SUCCESS) ok = false;
    if (agent.deregisterMem(dram_reg, &args) != NIXL_SUCCESS) ok = false;
    return ok;
}
```
