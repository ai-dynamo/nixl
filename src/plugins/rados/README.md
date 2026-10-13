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
asynchronous I/O interface. The backend is local-only: it serves `NIXL_WRITE`
from local `DRAM_SEG` to `OBJ_SEG` descriptors of the same agent and
`NIXL_READ` in the reverse direction. It does not support remote agents or
notifications. `queryMem` is not yet implemented and returns
`NIXL_ERR_NOT_SUPPORTED`; batched object-size queries land in a follow-up
commit of this series.

Transfers are direct: C AIO reads land directly in caller-provided destination
memory, and writes describe caller memory through a non-owning C++
`bufferlist::static_from_mem` view; the plugin never stages payload bytes in
its own buffers in either direction (the Ceph client and network stack may
still copy). Callers must retain readable, immutable write sources and
writable read destinations through a terminal status or a successful release —
that is, through completion-and-callback retirement (see Buffer lifetime);
reusing, freeing, or mutating those buffers early races the transfer and can
corrupt its data.

Each descriptor pair is an independent RADOS range operation; there is no
atomicity across pairs within a request. A write creates a missing object,
overwrites the selected range, and may extend the object; bytes outside the
written range are preserved (native RADOS range-write semantics).

## Dependencies

- **librados** (pkg-config name `rados`): headers and shared library from a
  Ceph client distribution. The plugin uses the librados C
  lifecycle/configuration/namespace/AIO-read surface (including pool-context
  namespace configuration) and the C++ `IoCtx` write surface
  (`IoCtx::aio_write` over a static, non-owning `bufferlist`). During setup, meson compiles and
  links a probe program using exactly this surface, and the plugin is
  built only when the probe succeeds; the runtime librados must be ABI/link
  compatible with the build headers (there is no runtime version
  negotiation).
- **A reachable Ceph cluster** with a readable Ceph configuration file and a
  client entity authorized to read and write objects in the configured pool.

## Build Instructions

`RADOS` is part of the default plugin set, which is built when its
dependencies are found. Standard build:

```bash
meson setup build
ninja -C build
meson install -C build
```

- **Explicitly enable:** `meson setup build -Denable_plugins=RADOS`.
  If librados is not found while the plugin is explicitly requested,
  configuration fails with an error.
- **Build without it:** `meson setup build -Ddisable_plugins=RADOS`, or
  simply request a plugin list that excludes it; when not explicitly requested
  and librados is missing, the plugin is silently skipped.
- **Dynamic (default):** a shared object `libplugin_RADOS.so` is
  installed under `<libdir>/plugins`. At runtime the plugin manager loads
  backends by name from `$NIXL_PLUGIN_DIR`, or from the `plugins/` directory
  next to the loading NIXL library. In debug builds the plugin path is also
  recorded in a `pluginlist` file in the build directory, which a debug-built
  NIXL library reads for discovery without installation.
- **Static:** adding `RADOS` to `-Dstatic_plugins` builds a static
  library linked into the core NIXL library instead of a loadable plugin; the
  same librados probe and configuration parameters apply.
- **Tests:** deterministic unit tests exercise the backend with no Ceph
  access. A separate, environment-gated live test performs a real-cluster
  round trip when `NIXL_RADOS_CONF`, `NIXL_RADOS_POOL`, and
  `NIXL_RADOS_NAME` are exported, and skips otherwise; when the optional
  `NIXL_RADOS_NS` is set the round trip runs under that `namespace` and the
  written object must stay invisible to a default-namespace backend on the
  same pool. These variables
  configure that live test only; they are not runtime plugin configuration —
  backend configuration comes solely from the custom parameters below.

## API Reference

### Custom parameters

Passed as the `nixl_b_params_t` map to `createBackend("RADOS", ...)`:

| Parameter | Description | Required |
|-----------|-------------|----------|
| `conf` | Path to a readable Ceph configuration file | Yes |
| `name` | Full Ceph entity name (normally `client.<id>`) | Yes |
| `pool` | RADOS pool opened at backend initialization | Yes |
| `namespace` | Optional object namespace for the opened pool context; absent keeps the default (unnamed) namespace, and a present value must be non-empty, NUL-free, and not the all-namespaces sentinel (see Object namespace) | No |
| `status_wait_us` | Optional bounded status wait for `checkXfer` polling: strict ASCII decimal (`[0-9]+`, leading zeros valid) in `[0, 1000000]` microseconds; absent or `0` keeps polling fully nonblocking; malformed or above-ceiling values fail backend creation (see Bounded status wait) | No |
| `ceph.<option>` | Any other parameter; the prefix is stripped and the option/value are applied verbatim to the Ceph configuration (empty values allowed) | No |

Backend creation is synchronous: at initialization the backend creates a
cluster handle with `name`, reads `conf`, applies the `ceph.<option>`
overrides, connects, and opens the pool — any of which may block on Ceph
configuration, authentication, connection, or pool-open work — then points
the opened ioctx at the optional `namespace` (local handle configuration with
no Ceph round trip) and binds a non-owning C++ `IoCtx` view of the opened raw
ioctx for writes. Overrides
therefore always take effect after the configuration file is read and before
the connection is made. The `ceph.rados_mon_op_timeout` and
`ceph.rados_osd_op_timeout` client timeouts (whole seconds; both default to
`0`, unlimited) are forwarded before connect. Authentication and keyring
resolution are handled by Ceph through `conf` and the overrides; the plugin
interprets nothing beyond the three required keys, the optional `namespace`,
and the numeric `status_wait_us` budget. A missing or empty `conf`, `name`,
or `pool`, an invalid `namespace`, an invalid `status_wait_us` value, a
parameter that is neither one of those keys
nor `ceph.<option>` (including the bare key `ceph.`), or a failure to connect
or open the pool makes `createBackend()` return an error without a usable
backend. The opened pool is fixed for the backend instance's lifetime;
working with several pools means creating separate backend instances (each
with its own `pool` parameter), as the public agent API permits.

### Bounded status wait (`status_wait_us`)

An experimental polling knob, reserved ahead of the unknown-key rule;
`ceph.status_wait_us` remains a normal Ceph option override.

- Absent or `0` (leading zeros valid, e.g. `"00000"`): `checkXfer` stays
  strictly nonblocking and the engine allocates and retains no timer, so the
  sleep is unreachable.
- Above `0` (ceiling 1,000,000 us): when a `checkXfer` poll of an ACTIVE
  request reaches its first unproven pair, the call releases the request-state
  lock, sleeps exactly once for the configured budget (a plain timeout, no
  early wake-up, no lock held), and returns `NIXL_IN_PROG` without rescanning;
  pairs that would complete during the sleep — and pairs after that first
  unproven one — are harvested on the caller's next poll. The C/C++ librados
  surface reports only combined complete-and-callback status, so an
  operationally complete but callback-unretired completion counts as unproven
  and may trigger the sleep.
- The budget is an intended sleep, not a wall-clock bound: scheduler delivery
  delay lies outside it. Release and teardown drain through the completion
  waits and never through this timer; overlapping a sleep with another
  thread's release or teardown stays outside the one-caller-per-handle rule.

### Object namespace (`namespace`)

Reserved ahead of the unknown-key rule, so `ceph.namespace` stays a normal
Ceph option. Absent keeps the default (unnamed) namespace and the setter is
never called; a present value must be non-empty, NUL-free, and not the
all-namespaces sentinel (single `0x01` byte), or creation fails before any
Ceph access. When set, the setter runs exactly once on the raw ioctx after
pool open and before the C++ view is bound, so every operation of the
instance resolves within that namespace; other namespaces need separate
backend instances. External listing or cleanup needs the namespace selected
(`rados -N <namespace>`), and isolation holds only where cephx caps enforce
it.

### Registration and object-key mapping

- Only `DRAM_SEG` and `OBJ_SEG` can be registered with this backend.
- `DRAM_SEG` registration is metadata-free: it does not pin, own, modify, or
  write the caller's buffer. Reads only touch destination memory once a
  transfer is posted (see Buffer lifetime).
- Each `OBJ_SEG` registration records only an in-memory `devId` → object-key
  mapping inside the backend: it does not create, reserve, lock, or stat any
  object in Ceph. The object key is the non-empty `metaInfo` verbatim, or the
  decimal `devId` when `metaInfo` is empty. A `devId` that is already
  registered is rejected; different `devId` values may deliberately resolve to
   the same object key. After the type and initialization checks, a `metaInfo`
   containing an embedded NUL byte is rejected with `NIXL_ERR_INVALID_PARAM`
   before registration reserves anything, because librados object operations
   take the key as a C string, so a NUL would silently truncate the effective
   key.
- In transfer descriptors, an `OBJ_SEG` `addr` is the byte offset inside the
  object. Transfer requests are matched against registrations by the agent:
  every local range must lie within a registered `DRAM_SEG` descriptor, and
  every remote range within one registered `OBJ_SEG` descriptor, so a
  registration's `addr`/`len` bound the object range usable for transfers
  (a zero-length `OBJ_SEG` registration denotes an unbounded range from its
  `addr` onward, per the NIXL object-segment convention). The backend itself
  resolves only the `devId` → key mapping.
- Deregistration performs no RADOS I/O: it removes the in-memory mapping only
  (it neither deletes nor truncates any object); it does not wait for
  in-flight requests, does not stop them, and does not end their buffer
  obligations (see Buffer lifetime).

### Transfer operations

- Request creation validates the pair list: only `NIXL_WRITE`
  (`DRAM_SEG` → `OBJ_SEG`) and `NIXL_READ` (`OBJ_SEG` → `DRAM_SEG`) against
  the local agent itself; equal, non-empty descriptor lists with non-zero
  per-pair lengths of at most `INT_MAX` bytes; non-null local DRAM addresses;
  local and object-offset ranges that do not wrap their address space;
  registered object `devId`s; and no overlapping write ranges to one object
  within a request. Anything else is rejected. On success, key, pointer,
  offset, length, and direction are snapshotted into the request.
- Posting submits each pair as an independent asynchronous RADOS range
  operation in descriptor order. Operations complete asynchronously, but
  posting is not guaranteed non-blocking: librados may wait on its Objecter
  client-side in-flight operation and byte admission budgets before accepting
  more work. Submission is best effort in descriptor order with no rollback:
  if a pair fails to submit, later pairs in that round are not submitted,
  already-accepted operations keep running and may still change objects or
  destination memory. If no pair was accepted the request terminates
  immediately with the mapped error, otherwise it runs in progress until
  polled.
- A posted request can be re-posted only from a terminal status: the previous
  round's results are fully reset and the same prepared ranges are submitted
  again from the current contents of the caller's buffers.
- Status polling never blocks by default: while submitted operations are still
  running the request reports in-progress; once all have completed, per-pair
  results are collected and the overall request status is the first non-success
  in descriptor-pair order. With `status_wait_us` above 0, a poll that reaches
  an unproven pair may first take a single budgeted timeout sleep outside the
  request-state lock before reporting in-progress (see Bounded status wait).
- Releasing a request (`releaseXferReq`) is cleanup, not result reporting:
  the I/O outcome is whatever the polled terminal status was, and a successful
  release only means the request's resources and references to caller memory
  and in-flight operations are gone — never that the transfer succeeded. The
  call is synchronous and blocks: nothing is cancelled and the plugin imposes
  no deadlines of its own, so it waits for every still-in-flight operation to
  finish and can take as long as the slowest one (bounding in-flight time
  relies on Ceph client timeout/configuration options applied via
  `ceph.<option>`). Every successfully created request must be released
  exactly once — including requests that were never posted or whose posting
  failed outright — and a released handle is finished and must never be used
  again. Destroying the agent blocks the same way: backend teardown drains
  every still-registered request before closing the pool context and the
  cluster connection, so release every transfer request before destroying the
  agent.
- Request handles are for one caller at a time: posting, polling, and
  releasing must not overlap on the same handle. The backend coordinates
  nothing across requests (the overlap rejection above applies within one
  request only), so serialize concurrent use of overlapping read destinations
  and of the same or overlapping object ranges across requests yourself.

### Results and errors

When a request reaches a terminal status, the public request status reports
one ordered aggregate result derived from the completed operations' per-pair
outcomes:

| Condition | Status |
|-----------|--------|
| Read of a missing object | `NIXL_ERR_NOT_FOUND` |
| Short read (delivered fewer bytes than requested; the delivered prefix has already landed in the destination) | `NIXL_ERR_MISMATCH` |
| Other I/O failure reported by Ceph | `NIXL_ERR_BACKEND` (logged) |

`queryMem` is not implemented at this increment and returns
`NIXL_ERR_NOT_SUPPORTED`.

### Buffer lifetime

The plugin's buffer-reuse boundary is completion-and-callback retirement of
each operation's librados completion (`is_complete_and_cb()` /
`wait_for_complete_and_cb()`), not operation completion alone. Callers observe
that boundary at exactly two plugin points: terminal status reporting (polling
retires every operation before a terminal status is reported) and a successful
synchronous release (which blocks until every still-in-flight operation has
retired). The boundary relies on librados' completion guarantee that a
retired completion no longer references caller memory; the plugin does not
independently observe librados' internal last-reference destruction moment.

- **Reads (C AIO):** librados writes reply bytes directly into the supplied
  destination; if a redirect or retry replaces that buffer, current Ceph's
  retry/transport path falls back to copying the data. Keep the destination
  allocated and writable from posting until the boundary above.
- **Writes (static bufferlist):** the non-owning `static_from_mem` view
  borrows caller memory, and librados keeps using the source across redirects
  and retries with no analogous fallback copy and no public notification when
  it finally drops its reference. Keep each source allocated, readable, and
  immutable until the same boundary; mutating it earlier races the transfer.
- A successful release of in-flight work ends both obligations; memory
  deregistration alone does not.

### Key classes

- `nixlRadosEngine` (`rados_backend.h/.cpp`): the backend engine
  implementing registration, transfers, status polling, and draining release.
- `nixl::rados::statusWaitTimer` (`rados_backend.h/.cpp`): the
  engine-owned timeout-only sleep seam, present exactly when
  `status_wait_us > 0`.
- `nixl::rados::parseCustomParams()` (`rados_config.h`):
  parameter parsing and validation for the table above.

## Example Usage

A local write-then-read round trip against a pool named `nixl` as
`client.nixl`, with one Ceph option override relocating the client log file.
Every step's result is checked, each request is driven to a terminal status
before its single release, the round-tripped bytes are compared, and the
deregistrations are verified:

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
        {"ceph.log_file", "/var/log/ceph/nixl-rados.log"},
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
