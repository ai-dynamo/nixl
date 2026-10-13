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

The `RADOS` backend targets local `DRAM_SEG` memory and RADOS objects
(`OBJ_SEG`) in a single Ceph pool. The backend is local-only: it declares
`DRAM_SEG` and `OBJ_SEG`, serves only its own agent, and does not support
remote agents or notifications.

This first increment implements the plugin surface up to memory
registration: backend bootstrap from custom parameters, `DRAM_SEG` and
`OBJ_SEG` registration/deregistration, and capability declaration. The
transfer request surface (`prepXfer`, `postXfer`, `checkXfer`,
`releaseReqH`) and `queryMem` return `NIXL_ERR_NOT_SUPPORTED` until the
asynchronous transfer and query commits land on top of this series.

## Dependencies

- **librados** (pkg-config name `rados`): headers and shared library from a
  Ceph client distribution. At this increment the plugin uses the librados C
  lifecycle/configuration surface (`rados_create2`, `rados_conf_read_file`,
  `rados_conf_set`, `rados_connect`, `rados_ioctx_create`,
  `rados_ioctx_destroy`, `rados_shutdown`). During setup, meson compiles and
  links a probe program using exactly this surface, and the plugin is built
  only when the probe succeeds; the runtime librados must be ABI/link
  compatible with the build headers (there is no runtime version
  negotiation).
- **A reachable Ceph cluster** with a readable Ceph configuration file and a
  client entity authorized to open the configured pool.

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
- **Discovery:** `pkg-config` lookup by name `rados` is tried first; when it
  fails the build falls back to a C++ compiler library lookup. Ubuntu splits
  the C and C++ librados development packages and ships no `rados.pc`, so
  installing `librados-dev` and `libradospp-dev` there still builds the
  plugin.
- **Dynamic (default):** a shared object `libplugin_RADOS.so` is
  installed under `<libdir>/plugins`. At runtime the plugin manager loads
  backends by name from `$NIXL_PLUGIN_DIR`, or from the `plugins/` directory
  next to the loading NIXL library. In debug builds the plugin path is also
  recorded in a `pluginlist` file in the build directory, which a debug-built
  NIXL library reads for discovery without installation.
- **Static:** adding `RADOS` to `-Dstatic_plugins` builds a static
  library linked into the core NIXL library instead of a loadable plugin; the
  same librados probe and configuration parameters apply.
- **Tests:** deterministic gtests exercise the engine and parameter parser
  with an injected fake lifecycle seam and perform no Ceph connection or I/O;
  agent-driven discovery tests load the built shared plugin through
  `NIXL_PLUGIN_DIR` and need no cluster. They run in the ordinary test suite.

## API Reference

### Custom parameters

Passed as the `nixl_b_params_t` map to `createBackend("RADOS", ...)`:

| Parameter | Description | Required |
|-----------|-------------|----------|
| `conf` | Path to a readable Ceph configuration file | Yes |
| `name` | Full Ceph entity name (normally `client.<id>`) | Yes |
| `pool` | RADOS pool opened at backend initialization | Yes |
| `ceph.<option>` | Any other parameter; the prefix is stripped and the option/value are applied verbatim to the Ceph configuration (empty values allowed) | No |

Backend creation is synchronous: at initialization the backend creates a
cluster handle with `name`, reads `conf`, applies the `ceph.<option>`
overrides, connects, and opens the pool — any of which may block on Ceph
configuration, authentication, connection, or pool-open work. Overrides
therefore always take effect after the configuration file is read and before
the connection is made. Authentication and keyring resolution are handled by
Ceph through `conf` and the overrides; the plugin interprets nothing beyond
the three required keys. A missing or empty `conf`, `name`, or `pool`, a
parameter that is neither one of those keys nor `ceph.<option>` (including
the bare key `ceph.`), or a failure to connect or open the pool makes
`createBackend()` return an error without a usable backend. The opened pool
is fixed for the backend instance's lifetime; working with several pools
means creating separate backend instances (each with its own `pool`
parameter), as the public agent API permits.

### Registration and object-key mapping

- Only `DRAM_SEG` and `OBJ_SEG` can be registered with this backend.
- `DRAM_SEG` registration is metadata-free: it does not pin, own, modify, or
  write the caller's buffer and reserves nothing in the backend.
- Each `OBJ_SEG` registration records only an in-memory `devId` → object-key
  mapping inside the backend: it does not create, reserve, lock, or stat any
  object in Ceph. The object key is the non-empty `metaInfo` verbatim, or the
  decimal `devId` when `metaInfo` is empty, so `devId` 0 with empty `metaInfo`
  resolves to key `"0"`. A `devId` that is already registered is rejected with
  `NIXL_ERR_INVALID_PARAM`; different `devId` values may deliberately resolve
  to the same object key. After the type and initialization checks, a
  `metaInfo` containing an embedded NUL byte is rejected with
  `NIXL_ERR_INVALID_PARAM` before registration reserves anything, because
  librados object operations take the key as a C string, so a NUL would
  silently truncate the effective key.
- Registration failures after a usable backend (metadata allocation failure)
  surface as `NIXL_ERR_BACKEND`; on an initialization-errored backend every
  registration returns `NIXL_ERR_BACKEND` and leaves the output metadata
  null.
- Deregistration performs no RADOS I/O: it removes the in-memory mapping only
  (it neither deletes nor truncates any object) and erases a reservation only
  when both the `devId` and key of the passed metadata still match, so a
  stale descriptor of a re-registered `devId` cannot drop the live mapping.
  A null metadata pointer is a no-op success.

### Transfers and queries

Not yet implemented: `prepXfer()`, `postXfer()`, `checkXfer()`,
`releaseReqH()`, and `queryMem()` all return `NIXL_ERR_NOT_SUPPORTED`.

### Key classes

- `nixlRadosEngine` (`rados_backend.h/.cpp`): the backend engine
  implementing bootstrap, capability declaration, and registration.
- `nixl::rados::parseCustomParams()` (`rados_config.h`):
  parameter parsing and validation for the table above.
