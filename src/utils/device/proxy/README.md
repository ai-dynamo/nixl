<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Device proxy

The host side of the CPU proxy. Where a transport has no device-callable submission path, GPU
kernels write fixed-layout records (`proxy_protocol.h`) into per-channel rings, and host threads
turn each record into an ordinary backend transfer through a `nixl::proxyTransport`.

## Transport contract

`nixl::proxyTransport` (`proxy_transport.h`) is the interface a backend implements to sit behind
the proxy. It is internal to `libnixl_device_proxy` and the plugins: not installed, not an ABI.

### Ownership

- `proxyRuntime::create()` takes ownership of the transport. The runtime calls `shutdown()` and
  destroys the transport after its workers have joined.
- Anything the transport references (engine workers, endpoints) must outlive the runtime. The
  owning engine shuts the runtime down first in its destructor.

### Threading

- `init()` and `shutdown()` run with no worker thread alive.
- `submit()`, `progress()` and `quiesce()` for any ring of channel c, and `checkCompletion()` for
  requests submitted on those rings, are made from a single thread at a time. Different channels
  may be served concurrently by different threads.
- `resolveDirectPtrs()` runs on an application thread, concurrently with the calls above, and must
  not touch per-ring state.
- `submit()`, `checkCompletion()` and `progress()` are the data path: no locks and no atomic
  read-modify-write shared across channels.

### Requests

- At most `config.ring_depth` requests are outstanding per (channel, peer).
- `submit()` returns `NIXL_IN_PROG` with a non-empty request, or a terminal status with an empty
  request (`NIXL_SUCCESS`: complete now; error: never posted). It must not return a transient
  error: a transport that cannot post now returns `NIXL_IN_PROG` and posts from `progress()`.
- The runtime polls a ring's requests in submission order and never passes a request again after
  `checkCompletion()` has returned a terminal status for it.
- A terminal status releases the handle. It does not promise that the transport has stopped
  accessing the operation's memory (UCX under err-mode none cannot abort a put). Only `quiesce()`
  promises that.
