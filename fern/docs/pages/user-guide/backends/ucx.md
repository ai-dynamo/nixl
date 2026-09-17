---
title: UCX
description: UCX transfer backend for high-performance VRAM and DRAM transfers via RDMA and TCP.
---

## Overview

UCX is the general-purpose high-performance network transport backend in NIXL. It supports RoCE, InfiniBand, and TCP, making it the default backend for VRAM and DRAM transfers between nodes. UCX is automatically selected when no specific backend is requested and both agents have it initialized.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ VRAM; VRAM ↔ DRAM; DRAM ↔ DRAM |
| **Protocol** | RoCE, InfiniBand, TCP |
| **Best For** | GPU-to-GPU and CPU-to-CPU transfers between nodes |

## Installation

UCX is the default transfer backend and is included automatically with the `pip install nixl` package. For source builds, UCX must be built before NIXL.

### Build from Source

NIXL is tested with UCX version 1.22.x.

```bash
git clone https://github.com/openucx/ucx.git
cd ucx
git checkout v1.22.x
./autogen.sh
./contrib/configure-release-mt       \
    --enable-shared                    \
    --disable-static                   \
    --disable-doxygen-doc              \
    --enable-optimizations             \
    --enable-cma                       \
    --enable-devel-headers             \
    --with-cuda=<cuda install>         \
    --with-verbs                       \
    --with-dm                          \
    --with-gdrcopy=<gdrcopy install>
make -j
make -j install-strip
ldconfig
```

Replace `<cuda install>` with the path to your CUDA installation (e.g., `/usr/local/cuda`) and `<gdrcopy install>` with the path to your GDRCopy installation if available.

<Tip>
[GDRCopy](https://github.com/NVIDIA/gdrcopy) is optional but recommended for maximum GPU memory registration performance. UCX and NIXL work without it, but performance may be reduced for GPU-to-GPU transfers.
</Tip>

See [Configuration](#configuration) for build options.

## Configuration

### Environment Variables

<Markdown src="/snippets/env-vars-ucx.mdx" />

### Backend Plugin Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `connection_mode` | `worker_address` | How connections to remote agents are established. `worker_address` exchanges UCX worker addresses (legacy behavior). `sockaddr` makes every agent run a UCP listener and connect to its peers by IP:port, so that connection establishment is performed by a UCX connection manager such as RDMA CM. |
| `listen_address` | `0.0.0.0` | `connection_mode=sockaddr` only: local address the UCP listener binds to. |
| `listen_port` | `0` | `connection_mode=sockaddr` only: local port the UCP listener binds to. `0` lets the OS pick a free port, which is then advertised to peers. |
| `connect_timeout_ms` | `30000` | `connection_mode=sockaddr` only: how long `loadRemoteConnInfo()` waits for the client/server wireup to complete before failing. |
| `advertise_address` | *(empty)* | `connection_mode=sockaddr` only: address advertised to remote agents when `listen_address` is a wildcard. Required if `listen_address` is `0.0.0.0` or `::`. |

#### connection_mode=sockaddr

In this mode `getConnInfo()` returns a versioned, human readable blob describing the
local listener instead of a UCX worker address:

```text
NIXLUCXSA/1 inet 192.168.10.27 18515
```

Both agents of a pair must use the same `connection_mode`; a mismatch is rejected in
`loadRemoteConnInfo()` with `NIXL_ERR_INVALID_PARAM` instead of failing later at
transfer time.

To force connection establishment over RDMA CM (and fail instead of silently falling
back to TCP):

```bash
export UCX_TLS=rc,self,sm
export UCX_NET_DEVICES=mlx5_2:1
export UCX_SOCKADDR_TLS_PRIORITY=rdmacm
```

Notes and current limitations:

- One listener per backend instance, created on the first UCX worker. Endpoints of all
  local workers connect to the single listener of the peer, mirroring the
  `worker_address` mode where only the first worker address is advertised.
  Tested with `num_workers=1`.
- `loadRemoteConnInfo()` blocks until the endpoints are fully connected (it flushes
  them while progressing the local workers), because `ucp_ep_rkey_unpack()` - and hence
  `loadRemoteMD()` - requires a connected endpoint. Progressing the local workers in
  that loop also accepts the peer's incoming connection requests, so two agents
  connecting to each other simultaneously do not deadlock.
- The peer must be progressing its workers (progress thread, or `nixlAgent` calls) for
  incoming connections to be accepted.
- Endpoints accepted from incoming connection requests are not associated with a
  remote agent name: NIXL only ever sends on the local client endpoints, so accepted
  endpoints exist purely to complete the UCX wireup and to receive. They are released
  when the last remote connection is dropped and at engine destruction. This is
  sufficient for a one-to-one pair of agents.
- IPv4 and IPv6 are both accepted in the connection info format; IPv4 is what has been
  exercised so far.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ucx_path` | System path | Path to UCX installation. |

## When to Use

- **GPU-to-GPU transfers via RDMA** -- UCX leverages RoCE or InfiniBand for high-bandwidth, low-latency GPU memory transfers.
- **CPU-to-CPU with InfiniBand or RoCE** -- Standard high-performance network transport for host memory.
- **General-purpose fallback** -- UCX supports both VRAM and DRAM, making it suitable for most transfer scenarios.
