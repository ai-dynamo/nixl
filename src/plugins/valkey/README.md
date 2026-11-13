<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL Valkey Plugin

The `VALKEY` plugin lets a NIXL agent read and write objects in a [Valkey](https://valkey.io) server running the [valkey-large-object](https://github.com/valkey-io/valkey-large-object) module. The payload does not go through the client connection. Instead, the server moves it directly to or from your registered host buffer over RDMA (AWS EFA in production, or TCP for development). Use it as a remote store for large blobs such as KV-cache blocks.

The plugin supports:

- `NIXL_WRITE` and `NIXL_READ` between a local `DRAM_SEG` buffer and a remote `OBJ_SEG` object, one whole object per descriptor.
- `queryMem` to check which objects exist and how large they are.
- Standalone or cluster mode, TLS, and ACL authentication.

Requirements:

- A Valkey 8.0 or later server with the valkey-large-object module and a fabric the client can reach.
- [Valkey GLIDE](https://github.com/valkey-io/valkey-glide) `v2.6.0-rc2` or later, built with RDMA, at build time (see [Building](#building)).
- libfabric 2.4+ at runtime.

## Building

### 1. Build GLIDE

The plugin needs three artifacts from GLIDE: the `ffi` crate as a static library, its C header, and the connection request `.proto`. `cargo build` leaves them scattered in the GLIDE source tree, so you collect them into a directory of your choice and pass it to meson as `-Dglide_path`. The examples below use `$HOME/glide-rdma`:

```text
$HOME/glide-rdma/lib/libglide_ffi.a
$HOME/glide-rdma/include/glide_ffi.h
$HOME/glide-rdma/proto/connection_request.proto
```

RDMA support is not in a GLIDE stable release yet, so these steps build a release candidate:

```bash
GLIDE_PATH=$HOME/glide-rdma

git clone --branch v2.6.0-rc2 https://github.com/valkey-io/valkey-glide
cd valkey-glide
(cd ffi && cargo build --release --features rdma)
cargo install --locked cbindgen@0.29.4

mkdir -p $GLIDE_PATH/lib $GLIDE_PATH/include $GLIDE_PATH/proto
cp ffi/target/release/libglide_ffi.a $GLIDE_PATH/lib/
(cd ffi && cbindgen --config cbindgen.toml --crate glide-ffi --lang c \
    --output $GLIDE_PATH/include/glide_ffi.h)
cp glide-core/src/protobuf/connection_request.proto $GLIDE_PATH/proto/
```

`protoc` and libprotobuf must also be installed. GLIDE's symbols stay private to the plugin, so the plugin can coexist with another GLIDE in the same process (for example the Python client).

### 2. Build NIXL with the plugin

```bash
meson setup build -Denable_plugins=VALKEY -Dglide_path=$HOME/glide-rdma
meson compile -C build VALKEY            # build/src/plugins/valkey/libplugin_VALKEY.so
```

For a static plugin, add `-Dstatic_plugins=VALKEY`.

If `VALKEY` is explicitly enabled or static and a dependency is missing, `meson setup` fails with an error. A default all-plugin build skips `VALKEY` with a warning instead.

## Server requirements

The server must load the valkey-large-object module, and the module's `fabric-provider` must match the plugin's `rdma_provider`:

| Host | Module `fabric-provider` | Plugin `rdma_provider` |
|---|---|---|
| EFA (production) | `EfaDirect` | `efa-direct` (default) |
| No fabric (development) | `Emulated` | `tcp` |

The plugin works with both of the module's operating modes, `Dram` and `Tiered`. The `Emulated`/`tcp` combination runs the same RDMA protocol over TCP, so it works on hosts without EFA:

```bash
valkey-server --bind 127.0.0.1 --port 6379 --save "" \
    --loadmodule /path/to/libvalkey_large_object.so \
        operating-mode Dram \
        fabric-provider Emulated
```

See the [module's documentation](https://github.com/valkey-io/valkey-large-object#configuration) for its other options.

## Configuration

Pass these as backend parameters to `createBackend("VALKEY", ...)`. Invalid values fail backend creation.

Connection:

| Parameter | Default | Description |
|---|---|---|
| `host`, `port` | `127.0.0.1`, `6379` | Valkey server, or any node of a cluster |
| `cluster_mode` | `false` | `true` discovers the cluster from `host` and routes each command by key |
| `tls` | `off` | `on` verifies the server certificate, `insecure` skips verification |
| `username`, `password` | empty | ACL credentials; `username` requires `password` |
| `timeout_ms` | `30000` | connect and command timeout |

RDMA:

| Parameter | Default | Description |
|---|---|---|
| `rdma_provider` | `efa-direct` | `efa-direct` on EFA hosts; `tcp` for development against an `Emulated` server |
| `rdma_interface` | empty | fabric domain to use when the host has several |
| `transfer_timeout_ms` | `0` | abort a transfer that has not completed within this time; `0` waits for the server indefinitely (see [Limitations](#limitations)) |
| `write_checksum` | `false` | `true` makes the server verify a CRC-32c of every write, at the cost of one CPU pass over the data; reads are always verified |

At runtime GLIDE loads libfabric with `dlopen`, looking under `/opt/amazon/efa` first. Set `GLIDE_LIBFABRIC_PATH` to use a different libfabric.

Objects and concurrency:

| Parameter | Default | Description |
|---|---|---|
| `key_prefix` | empty | prepended to every object key |
| `num_workers` | `8` | worker threads, one Valkey client each; this is the number of transfers in flight |

```cpp
nixlBackendH *backend;
nixl_b_params_t params = {
    {"host", "10.0.0.1"},
    {"key_prefix", "kv/"},
    {"num_workers", "16"},
};
agent.createBackend("VALKEY", params, backend);
```

## Transfer behavior

### Descriptors

| Side | Memory type | Descriptor |
|---|---|---|
| Local buffer | `DRAM_SEG` | address and length of a registered host buffer |
| Valkey object | `OBJ_SEG` | `metaInfo` is the key; if empty, the decimal `devId` is the key. `len` is the object size |

`key_prefix` is prepended to the key in either case. A descriptor always covers one whole object, so the `OBJ_SEG` offset (`addr`) must be 0 and the local descriptor's `len` must equal the object's size.

Register one large `DRAM_SEG` region (for example a whole KV pool) rather than many small ones. Each transfer uses just its descriptor's window of the region, and each registration is replicated across all `num_workers` clients (see [Limitations](#limitations)).

### Transfers

| NIXL call | Valkey command | Notes |
|---|---|---|
| `NIXL_WRITE` | `BLOB.RDMA_SET` | server reads the local buffer and stores it under the key |
| `NIXL_READ` | `BLOB.RDMA_GET` | server writes the object into the local buffer |
| `queryMem` | `BLOB.INFO key LEN` | one pipelined round trip for all keys; each found object reports `size` |

A read of a missing key fails with `NIXL_ERR_NOT_FOUND`; a read of an object whose stored size differs from the descriptor's `len` fails with `NIXL_ERR_MISMATCH`.

## Limitations

- **Whole objects only.** The module has no range commands, so a descriptor cannot address part of an object. An `OBJ_SEG` offset other than 0, or a local buffer shorter than the object, fails with `NIXL_ERR_NOT_SUPPORTED`.
- **Host memory only.** GLIDE registers host memory (no `FI_HMEM`), so `VRAM_SEG` is not supported yet.
- **A stalled transfer is only detected by `transfer_timeout_ms`.** The server sets no completion deadline, so with the default `0` a transfer the server never completes blocks `releaseReqH`, and `deregisterMem` of its region, forever. Set `transfer_timeout_ms` in production. Leave plenty of headroom over the time a normal transfer of your largest object takes, because a timeout cancels the transfer and drops the worker's connection, and the worker must then reconnect and re-register every region.
- **Registration cost scales with `num_workers`.** Every worker's client registers each `DRAM_SEG` region, so the RDMA device sees `region size × num_workers`. Past the device's registration limit `registerMem` fails with `NIXL_ERR_BACKEND`. Prefer fewer, larger regions.

## How it works

```text
valkey_backend.h/.cpp  nixlValkeyEngine: descriptors, registration, worker pool
valkey_client.h/.cpp   GLIDE C API wrapper and the module commands
valkey_plugin.cpp      plugin registration
```

Commands and replies travel over RESP as usual. The payload does not: the plugin passes the NIXL buffer to GLIDE's RDMA C API, and the server reads or writes it directly.

`postXfer` validates every descriptor, then queues one job per descriptor for a pool of `num_workers` threads. Each worker owns a blocking GLIDE client that runs one transfer at a time, so `num_workers` is the transfer concurrency. A separate client without RDMA serves `queryMem`; in cluster mode it is a non-atomic batch across nodes.

`registerMem` on a `DRAM_SEG` registers the region with every worker's client in parallel from the calling thread and fails if any worker cannot. `deregisterMem` releases those registrations before returning, after waiting for any transfer still using the region. When `transfer_timeout_ms` is set, a watchdog thread closes the RDMA regions of any worker whose transfer is overdue. That transfer fails, the worker drops its client, and it reconnects and re-registers its regions on its next transfer.
