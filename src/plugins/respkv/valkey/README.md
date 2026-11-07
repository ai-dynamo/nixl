<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL Valkey Plugin

The `VALKEY` plugin stores NIXL objects in Valkey with the
[valkey-large-object](https://github.com/valkey-io/valkey-large-object) module. Payloads
move by server-initiated RDMA (AWS EFA, or libfabric's tcp provider for development)
through the RDMA C API of [Valkey GLIDE](https://github.com/valkey-io/valkey-glide): RESP
carries control, and the server reads or writes the NIXL buffer directly.
Family layout and the shared `nixlRespKvEngine` base: [`../README.md`](../README.md).
GLIDE build instructions stay in this file.

| NIXL | Valkey |
|---|---|
| `OBJ_SEG` descriptor | key: `key_prefix` + `metaInfo`, or the decimal `devId` |
| `NIXL_WRITE` | `BLOB.RDMA_SET` |
| `NIXL_READ` | `BLOB.RDMA_GET` |
| `queryMem` | `BLOB.INFO key LEN`, one pipeline for all keys; the response carries `size` |

A read of a missing key returns `NIXL_ERR_NOT_FOUND`; a stored size different from the
descriptor returns `NIXL_ERR_MISMATCH`.

## Design

```text
valkey_backend.h/.cpp  nixlValkeyEngine: descriptors, registration, worker pool
valkey_client.h/.cpp   GLIDE FFI and the module commands
valkey_plugin.cpp      plugin registration
```

`postXfer` turns each descriptor into a job for a pool of worker threads, each owning a
blocking GLIDE client. A client runs one transfer at a time, so `num_workers` is the
transfer concurrency. A separate client without RDMA serves `queryMem`.

- `registerMem` registers a `DRAM_SEG` region with every worker's client, in parallel from
  the calling thread, and fails if any worker cannot. A transfer passes its descriptor's
  window of the region, so register a KV pool as one region.
- `deregisterMem` frees those registrations before returning. It does not wait for
  transfers in progress: as with other backends, memory a transfer still uses must not be
  deregistered.
- `DRAM_SEG` only: GLIDE registers host memory (no `FI_HMEM`). `VRAM_SEG` can follow once
  GLIDE registers device memory.

A backend that cannot use RDMA (GLIDE built without it, libfabric not loadable, no fabric
on the server) fails at creation.

## Limitations

- A descriptor must cover the whole object (BLOB has no range commands): an object offset
  other than 0, or a local buffer shorter than the object, returns `NIXL_ERR_NOT_SUPPORTED`.
- Standalone Valkey only.
- An RDMA transfer has no timeout unless `transfer_timeout_ms` is set; without it,
  `releaseReqH` waits for a transfer that never completes. Cancelling drops the worker's
  client, so its registrations are redone on its next transfer.
- Every worker's client registers each region, so the RDMA device sees the region size
  times `num_workers`; past its registration limit `registerMem` fails with
  `NIXL_ERR_BACKEND`.
- The GLIDE RDMA API and the module are pre-1.0.

## Dependencies

From valkey-glide `v2.6.0-rc2` or later (RDMA is not in a stable release yet), under one
prefix (`-Dglide_path`): the `ffi` crate as a static library with RDMA, its C header from
GLIDE's cbindgen config, and `connection_request.proto` (the connection request is
generated from it, as in GLIDE's C++ client). GLIDE's symbols stay private to the plugin.

```bash
cd <valkey-glide>   # git checkout v2.6.0-rc2
(cd ffi && cargo build --release --features rdma)
cargo install --locked cbindgen@0.29.4
mkdir -p <prefix>/lib <prefix>/include <prefix>/proto
cp ffi/target/release/libglide_ffi.a <prefix>/lib/
(cd ffi && cbindgen --config cbindgen.toml --crate glide-ffi --lang c \
    --output <prefix>/include/glide_ffi.h)
cp glide-core/src/protobuf/connection_request.proto <prefix>/proto/
meson setup build -Dglide_path=<prefix>
```

Also `protoc` and libprotobuf at build time, and at runtime libfabric 2.4+ (GLIDE loads it
with `dlopen`, preferring `/opt/amazon/efa`; `GLIDE_LIBFABRIC_PATH` overrides) and a Valkey
server with the module.

If VALKEY is explicitly enabled or static, a missing dependency is a configuration error;
a default all-plugin build skips VALKEY with a warning.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `host`, `port` | `127.0.0.1`, `6379` | Valkey server |
| `username`, `password` | empty | ACL credentials; `username` requires `password` |
| `rdma_provider` | `efa-direct` | `efa-direct`, or `tcp` (libfabric emulation, for development) |
| `rdma_interface` | empty | fabric domain when the host has several |
| `key_prefix` | empty | prepended to every key |
| `timeout_ms` | `30000` | connect and command timeout |
| `transfer_timeout_ms` | `0` | cancel an RDMA transfer after this long and fail its request; `0` waits for the server |
| `num_workers` | `8` | worker threads, one client each |
| `write_checksum` | `false` | the server verifies a CRC-32c of each write (one CPU pass over the data); reads are always verified |

Invalid values fail backend creation.

## Build and test

```bash
meson setup build -Denable_plugins=VALKEY -Dglide_path=<prefix>
meson compile -C build VALKEY            # build/src/plugins/respkv/valkey/libplugin_VALKEY.so
```

The unit tests run without a server and need a non-release build:

```bash
meson setup build-tests --buildtype=debug -Dbuild_tests=true \
    -Denable_plugins=VALKEY -Dglide_path=<prefix>
meson compile -C build-tests unit
meson devenv -C build-tests ./test/gtest/unit/unit --gtest_filter='valkey*'
```

For a static build, add `-Dstatic_plugins=VALKEY`.

## Usage

Start Valkey **8.0+** with the module. Loadmodule size args must be **raw bytes** (not `2gb`).
Without EFA, use `fabric-provider Emulated` and NIXL `rdma_provider=tcp`:

```bash
./src/valkey-server --bind 127.0.0.1 --port 6379 --save "" \
    --loadmodule /path/to/libvalkey_large_object.so \
        operating-mode Dram \
        segment-size 268435456 \
        max-object-size 67108864 \
        fabric-provider Emulated          # EfaDirect on EFA hosts
```

### nixlbench smoke (container)

Build the image with VALKEY; GLIDE FFI is compiled **inside Docker** (needs network for
`git`/`cargo`). Optionally pass `--glide-path` if you already have a host prefix.

```bash
cd benchmark/nixlbench/contrib
./build.sh --nixl ../.. --enable-valkey
# optional: --valkey-glide-ref v2.6.0-rc2
# optional: --glide-path /path/to/prefix   # skip in-image cargo
# image: nixlbench:v*.dev.*
```

Inside the container (`--network host`), with Valkey+large-object already listening on the host:

```bash
export VALKEY_HOST=127.0.0.1 VALKEY_PORT=6379
export VALKEY_RDMA_PROVIDER=tcp   # default in nixlbench if unset
nixlbench --help | grep VALKEY

nixlbench --backend=VALKEY --runtime_type=ASIO --op_type=WRITE \
  --start_block_size=4096 --max_block_size=4096 \
  --start_batch_size=1 --max_batch_size=1 --pipeline_depth=1 \
  --num_threads=1 --total_buffer_size=67108864 \
  --warmup_iter=32 --num_iter=208 --check_consistency=true

nixlbench --backend=VALKEY --runtime_type=ASIO --op_type=READ \
  --start_block_size=4096 --max_block_size=4096 \
  --start_batch_size=1 --max_batch_size=1 --pipeline_depth=1 \
  --num_threads=1 --total_buffer_size=67108864 \
  --warmup_iter=32 --num_iter=208 --check_consistency=true
```

READ seeding uses `BLOB.TCP_SET` via `valkey-cli`/`redis-cli`; transfers still use `BLOB.RDMA_*`.
Then run REDIS the same way with `--backend=REDIS` and `REDIS_HOST`/`REDIS_PORT` against a Redis
server (see [`../redis/README.md`](../redis/README.md)).

Objects are `OBJ_SEG` descriptors whose `metaInfo` is the key; the local side is a registered
`DRAM_SEG` buffer. A descriptor covers one whole object.

```cpp
nixl_b_params_t params = {{"host", "10.0.0.1"}, {"rdma_provider", "efa-direct"}};
nixlBackendH *backend;
agent.createBackend("VALKEY", params, backend);

nixl_reg_dlist_t dram(DRAM_SEG), obj(OBJ_SEG);
dram.addDesc(nixlBlobDesc((uintptr_t)buf, len, 0, ""));
obj.addDesc(nixlBlobDesc(0, len, 0, "kv/chunk-42"));
agent.registerMem(dram);
agent.registerMem(obj);

nixlXferReqH *req;
agent.createXferReq(NIXL_WRITE, dram.trim(), obj.trim(), agent_name, req);
agent.postXferReq(req);
while (agent.getXferStatus(req) == NIXL_IN_PROG) {}
```

```python
agent = nixl_agent("a", nixl_agent_config(backends=[]))
agent.create_backend("VALKEY", {"host": "10.0.0.1", "rdma_provider": "efa-direct"})
agent.register_memory([(buf.data_ptr(), n, 0, "")], mem_type="DRAM")
agent.register_memory([(0, n, 0, "kv/chunk-42")], mem_type="OBJ")
```
