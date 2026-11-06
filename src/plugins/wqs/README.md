<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WQS storage backend

`WQS` transfers complete KV objects between caller-owned DRAM and WQS through the
existing `SharedClient` and a local wqsd. It supports `DRAM_SEG` and `OBJ_SEG`,
local storage operations, and no notifications. A remote WQS cluster does not
require another NIXL agent or NIXL metadata exchange.

## Dependencies

- Linux with `memfd_create` and 2 MiB hugepage support.
- A WQS SDK providing `shared_client.h`, its transitive headers, and
  `libwqs_store.so`. The SDK must support `register_buffer_with_fd`, `getSize`,
  `batch_get_into`, and `batch_put_from`.
- The matching SPDK/DPDK shared libraries required by that SDK.
- A local wqsd connected to a WQS cluster, with memory registration acknowledged
  before it accepts client I/O.

Obtain the production SDK and its runtime dependencies from the WQS distribution
for your platform. They are not downloaded or bundled by the NIXL build.
For CI and plugin development, the public
[WQS mock SDK](https://github.com/expontech/wqs_mock) supplies the same interfaces
without wqsd, SPDK, or a WQS cluster; see the validation instructions below.
The vLLM example additionally requires vLLM 0.27.1; the WQS plugin itself does not
depend on vLLM, PyTorch, or CUDA.

## Read and write behavior

Register each object with `(0, object_bytes, unique_dev_id, key)` in `OBJ_SEG`.
The key must be nonempty. Different registered objects need different device IDs.
DRAM registrations describe the caller's address range. Object registration does
not create an object; deregistration does not delete one.

Every posted READ, including a reposted handle, calls WQS `getSize(key)` on the
worker thread. The returned logical size must be positive and equal the NIXL
transfer length. A mismatch fails with `NIXL_ERR_MISMATCH` before the affected
sub-batch issues a data read. The queried sizes are passed to `batch_get_into`;
returned sizes are checked before copying that sub-batch into caller memory.
The application does not need to call `queryMem`. No size cache is used.

WRITE copies caller DRAM into the registered staging pool and calls
`batch_put_from`. Both directions split batches by pool capacity, entry count,
and serialized payload size. The WQS library handles its own DIF header and
padding; the plugin passes logical sizes, including sizes not divisible by 512.
Objects must be smaller than `INT_MAX` and fit individually in the staging pool.
Partial-object transfers and duplicate keys within one WRITE are rejected.

An optional `queryMem(OBJ_SEG)` returns `{"size": "<bytes>"}` for positive sizes.
The existing WQS wrapper folds some failures into zero, so an empty query result
can mean a cache miss or a transport failure. A READ size lookup returning zero
or negative fails with `NIXL_ERR_BACKEND`; it is never used as a guessed length.

The same key must retain the same contents and length across writers. `getSize`
and READ are separate calls and cannot protect against concurrent variable-size
replacement. Batch failure does not roll back earlier successful sub-batches.

## Runtime and lifetime

One process-wide WQS session owns one worker, one `SharedClient`, and one
hugepage memfd staging region. The production session is retained until process
exit because the existing WQS API cannot unregister that region or safely
reinitialize its SPDK runtime. Engine destruction drains its own queued work.
An initialization failure after WQS startup requires restarting the process.
WQS session settings must agree across engines; `key_prefix` may differ.

The caller must keep buffers alive until completion or request release. Releasing
an active request waits for buffer access to stop; queued requests are skipped
when released before execution. Deregistration also waits for in-flight users of
that registration. Reposting an active handle returns `NIXL_ERR_REPOST_ACTIVE`.
When the bounded queue is full, `postXfer` returns `NIXL_ERR_NOT_ALLOWED` without
submitting WQS I/O, and the handle can be retried.

There is no hard cancellation deadline. Existing WQS calls can wait indefinitely
on certain in-flight failures; staging memory is not reclaimed just because a
timeout elapsed. Coexistence with other SPDK/WQS clients in the same process
requires deployment validation. Do not initialize the session before forking.

## Configuration

Backend parameters are validated once; unknown parameters are rejected.
They use NIXL's standard string parameter map: pass them to
`createBackend("WQS", params, ...)` in C++ or `agent.create_backend("WQS", params)`
in Python. No separate plugin configuration channel is required.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `unix_path` | `/var/run/wds/wqs` | Existing local wqsd socket |
| `staging_pool_bytes` | `1073741824` | Pool bytes, a positive multiple of 2 MiB |
| `max_inflight` | `64` | Maximum queued plus executing jobs, including queries |
| `max_batch_entries` | `64` | Maximum objects in one WQS data call |
| `payload_bytes` | `8192` | Request/response metadata limit; must not exceed the WQS client's configured payload buffer size |
| `key_prefix` | empty | Prefix prepended to every storage key |

The pool uses 2 MiB hugepages. Reserve sufficient hugepages and provide memlock
permissions for WQS/SPDK before creating the backend. WQS native options continue
to use its existing `WQS_NATIVE_CONFIG_PATH`; the plugin does not change process
environment variables. The payload limit describes metadata, not object bytes.

## Build

Provide the WQS SDK headers and `libwqs_store.so` with their dependencies.
Replace `/path/to/wqs` below with the SDK installation prefix:

```bash
meson setup build-wqs \
  -Denable_plugins=WQS \
  -Dwqs_inc_path=/path/to/wqs/include \
  -Dwqs_lib_path=/path/to/wqs/lib
meson compile -C build-wqs
```

Explicit WQS enablement fails configuration if the headers or library are absent.
Default all-plugin builds skip WQS when its dependencies are unavailable. For a
built-in backend, also pass `-Dstatic_plugins=WQS`. Dynamic builds produce
`build-wqs/src/plugins/wqs/libplugin_WQS.so`. Make WQS and SPDK/DPDK shared
libraries available through the loader's configured paths; use `NIXL_PLUGIN_DIR`
for plugin discovery when necessary.

To build NIXL without this backend, pass `-Ddisable_plugins=WQS` instead of
`-Denable_plugins=WQS`.

For the optional WQS SDK staging supported by `contrib/Dockerfile`, see
[the container build instructions](../../../contrib/README.md#wqs-sdk).

## Backend interface

| Entry point | Behavior |
| --- | --- |
| `registerMem` / `deregisterMem` | Track caller DRAM or a complete object descriptor; do not create or remove WQS objects |
| `prepXfer` | Validate a whole-object transfer and allocate a reusable request handle |
| `postXfer` | Queue asynchronous READ or WRITE work on the session worker |
| `checkXfer` | Return pending, success, or failure status without waiting for WQS |
| `releaseReqH` | Wait for active buffer access to finish and release the handle |
| `queryMem` | Optionally return a positive object size or an empty result for a miss |

`nixlWqsEngine` implements these NIXL backend entry points. `nixlWqsSession`
serializes client calls and owns the bounded queue; `nixlWqsBufferPool` manages
the staging allocations. `iWqsClient` isolates the SDK calls for injected tests.

## Validation

### Public mock SDK

Build and install the `SharedClient` mock with a C++17 compiler and CMake 3.13 or
newer. The optional legacy KVCache library is not needed for this plugin:

```bash
git clone https://github.com/expontech/wqs_mock.git
git -C wqs_mock checkout 3dc51f00e4c2f359fbd3752fc8cd785b95f1a1e3
cmake -S wqs_mock -B wqs_mock/build \
  -DWQS_MOCK_BUILD_LEGACY=OFF \
  -DCMAKE_INSTALL_PREFIX=/path/to/wqs-mock \
  -DCMAKE_INSTALL_LIBDIR=lib
cmake --build wqs_mock/build --parallel
ctest --test-dir wqs_mock/build --output-on-failure
cmake --install wqs_mock/build
```

Use `/path/to/wqs-mock/include` and `/path/to/wqs-mock/lib` for the Meson options
below. These unit tests need neither hugepages nor a running wqsd. The mock uses
process-local storage and validates registered ranges and exact logical read
lengths. It does not model WQS transport, persistence, or performance. Build
production deployments against the production SDK.

### Plugin unit tests

The injected-client tests need no running wqsd or hugepages, but the test binary
still links the production plugin and its library dependencies. Use a non-release
build with GoogleTest and GoogleMock:

```bash
meson setup build-wqs-tests --buildtype=debug \
  -Denable_plugins=WQS -Dbuild_tests=true \
  -Dwqs_inc_path=/path/to/wqs/include \
  -Dwqs_lib_path=/path/to/wqs/lib
meson compile -C build-wqs-tests wqs_unit
meson test -C build-wqs-tests --no-rebuild --suite wqs --print-errorlogs
```

### Storage smoke test

Run the Python round-trip example against a bound, ready wqsd using the newly
built NIXL installation:

```bash
python3 examples/python/wqs_example.py --socket /var/run/wds/wqs --bytes 1048576
```

The example verifies every byte and does not call `queryMem`. It writes a unique
key by default and leaves that object to WQS eviction. The example is a storage
smoke test; vLLM still needs a storage-offloading adapter selecting the WQS backend.

## vLLM integration example

`examples/python/vllm_wqs_offload.py` provides an out-of-tree secondary tier for
vLLM 0.27.1's `OffloadingConnector`. It reuses vLLM's object-tier transfer and
completion handling with the WQS backend. vLLM owns GPU/CPU transfers and a CPU
primary cache; NIXL transfers complete CPU blocks to and from WQS. This example
depends on experimental vLLM Python interfaces and should be checked against
other versions before use.

Add the example directory to `PYTHONPATH`, configure plugin discovery and shared
library paths, then pass this configuration through `--kv-transfer-config`:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "WqsOffloadingSpec",
    "spec_module_path": "vllm_wqs_offload",
    "cpu_bytes_to_use": 2147483648,
    "secondary_tiers": [{
      "type": "wqs",
      "prefix": "my-model-cache",
      "backend_params": {
        "unix_path": "/var/run/wds/wqs",
        "staging_pool_bytes": "536870912"
      }
    }]
  }
}
```

The Python bindings, NIXL core, and WQS plugin must come from the same build.
An older NIXL EP package can load incompatible core libraries before the storage
adapter starts; changing `PYTHONPATH` alone does not prevent that conflict.

Use the same `PYTHONHASHSEED` and configuration across processes when testing
cache reuse. The tier uses vLLM's `FileMapper` keys, including its model and KV
layout digest. Existence lookups call `query_memory` to decide whether blocks
can be reused; the plugin still calls `getSize` for every actual READ. To prove
WQS reuse, restart the inference process between writing and reading so both
GPU and CPU primary caches start empty. A GPU prefix-cache reset alone does not
exclude CPU primary-cache hits.
