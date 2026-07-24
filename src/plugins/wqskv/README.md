# WQSKV Backend Plugin

NIXL backend over the WDS KVCache vendor library (`libwclient_kvcache.so`).
Wraps the C-style vendor API (`wds_kvcache_init` / `wds_kvcache_put` /
`wds_kvcache_get_vec`, ...) so a NIXL agent can `PUT`/`GET` DRAM buffers into
WDS KVCache directly, without going through the Mooncake store.

## Scope

- Memory types: `DRAM_SEG` only
- Locality: local-only backend (`supportsRemote() == false`,
  `supportsLocal() == true`)
- Notifications: not supported

## Build Requirements

The plugin links against the WDS KVCache client ABI declared in
`kv_interface.h` and built into `libwclient_kvcache.so`. Two
implementations of that ABI are supported and the plugin itself does not
care which is installed:

- **`wqs-mock`** — an Apache-2.0 in-memory mock at
  [https://github.com/<owner>/wqs-mock](https://github.com/<owner>/wqs-mock).
  Recommended for development, CI, and smoke tests. The mock implements
  `wds_kvcache_init` / `wds_kvcache_put` / `wds_kvcache_get_vec`
  faithfully and stubs the rest of the ABI with `-ENOSYS`.
- **The proprietary WDS client library** maintained by the vendor.
  Required for production deployments that need the real distributed
  KVCache backend. Install it to the same prefix to replace the mock at
  runtime; the plugin source is unchanged.

The plugin is conditionally compiled. `meson setup` enables it only when
both of the following are satisfied at configure time:

1. `libwclient_kvcache.so` (either implementation) is discoverable.
   Either:
   - Pass `-Dwqskv_lib_path=/path/to/dir` at configure time, **or**
   - Make sure the directory is on the linker search path (e.g.
     `export LIBRARY_PATH=/path/to/dir`).
2. `jsoncpp` is installed and visible via `pkg-config`.

Header location for `kv_interface.h` can be controlled with
`-Dwqskv_inc_path=/path/to/include`.

Typical build against the mock (after `make install`-ing wqs-mock to
`/usr/local`):

```bash
meson setup build -Denable_plugins=WQSKV
```

(no explicit `wqskv_lib_path` / `wqskv_inc_path` needed; the mock's
default install layout is on the standard search paths.)

Build against a custom location:

```bash
meson setup build -Denable_plugins=WQSKV \
  -Dwqskv_lib_path=/path/to/lib \
  -Dwqskv_inc_path=/path/to/include
```

Disable explicitly:

```bash
meson setup build -Ddisable_plugins=WQSKV
```

When neither `-Denable_plugins` nor `-Ddisable_plugins` mentions WQSKV,
the plugin is built opportunistically: present if the deps resolve,
silently skipped otherwise.

## Runtime Configuration

The vendor library requires a JSON config consumed once per process by
`wds_kvcache_init`. The path is resolved in this order:

1. `customParams["config_path"]` passed to `createBackend` (highest priority)
2. `WDS_BACKEND_CONFIG_PATH` environment variable

Required JSON keys (see `loadKVCacheOptionsFromJson` in `wqskv_backend.cpp`):

| Key             | Type     | Notes                                          |
| --------------- | -------- | ---------------------------------------------- |
| `poolid`        | uint     | WDS pool id                                    |
| `thread_num`    | int      | vendor worker threads                          |
| `thread_mode`   | string   | `"poll"` or `"event"`                          |
| `wengine_conf`  | string   | path to WDS wengine.conf                       |
| `node_id`       | uint     | local node id                                  |
| `bvar_port`     | uint     | bvar dummy port                                |
| `bind_cpus`     | string   | comma-separated CPU ids, e.g. `"20,21"`        |
| `mem_size`      | uint     | mempool size (MiB)                             |

Optional:

| Key              | Type | Default                  |
| ---------------- | ---- | ------------------------ |
| `mempool_objsz`  | uint | vendor default           |
| `mempool_objnum` | uint | vendor default           |
| `use_round_robin`| bool | vendor default           |
| `wds_debug_log`  | bool | false                    |

The schema is identical to the one used by
`mooncake/mooncake-store/src/wds/wds_backend.cpp`, so a single config file
serves both consumers. Reference examples live in `config/wdsBachend*.json`.

`wds_kvcache_init` is guarded by `std::call_once` — only the first
`createBackend` call performs vendor init; subsequent calls reuse the global
init state.

## API Reference

### Core Class

- **`nixlWQSKVEngine`** (`wqskv_backend.h`) — concrete `nixlBackendEngine`
  implementation. Constructed by NIXL when an agent calls
  `create_backend("WQSKV", params)`. The constructor loads the JSON config
  and performs `wds_kvcache_init` exactly once per process
  (`std::call_once`).

### Overridden Methods

| Method | Behavior |
| --- | --- |
| `registerMem` | Accepts only `DRAM_SEG`. Records `devId -> key` using `nixlBlobDesc::metaInfo`; falls back to `std::to_string(devId)` when `metaInfo` is empty. |
| `deregisterMem` | Frees the per-registration metadata (`nixlWQSKVMetadata`). |
| `queryMem` | Returns availability info per descriptor; no extra capability bits. |
| `prepXfer` | Validates `op ∈ {NIXL_WRITE, NIXL_READ}` and both `local`/`remote` are `DRAM_SEG`. Allocates a request handle and pre-parses `opt_args->customParam` into a per-descriptor key vector when supplied. |
| `postXfer` | One descriptor → one vendor call (`wds_kvcache_put` for `NIXL_WRITE`, `wds_kvcache_get_vec` for `NIXL_READ`). Returns `NIXL_IN_PROG`; completion is signaled by the vendor callback decrementing a counter on the handle. |
| `checkXfer` | Lock-free read of the pending counter — `NIXL_SUCCESS` at zero, otherwise `NIXL_IN_PROG`. |
| `releaseReqH` | Waits on a condition variable until the counter reaches zero, then frees the handle. |
| `getSupportedMems` | Returns `{DRAM_SEG}`. |
| `supportsRemote` / `supportsNotif` | Return `false`. |
| `supportsLocal` | Returns `true`. |
| `connect` / `disconnect` / `loadLocalMD` / `unloadMD` | Trivially succeed; this backend has no remote-state lifecycle. |

### Per-Backend Initialization Parameters (`customParams`)

| Key | Required | Effect |
| --- | --- | --- |
| `config_path` | Yes, unless `WDS_BACKEND_CONFIG_PATH` is set | Absolute path to the WDS KVCache JSON config consumed by `wds_kvcache_init`. |

### Per-Transfer Optional Args (`nixl_opt_b_args_t::customParam`)

A `'\n'`-delimited string of keys. When the count matches the descriptor
count exactly, each descriptor's key is overridden for that single
transfer. Useful for `NIXL_WRITE` workloads that need a fresh key per
iteration to avoid WDS's no-overwrite rule.

## Example Usage

### Python

```python
from nixl._api import nixl_agent, nixl_agent_config

agent = nixl_agent("agent-1", nixl_agent_config(backends=[]))

params = agent.get_plugin_params("WQSKV")
params["config_path"] = "/path/to/wdsBackendClient.json"
agent.create_backend("WQSKV", params)

# Register DRAM buffer; metaInfo becomes the WDS key for this registration.
reg_dlist = [(buf_addr, buf_len, 0, "my-kvcache-key")]
agent.register_memory(reg_dlist, "DRAM")

# Issue a local PUT.
xfer = agent.initialize_xfer(
    "WRITE",
    [(buf_addr, buf_len, 0)],   # local descriptors
    [(0, buf_len, 0)],          # remote descriptors (devId 0 → "my-kvcache-key")
    "agent-1",                  # local-only: remote_agent == self
)
state = agent.transfer(xfer)
while state not in ("DONE", "ERR"):
    state = agent.check_xfer_state(xfer)
agent.release_xfer_handle(xfer)
```

### C++

```cpp
#include "nixl.h"

nixl_b_params_t params;
params["config_path"] = "/path/to/wdsBackendClient.json";

nixlAgent agent("agent-1", nixlAgentConfig{});
nixlBackendH *bh = nullptr;
agent.createBackend("WQSKV", params, bh);

// Register a DRAM buffer with an explicit key via metaInfo.
nixl_reg_dlist_t regs(DRAM_SEG);
regs.addDesc({buf_addr, buf_len, /*devId=*/0, "my-kvcache-key"});
agent.registerMem(regs);

// PUT: local DRAM → WDS KVCache under "my-kvcache-key".
nixl_xfer_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
local.addDesc({buf_addr, buf_len, 0});
remote.addDesc({0, buf_len, 0});      // devId 0 → "my-kvcache-key"

nixlXferReqH *req = nullptr;
agent.createXferReq(NIXL_WRITE, local, remote, "agent-1", req);
agent.postXferReq(req);
while (agent.getXferStatus(req) == NIXL_IN_PROG) { /* spin or wait */ }
agent.releaseXferReq(req);
```

For benchmark-driven usage, see
`benchmark/nixlbench/src/worker/nixl/nixl_worker.cpp`
(`execWqskvWriteIterations` / `execWqskvReadIterations`), which demonstrates
per-iteration key rotation via `opt_args->customParam`.

## Key Resolution

Each transfer descriptor maps to one vendor key:

- `registerMem` records `devId -> key` using `nixlBlobDesc::metaInfo` as the
  key (falling back to `std::to_string(devId)` when `metaInfo` is empty).
- During `postXfer`, the key is taken from `nixlMetaDesc::metadataP` if
  present, otherwise looked up via `devId` in the `devIdToKey_` map.
- For per-transfer overrides (e.g. nixlbench warmup+bench needing a fresh key
  per iteration to avoid WDS's no-overwrite rule), pass one key per descriptor
  as a `'\n'`-delimited blob in `opt_args->customParam`. The override is
  accepted only when the key count matches the descriptor count exactly.

## Async Semantics

`postXfer` is asynchronous. Each descriptor becomes one vendor call; the
vendor invokes a per-request callback that decrements a pending counter held
on the request handle.

- `checkXfer` reads the counter lock-free.
- `releaseReqH` waits on a condition variable until the counter hits zero,
  then frees the handle.
