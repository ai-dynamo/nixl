# NIXL SPDK Plugin

This plugin moves data between host DRAM and storage using
[SPDK](https://spdk.io) bdevs. It stands up a private, user-space SPDK runtime
(env, thread, accel, and bdev subsystems), opens the bdevs described by a
configuration, and services NIXL transfers as SPDK bdev I/O — the whole data
path runs in user space and bypasses the kernel block stack.

## No DPDK

The plugin links SPDK's static libraries but supplies its **own implementation of
SPDK's env interface** instead of `spdk_env_dpdk`, so DPDK is never linked or
initialized. Three consequences shape everything below:

- **No hugepages and no EAL setup.** Data buffers are ordinary page-aligned host
  memory; hugepages are used opportunistically if available but never required,
  and the process needs no special privileges.
- **No local PCI devices.** PCI probing is stubbed out, so a locally attached
  NVMe drive cannot be used. Transport-attached storage (NVMe-oF over TCP/RDMA)
  and in-memory or file-backed bdevs work normally.
- **No dynamically loaded bdev modules.** Only the modules statically linked into
  the plugin are available: `malloc`, `aio`, `null`, `nvme`, `passthru`, `delay`,
  `error`, `gpt`, `lvol`, `raid`, `split`, `ftl`, and `virtio`. To add another,
  extend `spdk_pc_components` in `src/plugins/spdk/meson.build`.

Build against SPDK **v26.05** or newer.

## Memory types and the transfer model

- **`DRAM_SEG`** is the local staging buffer (host memory).
- **`BLK_SEG`** is a block device (the "remote" side, even though the transfer
  is local/loopback).
- **`OBJ_SEG`** is a key-value object on an NVMe Key-Value device.

Transfers are always DRAM ↔ device, expressed as a loopback transfer whose remote
agent name is the agent's *own* name:

- `NIXL_WRITE`: DRAM → device (bdev write, or KV **Store** for `OBJ_SEG`)
- `NIXL_READ`:  device → DRAM (bdev read, or KV **Retrieve** for `OBJ_SEG`)

The plugin reports `supportsLocal() == true`, `supportsRemote() == false`, and
`supportsNotif() == false`.

## Registration contract

Both sides are registered with `registerMem` before use.

| Field      | `DRAM_SEG` (buffer)    | `BLK_SEG` (bdev)                  | `OBJ_SEG` (KV object)           |
|------------|------------------------|-----------------------------------|---------------------------------|
| `devId`    | (unused)               | (unused)                          | (unused)                        |
| `metaInfo` | (unused)               | **bdev name** (e.g. `Nvme0n1`)    | **object key** (1–16 bytes)     |
| `addr`     | Buffer virtual address | Byte offset into the bdev         | Must be `0` (whole-value)       |
| `len`      | Buffer length          | Accessible length in bytes        | Value length                    |

- **DRAM must be page-aligned (4 KiB) in address and length.** The plugin
  registers host memory directly for zero-copy DMA and does **not** fall back to
  bounce buffers, so registration hard-fails for memory it cannot map.
  `posix_memalign(&buf, 4096, len)` with a 4 KiB-multiple length is sufficient.
- **BLK transfers must be block-aligned**: offset and length must be multiples of
  the bdev block size (and of its write-unit size) and stay within capacity.
- **The bdev is selected by name** via `metaInfo`. An NVMe controller attached as
  `Nvme0` exposes its namespace as `Nvme0n1`.

## OBJ_SEG: NVMe Key-Value

`OBJ_SEG` maps a NIXL object onto a key on an NVMe Key-Value namespace. Requests
become raw NVMe KV commands (Store for `NIXL_WRITE`, Retrieve for `NIXL_READ`)
carried over **bdev NVMe passthru** — the plugin builds the `spdk_nvme_cmd`
itself, as SPDK has no KV bdev abstraction.

The KV device is the backend's `bdev_name` bdev. On the first `OBJ_SEG`
registration the plugin opens it and requires that it both supports NVMe passthru
(`SPDK_BDEV_IO_TYPE_NVME_IO`) and identifies as the KV command set
(`spdk_bdev_get_nvme_csi() == SPDK_NVME_CSI_KV`); anything else fails with
`NIXL_ERR_NOT_SUPPORTED`.

Keys are per-object, as in NIXL's other OBJ backends: the key comes from
`metaInfo` at registration and is recovered at transfer time. `metaInfo` is
required — there is no fallback, because deriving a key from another field would
silently collide when two descriptors shared it.

```cpp
// Store a value under the key "user:42", then read it back.
nixl_b_params_t params;
params["json_config"] =
    R"({"subsystems":[{"subsystem":"bdev","config":[
        {"method":"bdev_nvme_attach_controller",
         "params":{"name":"Kv0","trtype":"TCP","adrfam":"IPv4",
                   "traddr":"192.168.1.10","trsvcid":"4420",
                   "subnqn":"nqn.2016-06.io.spdk:kv0"}}
    ]}]})";
// bdev_name selects the KV bdev itself, so it is the namespace (Kv0n1), not
// the controller (Kv0) that the JSON above attaches.
params["bdev_name"] = "Kv0n1";
agent.createBackend("SPDK", params, spdk);

nixl_opt_args_t args;
args.backends.push_back(spdk);

void *buf = nullptr;
posix_memalign(&buf, 4096, 4096);

nixl_reg_dlist_t dram(DRAM_SEG);
dram.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buf), 4096, 1));
agent.registerMem(dram, &args);

nixl_reg_dlist_t obj(OBJ_SEG);              // one registration per key
obj.addDesc(nixlBlobDesc(0, 4096, 1, "user:42"));
agent.registerMem(obj, &args);

nixl_xfer_dlist_t src(DRAM_SEG), dst(OBJ_SEG);
src.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(buf), 4096, 1));
dst.addDesc(nixlBasicDesc(0, 4096, 1));     // addr must be 0
nixlXferReqH *req = nullptr;
agent.createXferReq(NIXL_WRITE, src, dst, agentName, req, &args);   // KV Store
```

**Whole-value only.** NVMe KV transfers the entire value, so the OBJ descriptor's
offset must be `0` and its length must equal the local length. On Retrieve the
caller sizes the buffer; the device's actual value length (returned in the
completion) is not yet surfaced.

> **Testing.** This is the *initiator* side; exercising it needs a target that
> exposes an NVMe KV namespace. The in-memory `malloc` bdev is not KV, so the
> bundled test only verifies that `OBJ_SEG` registration is correctly rejected on
> a non-KV device.

## Configuration

The backend is created with `createBackend("SPDK", params, …)`. All parameters
are string key/value pairs; call `getPluginParams("SPDK", …)` to discover them
and their defaults at runtime.

The bdev configuration is supplied one of three ways, in precedence order:

1. `json_config` — inline SPDK subsystem JSON
2. `json_config_file` — path to a file containing the same JSON
3. Convenience parameters (`bdev_type` + `bdev_name` + a type-specific source)

Exactly one source must be provided or `createBackend` fails.

### Convenience parameters

For the common single-bdev case you can skip SPDK JSON entirely:

| `bdev_type` | Required                                | Optional                       | Resulting bdev name |
|-------------|-----------------------------------------|--------------------------------|---------------------|
| `malloc`    | `bdev_name`, `bdev_num_blocks`          | `bdev_block_size` (def. `512`) | `bdev_name`         |
| `aio`       | `bdev_name`, `bdev_filename`            | `bdev_block_size`              | `bdev_name`         |
| `nvme`      | `bdev_name`, `bdev_trtype`, `bdev_traddr`, `bdev_subnqn` | `bdev_trsvcid`, `bdev_adrfam` | `<bdev_name>n1`     |

`bdev_trtype` must name a network transport (`RDMA` or `TCP`). `PCIe` is
rejected at configuration time: this plugin does not probe PCI devices — see
[No DPDK](#no-dpdk).

```cpp
// In-memory malloc device for testing — no JSON, no file:
nixl_b_params_t params;
params["bdev_type"]       = "malloc";
params["bdev_name"]       = "Malloc0";
params["bdev_num_blocks"] = "131072";   // 64 MiB at 512 B blocks
params["bdev_block_size"] = "512";
agent.createBackend("SPDK", params, spdk);
// ... register BLK_SEG with metaInfo = "Malloc0" ...
```

### General case: SPDK JSON

Any bdev module and options SPDK supports, e.g. an NVMe-oF target over TCP:

```cpp
params["json_config"] =
    R"({"subsystems":[{"subsystem":"bdev","config":[
        {"method":"bdev_nvme_attach_controller",
         "params":{"name":"Nvme0","trtype":"TCP","adrfam":"IPv4",
                   "traddr":"192.168.1.10","trsvcid":"4420",
                   "subnqn":"nqn.2016-06.io.spdk:cnode1"}}
    ]}]})";
// ... register BLK_SEG with metaInfo = "Nvme0n1" ...
```

A `trtype: PCIe` controller will not attach — see [No DPDK](#no-dpdk).

### Runtime parameters

| Parameter          | Default       | Meaning |
|--------------------|---------------|---------|
| `spdk_name`        | `nixl_spdk`   | Names the SPDK thread in logs and traces. |
| `core_mask`        | `""`          | Core mask or `[list]` for the SPDK thread's affinity. |
| `msg_mempool_size` | `0` (default) | SPDK thread message-pool size. The default (256K entries) is generous; lower it for a small footprint. |

`core_mask` and `msg_mempool_size` configure the process-wide SPDK runtime, so
only the backend that starts it applies them — see
[Multiple backends](#multiple-backends).

## Multiple backends

SPDK's env, thread, accel and bdev subsystems are singletons within this copy of
SPDK, so the plugin shares one runtime across the process: the first backend
brings it up, the last one to go away tears it down, in any order. Each backend
still gets its own SPDK thread, its own I/O channels and its own bdev
configuration. Because the bdev registry is shared, **each backend must name its
bdevs uniquely** — a second backend that configures a bdev the first already
created will fail to start.

Bdevs created by a backend live until the *last* backend in the process is
released, not until that backend is released. Sequential create/destroy/create
cycles are therefore fine; overlapping ones must use distinct names.

The plugin links SPDK privately, so it is unaffected by — and does not interfere
with — a host application that initializes its own copy of SPDK.

## Limitations

- **No local PCI / NVMe devices, no external bdev modules** — see
  [No DPDK](#no-dpdk).
- **No bounce-buffer path.** DRAM registration requires 4 KiB-aligned address and
  length; anything else is rejected rather than silently copied.
- **Local transfers only.** No remote-agent support and no notifications.
- **Config load needs a writable `/var/tmp`.** `spdk_subsystem_load_config`
  briefly opens an RPC socket there.
- **`OBJ_SEG` is whole-value KV**: one KV device per backend, one registration
  per key, keys 1–16 bytes, no partial-object access.

## Building

```bash
meson setup build -Denable_plugins=SPDK \
    -Dspdk_include_path=/path/to/spdk/install/include \
    -Dspdk_library_path=/path/to/spdk/install/lib
ninja -C build
```

If SPDK is discoverable via `pkg-config` — a system install, or any prefix on
`PKG_CONFIG_PATH` — both `*_path` options may be omitted; the headers and the
library directory are taken from `--cflags-only-I` and `--libs-only-L`. Set them
only when the `.pc` files do not report those paths.

The plugin resolves the SPDK library closure at **configure** time, so re-run
`meson setup --reconfigure` after rebuilding or moving SPDK.

Both plugin models are supported: the default builds a dynamically loaded
`libplugin_SPDK.so`, and `-Dstatic_plugins=SPDK` bakes the plugin into `libnixl`
instead.

## Tests

Two unit tests live under `test/unit/plugins/spdk/`:

- `nixl_spdk_test` — plugin metadata and advertised-parameter checks (no SPDK
  runtime required).
- `nixl_spdk_runtime_test` — a round trip over a `malloc` bdev: write/read/verify,
  handle repost, release-while-in-flight, `OBJ_SEG` capability rejection, and a
  second backend sharing the runtime with the first (including releasing the
  first one while the second keeps running).

```bash
# Tests are only built when buildtype != release, and the runtime test finds the
# plugin through the pluginlist that a debug build generates.
meson setup build --reconfigure -Dbuildtype=debug -Dbuild_tests=true
meson test -C build spdk_plugin_test spdk_plugin_runtime_test
```

The runtime test prints `SKIP` and exits cleanly if the SPDK runtime cannot start.
Set `NIXL_SPDK_TEST_PROG_THREAD=0` to run it in caller-driven mode instead of
with a progress thread.

## Using SPDK with NIXLBench

NIXLBench drives the backend with `--backend=SPDK` over `BLK_SEG`. Storage
transfers are single-process (local DRAM initiator, bdev target); `--op_type`
sets direction.

```bash
./nixlbench --backend=SPDK \
            --spdk_json_config_file=contrib/spdk_malloc_bdev.json \
            --spdk_bdev_name=Malloc0 \
            --op_type=WRITE \
            --start_block_size=4096 --max_block_size=65536 \
            --total_buffer_size=$((1024*1024))
```

- `--spdk_json_config_file` (required): bdev subsystem JSON config file.
- `--spdk_bdev_name` (required): bdev name used for the `BLK_SEG` descriptors.
- `--spdk_bdev_offset`: starting byte offset into the bdev (default `0`).
- `--spdk_msg_mempool_size`: SPDK message-pool size (`0` = default).

A sample malloc config is provided at
`benchmark/nixlbench/contrib/spdk_malloc_bdev.json`. Data buffers use ordinary
page-aligned host memory; `--use_hugepages` is optional, not required.
