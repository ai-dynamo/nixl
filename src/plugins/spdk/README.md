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

## Temporary vfio-user shared-memory support

The custom env uses FD-backed `MAP_SHARED` allocations for SPDK DMA memory
(hugetlb memfd when available, regular memfd otherwise). FD/offset lookup is
published before registration callbacks and remains valid through unregister.
This adapts the allocator from the local SPDK `spdk-env` branch at `baddf75312`;
it does not import its PCI/IOMMU implementation or change `spdk_vtophys()`.

Use `bdev_type=nvme`, `bdev_trtype=VFIOUSER`, `bdev_name=Vfio`,
`bdev_traddr=/path/to/vfio-directory`, and the target's `bdev_subnqn`.
The first namespace is `Vfion1` for BLK registration. Omit network-specific
`bdev_adrfam` and `bdev_trsvcid`. The directory contains the target's socket;
it is not a PCI address. For the VM's PA-mode configuration, reserve sufficient
2 MiB hugepages for both processes and set `NIXL_SPDK_REQUIRE_HUGEPAGES=1`
before initializing NIXL. This disables the DMA allocator's ordinary-page
fallback and reports allocation failures. Back caller payloads with hugetlb
memfds too, rounding mapping/file lengths to 2 MiB. The target can then use
`--iova-mode=pa`; an IOMMU is not required. Without the opt-in, unprivileged
malloc/TCP configurations retain the ordinary-page fallback.

Caller payloads must already be shareable. Applications use only the existing
`agent.registerMem(DRAM_SEG)` / `agent.deregisterMem()` API; there is no public
SPDK-specific import API. Registration discovers the containing VMA through
`/proc/self/maps`, requires a readable/writable `MAP_SHARED` mapping, and
recovers its backing FD through `/proc/self/map_files/<start>-<end>`. If kernel
permissions prevent that, it duplicates an existing writable FD in the process
whose device/inode match the mapping. It never reopens a pathname from maps.

Keep the original backing FD open through registration for unprivileged use.
Once registered, the backend owns a separate FD until deregistration finishes.
With access to `map_files` (CAP_SYS_ADMIN or CAP_CHECKPOINT_RESTORE in the
initial user namespace on current Linux), already-shared anonymous mappings
can also be recovered even though the application has no FD. This includes
`MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB`; it does not convert private pages.

Each descriptor must fit one VMA and have a 4 KiB-aligned address, length and
file offset. Private mappings (including anonymous hugepages and malloc),
read-only mappings, ranges crossing VMAs and unrecoverable backing objects are
rejected, for all SPDK backend transports. Keep mappings stable and do not
truncate their backing files until all I/O and deregistration complete. No
mapping replacement, staging allocation or data copy is performed.

`spdk_plugin_memory_test` covers offsets, sharing, registration notifications,
late allocations, replay, rejection and descriptor lifetime. The runtime test
accepts `NIXL_SPDK_TEST_VFIO_SOCKET` for explicit block transport testing; its
malloc regression remains the default. KV via vfio-user and nixlbench's shared
payload integration require separate qualification.

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
  Use a writable shared mapping with a discoverable backing FD; alignment alone
  is not sufficient. Private allocations such as `posix_memalign()` are rejected.
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

// Check syscall and NIXL return values in application code. Keep the FD and
// mapping alive until requests are released and DRAM is deregistered.
int fd = memfd_create("kv-payload", MFD_CLOEXEC);
ftruncate(fd, 4096);
void *buf = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

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

### Threading

The agent's `useProgThread` picks one of two models:

- **Progress thread on.** The backend runs its own thread. `postXfer` hands the
  request to it and returns immediately; completions are reaped there and
  `checkXfer` only reads the result. `pthrDelay` throttles the poll (0 busy-polls).
- **Progress thread off.** I/O completion is the caller's job: `postXfer` and
  `checkXfer` poll inline, so a request makes no progress between your calls.

`bdev_nvme` registers its admin-queue poller on the SPDK app thread, and that
poller is what sends keep-alive and drives reconnect and AER handling. If nothing
ran it, an idle backend on an NVMe-oF bdev would lose its controller after the
keep-alive timeout, with no I/O outstanding at all. So a background loop stays
alive in both models: the progress thread wakes on a bounded wait instead of
sleeping until the next submit, and the caller-driven modes run a low-rate
maintenance thread. Both poll at 100 ms, well inside the 10 s default KATO.

The maintenance thread never reaps a bdev I/O, so completion stays where the model
says it is. It polls the app thread, which holds no I/O channels, and it polls this
backend's own SPDK thread only when nothing is in flight, leaving no completion to
take. That second part needs the backend lock, so it happens only under
`NIXL_THREAD_SYNC_STRICT`/`_RW`. Under `NIXL_THREAD_SYNC_NONE` you promised
single-threaded access, so it leaves your SPDK thread alone and services the app
thread only.

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
  handle repost, release-while-in-flight, `OBJ_SEG` capability rejection, an idle
  period spanning several maintenance wakeups followed by more I/O, and a second
  backend sharing the runtime with the first (including releasing the first one
  while the second keeps running).

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
