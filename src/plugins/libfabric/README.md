# NIXL Libfabric Plugin

This plugin provides a high-performance RDMA backend for NIXL using the OpenFabrics Interfaces (OFI) Libfabric library.

## Overview

The Libfabric plugin provides a high-performance RDMA communication backend with the following key capabilities:

- **Multi-Rail RDMA**: Automatic discovery and utilization of multiple network devices for increased bandwidth
- **Accelerator Memory Support**: Zero-copy transfers between accelerator memory (VRAM) and remote systems. Peer-direct RDMA (e.g. GPU Direct RDMA) support is currently required. Accelerator vendors are handled generically through the HMEM vendor table (see [Adding an accelerator vendor](#adding-an-accelerator-vendor)); CUDA, Level Zero and Neuron are supported today.
- **Scalable Connection Management**: Efficient multi-agent connectivity with robust state tracking and automatic reconnection
- **Asynchronous Processing**: Non-blocking RDMA operations with pre-allocated request pools and completion processing
- **Thread-Safe Concurrency**: Background progress threads with lock-free data structures and configurable threading patterns
- **Topology-Aware Optimization**: Hardware-aware GPU-to-EFA and NUMA-to-EFA mapping using hwloc for optimal performance (EFA-specific)

## Dependencies

### Required Dependencies

- **Libfabric**
  - Many systems will have libfabric already installed. If not, custom libfabric installation is available via https://ofiwg.github.io/libfabric/ - Minimum required version: `v1.21.0`
  - For EFA enabled AWS instances, it is recommended to install through AWS EFA installer: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html - Recommend to use the latest version

- **hwloc**
  - hwloc is used to understand the underlying architecture to optimize application performance. Suggested version: 2.10.0 or newer

- **numa**
  - numa (libnuma-dev on Debian/Ubuntu or libnuma-devel on RPM-based systems) is required for supporting DRAM_SEG memory type NUMA-aware rail selection (for imposing NUMA-aware bandwidth limitation). Suggested version: 2.0.18 or newer.

### Optional Accelerator Dependencies

Each accelerator runtime is optional and detected independently; `VRAM_SEG` is advertised if *any*
of them is usable. All three are needed only at build time as headers — the runtime libraries are
`dlopen`'d, so one binary works on hosts with and without each vendor's driver.

| Accelerator | Build dependency | Runtime library | HMEM interface |
|---|---|---|---|
| NVIDIA GPU | CUDA toolkit (`cuda_dep`) | `libcuda.so.1` | `FI_HMEM_CUDA` |
| Intel GPU (Xe / Arc / Data Center GPU / CRI) | `level_zero/ze_api.h` (`libze-dev`, or oneAPI) | `libze_loader.so.1` | `FI_HMEM_ZE` |
| AWS Trainium / Inferentia | none | `libnrt.so.1` | `FI_HMEM_NEURON` |

Meson reports what it found — look for `Found Level Zero headers, libfabric backend will build with
FI_HMEM_ZE support` in the configure output. At startup the plugin logs the accelerators hwloc found
per vendor (`Topology: ... CUDA=0 NEURON=0 ZE=1`) and the runtime it will assume for pointers it
cannot attribute (`System runtime: ZE`).

Notes specific to Intel GPUs:

- Only **discrete** GPUs are used for PCIe-proximity rail selection. Integrated graphics are
  filtered out using the Level Zero driver's own `ZE_DEVICE_PROPERTY_FLAG_INTEGRATED` flag rather
  than a PCI class check, which cannot separate the two (a discrete Battlemage and an iGPU both
  report class `0x0300`) and which wrongly excludes CRI cards, that carry no display class at all.
  Memory on an integrated GPU is still registered correctly; it just does not form a proximity
  group.
- Device memory over `verbs` needs a libfabric build whose verbs provider advertises `FI_HMEM` — see
  the note under [Provider Selection](#provider-selection). `efa` cannot carry `FI_HMEM_ZE` at all
  (`efa_hmem_ifaces[]` has no entry for it).

### Network Hardware Requirements

Validated compatibility with:

- **AWS EFA** (Elastic Fabric Adapter)

Any other Libfabric providers should also work but have not been validated in production environments. Community validation and feedback are highly appreciated!

### AWS Neuron (Trainium) requirements

For `FI_HMEM_NEURON` (VRAM_SEG registration of Neuron device memory) the plugin requires **one EFA network interface per Neuron device** attached to the instance at launch time, so that each Neuron device has a paired EFA NIC for peer-direct RDMA.

Not all Neuron instance sizes meet this requirement. In particular:

| Instance type | Neuron devices | EFA interfaces | `FI_HMEM_NEURON` supported |
|--|--|--|--|
| `trn2.48xlarge` | 16 | 16 (one per Neuron device) | Yes, when launched with all 16 EFA NICs attached |
| `trn2.3xlarge` | 1 | 1 (host-level EFA only) | **No** -- single host EFA cannot be routed to individual Neuron devices |
| `trn3.*` | varies | varies | Match Neuron device count to EFA interface count |

**Launching trn2.48xlarge with all EFA NICs:** the default `aws ec2 run-instances` invocation only attaches a single ENA (non-EFA) interface, even on instance types that support 16 EFA NICs. You must explicitly attach 16 EFA-typed network interfaces at launch time -- one per `NetworkCardIndex` (0-15). The launch spec's `NetworkInterfaces` array should contain 16 entries in this shape (values shown are illustrative; substitute real subnet and security-group IDs):

```json
{
  "DeviceIndex": <0 for the primary interface, 1 for the rest>,
  "NetworkCardIndex": <0..15>,
  "InterfaceType": "efa",
  "SubnetId": "subnet-xxxxxxxxxxxxxxxxx",
  "Groups": ["sg-xxxxxxxxxxxxxxxxx"]
}
```

Or generate the full 16-entry array programmatically with a short Python helper:

```python
import json
nifs = [{"DeviceIndex": 0 if i == 0 else 1,
         "NetworkCardIndex": i,
         "InterfaceType": "efa",
         "SubnetId": "subnet-xxxxxxxxxxxxxxxxx",
         "Groups": ["sg-xxxxxxxxxxxxxxxxx"]} for i in range(16)]
print(json.dumps({"NetworkInterfaces": nifs}, indent=2))
```

Verify on the running instance:

```bash
ls /sys/class/infiniband/ | wc -l          # should equal Neuron device count
lspci | grep -c 'Elastic Fabric'           # same
/opt/amazon/efa/bin/fi_info -p efa | grep -c '^provider:'  # non-zero
```

The plugin emits an `NIXL_WARN` at initialization time if it detects a Neuron accelerator count vs EFA NIC count mismatch, distinguishing the "wrong instance type" case (`trn2.3xlarge`-style, no per-device EFA in hardware) from the "instance launched without EFA NICs" case (missing `InterfaceType: efa` at `run-instances` time).

## Provider Selection

The plugin picks one provider at startup and creates one rail per device of that provider. The
preference order is:

1. `cxi`
2. `efa`
3. `verbs` (InfiniBand / RoCE, reached through the `verbs;ofi_rxm` layered endpoint)
4. `tcp`
5. `sockets`

Only the RDM-capable verbs domains are used: the `-xrc` and `-dgram` endpoint variants of the same
NICs are skipped, as is the `ofi_rxd`-layered stack, which would place a second, slower rail on
hardware the `ofi_rxm` rail already covers.

To force a different choice, restrict what libfabric reports rather than looking for a NIXL-specific
knob — for example `FI_PROVIDER=tcp` or `FI_PROVIDER=^verbs`.

Two notes specific to verbs:

- verbs does not report a PCIe bus address through libfabric (`bus_type` is `FI_BUS_UNKNOWN`), so the
  plugin resolves each NIC's address through hwloc instead, by looking up the verbs domain name as an
  OS device. Topology-aware rail selection therefore works the same way it does for EFA.
- Device memory over verbs needs a libfabric build whose verbs provider advertises `FI_HMEM`. That
  requires either NVIDIA peer-memory support or per-device dmabuf probing, the latter of which
  landed after libfabric v2.6.0. On a build without it the rail comes up without `FI_HMEM` (logged at
  startup) and device buffers are registered as host memory.

## Rail Pairing

A transfer posts on a local rail and must name a peer rail to post *to*. Those used to be chosen by
position — local rail *k* talks to peer rail *k* — which silently assumes both agents enumerated their
rails onto the same fabrics in the same order. Nothing guarantees that. libfabric domain names are
host-local (one host calls a NIC `mlx5_0`, another calls the same model `rocep153s0f0`), and the
cabling need not follow the naming.

When the assumption breaks the failure is ugly: the peer address is *unreachable* rather than invalid,
so `fi_writemsg`/`fi_senddata` returns `-FI_EAGAIN` forever and the transfer hangs with no error.

Rails are now paired by **where they actually are**. Each rail's own endpoint name is already sent to
the peer during connection setup; for a sockaddr-based provider that name contains the NIC's IP, so
each side matches its local rails against the peer's by subnet, using the local netmask
(`LibfabricUtils::sameFabric`). The assignment is greedy and one-to-one, so two local rails never
target the same peer rail. A rail with no match falls back to index order, and a provider whose
address format cannot be read (efa, cxi) keeps index order throughout — those enumerate consistently
across hosts anyway.

Nothing new goes on the wire, and every combination was already addressable: `insertAllAddresses()`
inserts every peer endpoint into every local rail's address vector, so this only changes *which* one a
transfer uses.

Two consequences worth knowing:

- **Control traffic follows the pairing too.** Notifications and handshakes must *arrive* on the
  peer's rail 0, because that is the only rail with the receive callbacks installed. So they are sent
  from whichever local rail is paired with peer rail 0 (`nixlLibfabricConnection::control_rail_`).
  Sending them from local rail 0 regardless was the same bug in a different place.
- **A non-identity pairing is logged at INFO**, since it is otherwise invisible:
  `Rail pairing differs from index order (2/2 matched by fabric): 0->1 1->0`.

Measured on two hosts whose two 200G ConnectX-7 ports are cabled across each other, so index pairing
was wrong for every rail:

| Configuration | Before | After |
|---|---|---|
| DRAM, 2 rails, 4 MB blocks | hangs on `-FI_EAGAIN` | 30.3 GB/s (243 Gbps) |
| DRAM, 1 rail, 4 MB blocks | 20.4 GB/s | 20.0 GB/s (unchanged) |

## Build Instructions

```bash
# Basic build setup with default options
$ meson setup <name_of_build_dir>

# Setup with custom options (example)
$ meson setup <name_of_build_dir> \
    -Dlibfabric_path=/path/to/libfabric

# Build and install
$ cd <name_of_build_dir>
$ ninja && ninja install
```

## Runtime Configuration

Following are the environment variables that control the runtime behavior of the plugin.

### NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG

Normally, DRAM_SEG memory type buffers should not use more bandwidth than the PCIe switches can
sustain, as buffers travel from host (main memory) to EFA device via PCIe topology.

For this reason, the plugin computes the maximum bandwidth limit that would cause the PCIe switches
on each NUMA node **not** to be saturated. This way when DRAM_SEG memory type is used, only a
limited number of rails is selected, such that PCIe congestion is avoided. The rail selection is
made only from the NUMA node of the origin memory buffer. This is because NUMA nodes interconnect
bandwidth is much smaller than the PCIe link, and it is counterproductive to stress the interconnect
for only reduced additional network bandwidth.

In case it is desired though to set a different bandwidth limit (e.g. when computed bandwidth limit
is not suitable on some PCIe topology), the user can override this computed value through the
environment variable NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG.

To summarize:

- NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG is used to configure NUMA-aware rail selection policy for
DRAM_SEG memory type registration
- It controls the bandwidth limit on DRAM_SEG memory type buffers
- It should be specified as decimal Gbps (Gigabits per second), e.g. 100, 200, 400, etc.
- If not specified, then it is computed as the maximum possible bandwidth that would not saturate
the topmost PCIe bridge/switch devices of the NUMA node of the origin buffer
- It can also be passed as a custom parameter during plugin/backend creation (see
nixlAgent::createBackend()), with key "max_bw_per_dram_seg"
- Environment variable override takes precedence over custom parameter configuration

Notes:

- The bandwidth limit is converted to a rail count limit. During memory registration phase of
DRAM_SEG memory type, a subset of rails is selected, such that the bandwidth limit is enforced
- The subset of rails being selected is made sure not to saturate any topmost PCIe switch of the NUMA node
- The subset of rails being selected is limited to the NUMA node of the origin buffer
- The subset of rails being selected each time uses different rails to ensure optimal resource utilization
- Rail selection is thread-safe
- If user override exceeds total topmost PCIe switch capacity, then additional rails are chosen from
the same NUMA node (while causing saturation of one or more topmost PCIe switches)
- If user override exceeds total capacity of EFA devices connected to the NUMA node, then additional
rails are selected from adjacent NUMA nodes, according to NUMA distance (i.e. rails from closer
nodes are selected first), while keeping the same effort to avoid saturating topmost PCIe bridges
- If user override exceeds total capacity of all EFA devices on the machine, then all rails will be
used for DRAM_SEG memory type

### Summary

The following table summarizes briefly the plugin's runtime configuration:

| Name | Effect | Configuration Source | Values | Examples | Notes |
|--|--|--|--|--|--|
| max_bw_per_dram_seg | Controls the bandwidth limit on DRAM_SEG memory type buffers per NUMA node | Backend init param or `NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG` environment variable | integer | 100, 200 | Units are Gbps (Gigabits per second), auto-computed by PCIe topology, normally does not require user override |
| num_threads | Enables a thread pool for parallel descriptor posting in postXfer | Backend init param | integer | 4, 8 | Default 0 keeps the serial posting path |
| split_batch_size | Minimum descriptor count before postXfer uses the posting thread pool | Backend init param | integer | 1024, 4096 | Default 1024; only applies when num_threads is greater than 0 |
| dmabuf_mapping | Selects the CUDA dmabuf mapping type a VRAM_SEG registration exports for an EFA rail | Backend init param or `NIXL_LIBFABRIC_DMABUF_MAPPING` environment variable | auto, pcie, default, off | auto, off | Default auto takes a per-NIC verdict from the PCIe-path topology; pcie and default force the GPU's PCIe aperture (BAR1) or the platform default mapping on every device; off registers by virtual address. Only applies to CUDA device memory on an EFA provider whose fabric carries libfabric API 1.20 or newer |

## API Reference

### Core Classes

- **`nixlLibfabricEngine`** - Main backend engine providing multi-rail RDMA operations with peer-direct RDMA support
- **`nixlLibfabricRailManager`** - Manages multiple network rails with topology-aware selection and striping strategies
- **`nixlLibfabricRail`** - Individual network rail handling libfabric resources and completion processing
- **`nixlLibfabricTopology`** - Hardware topology discovery for optimal accelerator-to-NIC and NUMA-to-NIC mapping
- **`nixlLibfabricBackendH`** - Request handle for tracking multi-request transfer completion with atomic counters
- **`nixlLibfabricConnection`** - Multi-rail connection metadata for remote agents with state management
- **`nfi_hmem.{h,c}`** - The libfabric HMEM shim: everything libfabric knows about attributing a device pointer but does not export. Plain C, no nixl dependencies, written to be donated upstream
- **`libfabric_accel.{h,cpp}`** - nixl's side of the accelerator boundary: hwloc matching, rail-selection policy, and the `nixlAccelScope` RAII guard
- **`nixlLibfabricProviderInfo`** - Table entry describing one core libfabric provider (in `src/utils/libfabric/libfabric_common.cpp`); the single place provider-specific knowledge lives
- **`nixlAccelScope`** - RAII guard that pushes and pops an accelerator context around the operations that need one; a no-op for runtimes with no thread-current state

### Two tables

The plugin has exactly two places where hardware-specific knowledge lives, and both are tables so
that adding hardware is an edit to one entry rather than a hunt through the codebase:

| | Table | Answers |
|---|---|---|
| Accelerators | `nfi_vendors[]` in `nfi_hmem.cpp` | who owns this pointer, which `fi_mr_attr` arm, is this runtime usable, which devices does it have |
| Providers | `knownProviders()` in `libfabric_common.cpp` | which provider to prefer, what hints a rail needs, does it need topology discovery, does it need all-rail progress |

Everything else asks through the table-derived queries and stays hardware-blind.

### Why the accelerator table is a separate C library

`fi_mr_regattr()` requires the caller to declare `fi_mr_attr::iface` and the correct arm of the
`fi_mr_attr::device` union. libfabric knows how to derive both from a pointer -- `ofi_get_hmem_iface()`
in `src/hmem.c` does exactly that, and `prov/hook/hook_hmem` even auto-fills `fi_mr_attr` from it --
but neither is reachable by an application: `include/ofi_hmem.h` is not installed and `libfabric.so`
exports only the `fi_*` API. So every consumer that registers device memory reimplements libfabric's
vendor probing, and this is that reimplementation, factored so it can eventually be deleted.

`nfi_hmem.{h,cpp}` is therefore constrained at its *interface*: a C-compatible surface (POD structs,
plain functions, `extern "C"`) with no dependency beyond libfabric's public headers, `dlopen` and the
vendor SDK headers -- no hwloc, no nixl types, no nixl logging (diagnostics go through a
caller-installed callback). The header compiles as C99 as well as C++20, and there is a test for that.
The implementation is ordinary C++ like the rest of nixl. If libfabric later exports an equivalent,
the shim collapses to a version gate rather than being thrown away.

Three things stay on the nixl side in `libfabric_accel.{h,cpp}`, because libfabric would not want
them: hwloc predicates, rail-selection policy (`groupable`, strict attribution), and C++ ergonomics.

### Where an accelerator context is needed

Not where you would guess, so to be explicit:

- **Registration needs one.** libfabric exports a dmabuf handle from inside `fi_mr_regattr()` for
  providers configured that way, and that export requires a context on the buffer's device.
- **Detection does not.** Unified virtual addressing lets the driver attribute any allocation in the
  process, whatever context is current. UCX draws the line in exactly the same place, for the same
  stated reason (`uct_cuda_copy_md_mem_query`).
- **Posting does**, because a provider may take a host-copy path for a small transfer.

`nixlAccelScope` pushes and **pops**, so a binding never leaks onto an application thread. It binds
the device's *primary* context rather than one observed on a pointer, which is why there is no
"address workaround" mode to discover or fall out of.

## Adding an accelerator vendor

Backend, rail manager and topology code asks vendor questions only through the shim's API and the
policy helpers in `libfabric_accel.h`, so none of them change for a new vendor. Supporting one costs
at most four edits, the first two in `nfi_hmem.cpp` and the rest in `libfabric_accel.cpp`:

1. **One `nfi_vendors[]` entry.** The vendor name, its `fi_hmem_iface`, an `init` hook (leave `NULL`
   when the SDK is absent at build time, which makes the runtime permanently unavailable rather than
   a build break), a `query` hook that attributes a pointer, an optional `device_bdfs` enumerator,
   and optional `scope_push`/`scope_pop` if the runtime keeps thread-current state. Table order is
   the probe order and first hit wins; keep it descending by `fi_hmem_iface` so it matches
   libfabric's own `ofi_get_hmem_iface()` — several providers call that for themselves.
2. **One `fi_mr_attr::device` union arm** in `nfi_hmem_fill_mr_attr()`. Forced by libfabric's ABI:
   the union member is chosen by `iface` at compile time and cannot be written generically.
3. **One branch in `accelIfaceOfHwlocObj()`**, if the hardware is identifiable from hwloc at all.
   Prefer a PCI vendor ID plus class check, which keeps discovery working without the SDK and
   independent of device masking like `CUDA_VISIBLE_DEVICES`. Fall back to `nfi_hmem_device_bdfs()`
   membership only when PCI attributes cannot separate the hardware, as with Intel GPUs.
4. **A `groupable` decision** in `accelIfaceIsGroupable()`, and a line in
   `accelUnclaimedVramIface()` if the vendor's pointer query can miss its own memory, so that on a
   host without CUDA a VRAM pointer no vendor claimed may be its and should fall back to it rather
   than fail. Hosts where CUDA is usable always treat an unclaimed pointer as a caller error.

An SDK that is optional at runtime must be reached by `dlopen()` (as CUDA, Level Zero and Neuron all
are) rather than linked, so a build with the headers still runs on a host without the driver. Only
the headers may be a build-time dependency.

## Adding a provider

Provider-specific behaviour is confined to the table returned by `knownProviders()` in
`src/utils/libfabric/libfabric_common.cpp`. Adding a provider is one entry, and the fields are the
questions the rest of the plugin used to answer with its own `provider == "..."` comparison:

- `name` / `description` — the core provider name (what appears before any `;` in a libfabric
  `prov_name`) and its label in the discovery log line.
- `extra_caps`, `mr_mode`, `mr_key_size` — what a rail asks `fi_getinfo` for.
- `needs_prov_name_hint` — set when the domain name alone does not identify the provider, as for
  verbs, whose domains are also offered by psm3.
- `needs_full_topology` — set for an RDMA provider that should get hwloc discovery and
  accelerator-to-NIC proximity grouping; clear for a host-stack provider.
- `needs_all_rail_progress` — set for a provider layered over a connection-oriented core, where the
  passive side must poll a rail for the peer's connection to be accepted.
- `single_domain` — set when one rail reaches every peer, so extra rails would only add host copies.
- `topology_query_needs_mr_mode` — set only if the provider returns nothing from the PCIe-mapping
  `fi_getinfo` query without its `mr_mode` repeated, as cxi does because of `FI_MR_ENDPOINT`.
- `select_domains` — a hook, needed only if the generic rule (every domain under this core provider,
  de-duplicated, in `fi_getinfo` order) is not enough. `selectVerbsDomains()` is the one example.

Table order is the preference order. A provider with no entry still works: it gets
`defaultProviderInfo()`, the plain-RDMA settings, rather than a special code path.

## Troubleshooting

### Debug Information

Enable debug logging by setting environment variables:

```bash
# Libfabric debug logging
export FI_LOG_LEVEL=debug
export FI_LOG_PROV=efa  # or verbs, tcp, etc.

# NIXL debug logging
export NIXL_LOG_LEVEL=debug
```

### Common Issues

**No network devices detected:**

```bash
# Check available fabric interfaces
fi_info -l

# For checking specific devices (e.g. EFA as an example)
fi_info -p efa
```

**`fi_enable` or `fi_mr_reg` fails with "Cannot allocate memory" on verbs:**

The locked-memory limit is too low. Rails pin their request pools, and `ofi_rxm` sizes its internal
completion queue by `FI_UNIVERSE_SIZE`, so a default `ulimit -l` of a few megabytes is not enough.
Check it with `ulimit -l` and raise it to `unlimited` (`memlock` in `/etc/security/limits.conf`, or
`--ulimit memlock=-1` for a container), as any verbs-based stack requires.

**Transfers hang on a layered provider with `useProgThread = false`:**

**A progress thread is still required for verbs.** Enable it (`nixlAgentConfig::useProgThread`, or
`--enable_pt=1` under nixlbench).

A layered provider such as `verbs;ofi_rxm` establishes its per-peer MSG connection lazily and the
*passive* side must poll its completion queue for that connection to be accepted. Two things are
needed for that to happen, and only one of them is in place:

- *Which* rails get polled — handled. `progressActiveRails()` used to poll only rail 0 plus rails with
  a non-zero active refcount, and a target that has posted nothing has no active rails, so any rail
  above rail 0 was never polled on that side. The provider table now marks such providers
  `needs_all_rail_progress` and every rail is polled for them regardless of refcount.
- *When* progress is called — **not handled.** Without a progress thread the only thing that polls is
  `progressActiveRails()`, reached from `checkXfer()`/`getNotifs()`. Neither is called during
  connection and metadata setup, so nothing drains the CQ in that window and the rxm connection
  never completes.

Measured on two ConnectX-7 RoCE hosts against libfabric 2.2.0rc1: with the progress thread enabled a
two-host DRAM run completes normally (19.7 GB/s at 4 MB blocks). With it disabled both sides reach
"All processes are ready to proceed" and stall there — no EAGAIN spin, simply nothing progressing.
Fixing that properly means driving progress during setup rather than only from the transfer-side
entry points.

Connectionless RDM providers (`efa`, `cxi`, `tcp`) need no accept and are unaffected.

**A second transfer to the same peer fails with "Handshake from peer ... not received after 60s":**

The inbound handshake that tells an agent its own index at the peer is sent only from
`loadRemoteConnInfo()`, and only on the path that creates a new connection -- an existing
connection returns early with "already exists, skipping duplicate loadRemoteConnInfo". So after one
side calls `invalidateRemoteMD()` and then re-loads the peer's metadata, it creates a fresh
connection and waits for a handshake the peer will never re-send, because the peer's own connection
object still exists. The wait times out after `NIXL_LIBFABRIC_HANDSHAKE_TIMEOUT_S` and the transfer
fails with `NIXL_ERR_REMOTE_DISCONNECT`. This is independent of the provider in use. Until it is
fixed, do not invalidate and re-load a peer's metadata within the lifetime of an agent pair.

For additional support, check the NIXL documentation and Libfabric provider-specific guides.
