# NIXL Libfabric Plugin

This plugin provides a high-performance RDMA backend for NIXL using the OpenFabrics Interfaces (OFI) Libfabric library.

## Overview

The Libfabric plugin provides a high-performance RDMA communication backend with the following key capabilities:

- **Multi-Rail RDMA**: Automatic discovery and utilization of multiple network devices for increased bandwidth
- **GPU Direct Support**: Zero-copy transfers between GPU memory (VRAM) and remote systems with CUDA integration. GDR (GPU Direct RDMA) support is currently required.
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

### Network Hardware Requirements

Validated compatibility with:

- **AWS EFA** (Elastic Fabric Adapter)
- **InfiniBand / RoCE through `verbs`** (NVIDIA ConnectX)

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
knob -- for example `FI_PROVIDER=tcp` or `FI_PROVIDER=^verbs`.

Notes specific to verbs:

- verbs works with or without the progress thread. Without it, the rails are opened with
  `FI_PROGRESS_AUTO`, so `ofi_rxm` accepts peers' connections on its own thread; see *Connection
  setup with a layered provider and `useProgThread = false`* under troubleshooting.
- verbs does not report a PCIe bus address through libfabric (`bus_type` is `FI_BUS_UNKNOWN`), so the
  plugin resolves each NIC's address through hwloc instead, by looking up the verbs domain name as an
  OS device. Topology-aware rail selection therefore works the same way it does for EFA.
- `ofi_rxm` does not report a usable source address for a remote write, which the target needs to
  credit the write to its sender's pending notification. So on verbs each agent hands every peer a
  *write tag* in its connection handshake, and the peer carries it in the immediate data of its
  writes. Tags are 8 bits, which limits an agent to **255 connected peers** on verbs (its
  self-connection included); connecting beyond that fails with `NIXL_ERR_NOT_SUPPORTED`. Each tag
  also carries a generation, so writes still in flight from a peer whose tag was released and
  reassigned are dropped rather than credited to the new holder. The handshake fields are optional,
  so this needs no protocol version change.
- Device memory over verbs needs a libfabric build whose verbs provider advertises `FI_HMEM`. That
  requires either NVIDIA peer-memory support or per-device dmabuf probing, which needs a libfabric
  newer than v2.6.0. On a build without it the rail comes up without `FI_HMEM` (logged at
  startup) and device buffers are registered as host memory.

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

- **`nixlLibfabricEngine`** - Main backend engine providing multi-rail RDMA operations with GPU Direct support
- **`nixlLibfabricRailManager`** - Manages multiple network rails with topology-aware selection and striping strategies
- **`nixlLibfabricRail`** - Individual network rail handling libfabric resources and completion processing
- **`nixlLibfabricTopology`** - Hardware topology discovery for optimal GPU-to-EFA and NUMA-to-EFA mapping
- **`nixlLibfabricBackendH`** - Request handle for tracking multi-request transfer completion with atomic counters
- **`nixlLibfabricConnection`** - Multi-rail connection metadata for remote agents with state management
- **`nixlLibfabricProviderInfo`** - Table entry describing one core libfabric provider (in `src/utils/libfabric/libfabric_common.cpp`); the single place provider-specific settings live

### Adding a provider

Provider-specific settings are confined to the table returned by `knownProviders()` in
`src/utils/libfabric/libfabric_common.cpp`. Adding a provider is one entry:

- `name` / `description` -- the core provider name (what appears before any `;` in a libfabric
  `prov_name`) and its label in the discovery log line.
- `extra_caps`, `mr_mode`, `mr_key_size` -- what a rail asks `fi_getinfo` for.
- `needs_prov_name_hint` -- set when the domain name alone does not identify the provider, as for
  verbs, whose domains are also offered by psm3.
- `needs_full_topology` -- set for an RDMA provider that should get hwloc discovery and
  accelerator-to-NIC proximity grouping; clear for a host-stack provider.
- `needs_all_rail_progress` -- set for a provider layered over a connection-oriented core, where the
  passive side must poll a rail for the peer's connection to be accepted.
- `needs_write_tag` -- set when a remote-write completion carries no usable source address, so
  peers must name themselves in the write's immediate data.
- `single_domain` -- set when one rail reaches every peer, so extra rails would only add host copies.
- `select_domains` -- a hook, needed only if the generic rule (every domain under this core provider,
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

**Connection setup with a layered provider and `useProgThread = false`:**

A layered provider such as `verbs;ofi_rxm` establishes its per-peer MSG connection lazily, and the
*passive* side must progress it for that connection to be accepted. Two things are needed for that:

- *Which* rails get progressed. A target that has posted nothing has no active rails, so polling
  only rail 0 plus rails with outstanding requests would never poll any rail above rail 0 on that
  side. The provider table marks such providers `needs_all_rail_progress`, and every rail is
  polled for them.
- *When* progress happens. Without a progress thread the plugin polls only from
  `checkXfer()`/`getNotifs()`, and neither is called during connection and metadata setup. So for
  these providers the rails are opened with `FI_PROGRESS_AUTO` (and `FI_THREAD_SAFE`, which it
  requires): `ofi_rxm` then runs its own thread per endpoint, asleep in `poll()` on the connection
  event queue, that accepts connections as they arrive. Data completions are still read by the
  plugin. The log says so per rail: `progresses connection setup on its own thread`.

If the provider does not grant `FI_PROGRESS_AUTO`, rail creation fails with an error saying so;
enable the progress thread (`nixlAgentConfig::useProgThread`, or `--enable_pt=1` under nixlbench)
or pick another provider with `FI_PROVIDER`.

Connectionless RDM providers (`efa`, `cxi`, `tcp`) need no accept and are unaffected.

For additional support, check the NIXL documentation and Libfabric provider-specific guides.
