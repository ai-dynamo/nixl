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

## API Reference

### Core Classes

- **`nixlLibfabricEngine`** - Main backend engine providing multi-rail RDMA operations with GPU Direct support
- **`nixlLibfabricRailManager`** - Manages multiple network rails with topology-aware selection and striping strategies
- **`nixlLibfabricRail`** - Individual network rail handling libfabric resources and completion processing
- **`nixlLibfabricTopology`** - Hardware topology discovery for optimal GPU-to-EFA and NUMA-to-EFA mapping
- **`nixlLibfabricBackendH`** - Request handle for tracking multi-request transfer completion with atomic counters
- **`nixlLibfabricConnection`** - Multi-rail connection metadata for remote agents with state management

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

For additional support, check the NIXL documentation and Libfabric provider-specific guides.
