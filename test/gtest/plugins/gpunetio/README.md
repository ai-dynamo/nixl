# GPUNETIO QP-progress GoogleTest harness

`gpunetio_qp_progress_gtest` is a paired, environment-configured integration
executable. It is intentionally not part of the default test suite: it needs
one source GPU and two target GPUs, a numeric target IPv4 address, unique OOB
ports, and a shared directory used only to exchange serialized NIXL metadata.
Control/verification acknowledgements use TCP, outside the measured interval.

The GPUNETIO GTest parent wires this directory. The local `meson.build` builds
the paired executable and registers it with CTest; absent role configuration is
an explicit GoogleTest skip before any CUDA query.

## Required environment

Both processes need `NIXL_PLUGIN_DIR` and `LD_LIBRARY_PATH` pointing to the
same feature-tree build. Set the following values without placing site-specific
values in this repository:

| Variable | Source | Target | Meaning |
| --- | --- | --- | --- |
| `NIXL_QP_PROGRESS_ROLE` | `source` | `target` or `target-fault` | Process role |
| `NIXL_QP_PROGRESS_COORD_DIR` | same path | same path | Fresh shared metadata directory |
| `NIXL_QP_PROGRESS_TARGET_IPV4` | required | unset | Numeric IPv4 of the target process |
| `NIXL_QP_PROGRESS_CONTROL_PORT` | same | same | TCP control port |
| `NIXL_QP_PROGRESS_SOURCE_OOB_PORT` | same | same | Source agent listener port |
| `NIXL_QP_PROGRESS_TARGET_A_OOB_PORT` | same | same | Target GPU 0 listener port |
| `NIXL_QP_PROGRESS_TARGET_B_OOB_PORT` | same | same | Target GPU 1 listener port |
| `NIXL_QP_PROGRESS_NETWORK_DEVICE` | optional | optional | GPUNETIO network device |
| `NIXL_QP_PROGRESS_OOB_INTERFACE` | optional | optional | OOB interface |
| `NIXL_QP_PROGRESS_GID_INDEX` | optional | optional | GID index |
| `NIXL_QP_PROGRESS_CONTROL_BYTES` | optional | optional | Single-active-QP control size; defaults to 4096 |

Run target first, then source with the same `--gtest_filter`. Use a fresh
process for each fault trial (`target-fault` plus `source` filtered to the
fault case). The harness skips if its role, coordinate directory, GPU count,
or source target IPv4 is unavailable.

```bash
# Terminal/host with two visible target GPUs.
NIXL_QP_PROGRESS_ROLE=target \
  gpunetio_qp_progress_gtest --gtest_filter=QpProgress.PerformanceMixed2MiBAnd4KiB

# Terminal/host with one visible source GPU. Use the target's numeric IPv4.
NIXL_QP_PROGRESS_ROLE=source NIXL_QP_PROGRESS_TARGET_IPV4=<target-ipv4> \
  gpunetio_qp_progress_gtest --gtest_filter=QpProgress.PerformanceMixed2MiBAnd4KiB
```

The performance case runs one 20-warmup/100-measured pair. Run three fresh
source/target process pairs externally for independent repeats.
It writes `qp_progress_performance.json` into the coordinate directory. The
JSON reports p50/p99 transfer-window latency, actual payload and marker bytes,
and separate wall time. It does not claim CQ timing or backend-internal timing.

`OutstandingDescriptorsAndAttachedStreamOrder` uses 513 descriptors per
request (512 data plus an epoch marker) with descriptor merging disabled. It
holds one delayed source-peer request while completing and releasing 129
fast-peer requests, forcing live 32-slot ring reuse without exceeding capacity.
The delay is a test-only CUDA kernel enqueued before the attached transfer; it
is not a backend option or a production hook. The target verifies every fast
payload and the delayed payload after a system-acquire of each exact epoch
marker, then checks data-coupled notifications plus one standalone notification
per peer. `SingleQpReadWriteControl` provides the paired one-QP WRITE and READ
control performance path. It re-posts one prepared WRITE and one prepared READ
handle across changing payload epochs, with one active QP at a time, and emits
source API p50/p99 windows separately from the mixed WRITE performance JSON.
Set `NIXL_QP_PROGRESS_CONTROL_BYTES=2097152` to repeat the same control at 2 MiB.
`RemoteDeregisterReturnsBackendError` requires its fresh fault role and checks
bounded `NIXL_ERR_BACKEND` and release after target B memory is deregistered
after source metadata consumption and before its post.
