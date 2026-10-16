# GPUNETIO QP-progress manual replay

`qp_progress_replay` is a paired, environment-configured integration
executable. Running its paired cases needs
one source GPU and two target GPUs, a numeric target IPv4 address, unique OOB
ports, and a shared directory used only to exchange serialized NIXL metadata.
Control/verification acknowledgements use TCP, outside the measured interval.

This optional tool uses GoogleTest assertions but is built and run manually.
It is not registered with the project's GTest/Meson suite and adds no default
CI workload. Absent role configuration is an explicit skip before any CUDA query.

## Release-build component reproduction

Current component measurements use release core/plugin builds and one common
`-O3` harness for both plugin arms. After installing the release build to `PREFIX`:

```bash
: "${PREFIX:?Set the matching NIXL installation prefix}"
nvcc --std=c++20 -O3 -arch=sm_90 --cudart shared \
  -I"$PREFIX/include" contrib/gpunetio_qp_progress/qp_progress_replay.cu \
  -L"$PREFIX/lib/x86_64-linux-gnu" -L"$PREFIX/lib" \
  -lnixl -lnixl_build -lgtest -Xcompiler=-pthread -o qp_progress_replay
export QP_TEST="$PWD/qp_progress_replay"
export CUDA_MODULE_LOADING=EAGER CUDA_DEVICE_MAX_CONNECTIONS=8
```

For the longer 1200-warmup/1000-measured qualification, add
`-DNIXL_QP_PROGRESS_WARMUP=1200 -DNIXL_QP_PROGRESS_MEASURED=1000` to that
same `nvcc` command and use the resulting single executable for both arms.
Without these flags, the defaults remain 20/100. Record CPU affinity as well
as GPU placement; if pinning workers, use the same distinct physical CPU cores
for both plugin arms, and keep bound/unbound results separate.

Keep that executable and the core/DOCA library search path unchanged when
switching `NIXL_PLUGIN_DIR`. Use five fresh pairs per arm for the mixed case and
each 4 KiB/2 MiB single-active-QP control, alternating arm order. Three distinct
GPUs on one host suffice (one source, two targets), with distinct OOB ports.
Avoid two workers time-sharing one GPU. Record compiler/SDK, commit and binary
hashes. These are source API completion windows, not NIC or serving measurements.

## Required environment

Build with CUDA, DOCA, GoogleTest and the normal NIXL development dependencies
available. Use the manual compile command above with a matching release NIXL
installation; it does not require `build_tests=true`. Architecture90 describes
the tested H20 build; adapt it to the GPU/compiler in use. Include the NIXL and
DOCA library directories in `LD_LIBRARY_PATH` if they are not already in the
system loader configuration. Record commit and executable/plugin hashes.

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

Payload generation mixes the full epoch/source seed and byte offset, so both
payload bytes and markers change between iterations. Generation and validation
remain outside the measured post-to-completion windows.

```bash
# Terminal/host with two visible target GPUs.
NIXL_QP_PROGRESS_ROLE=target \
  "$QP_TEST" --gtest_filter=QpProgress.PerformanceMixed2MiBAnd4KiB

# Terminal/host with one visible source GPU. Use the target's numeric IPv4.
NIXL_QP_PROGRESS_ROLE=source \
  "$QP_TEST" --gtest_filter=QpProgress.PerformanceMixed2MiBAnd4KiB
```

The performance case runs one 20-warmup/100-measured pair. Run five fresh
source/target process pairs externally for independent repeats.
It writes `qp_progress_performance.json` into the coordinate directory. The
JSON reports p50/p99 transfer-window latency, actual payload and marker bytes,
and separate wall time. It does not claim CQ timing or backend-internal timing.
Both endpoints must print `PASSED`, not `SKIPPED`. Keep all five JSON files per
arm. Compare medians of the five per-run metrics, reporting small-transfer and
bulk-transfer tails plus aggregate rate; an isolation win is not automatically
a throughput win. Profiled runs must not enter the performance comparison.

`OutstandingDescriptorsAndAttachedStreamOrder` uses 513 descriptors per
request (512 data plus an epoch marker). Adjacent rows are swapped in the
descriptor order on both ends, preserving the full payload while preventing
backend READ coalescing or WRITE pairing from removing multi-ring pressure. It
posts seven requests per peer, then releases the alternating delayed/fast A
requests before performing 129 sequential B-peer reuse posts. The delay is a
test-only CUDA kernel enqueued before the attached transfer; it is not a backend
option or a production hook. The target verifies every payload after a
system-acquire of its exact epoch marker, then checks data-coupled notifications
plus one standalone notification per peer. With descriptor merging disabled,
the initial 14 requests occupy 28 of the 32 ring entries (513 descriptors use
two entries each). After those requests are released, the harness performs one
matched 513-descriptor READ into the reused B source buffer and GPU-verifies its
peer-distinct payload and epoch on the attached stream before host status polling.
A timing-disabled event also requires first B completion while the deliberately
delayed A posting is still pending. If A is no longer delayed before B submission,
the fixture explicitly skips and must not be counted as a pass.
`AttachedReadOrdersFollowingConsumer` separately checks a one-ring attached READ
consumer before host polling; set `NIXL_QP_PROGRESS_CONTROL_BYTES=2097152` for a
2 MiB payload. Its verification is not part of a performance sample.
`SingleQpReadWriteControl` provides the paired
one-active-QP WRITE and READ
control performance path. It re-posts one prepared WRITE and one prepared READ
handle across changing payload epochs, with one active QP at a time, and emits
source API p50/p99 windows separately from the mixed WRITE performance JSON.
Set `NIXL_QP_PROGRESS_CONTROL_BYTES=2097152` to repeat the same control at 2 MiB.
`RemoteDeregisterReturnsBackendError` requires its fresh fault role and checks
bounded `NIXL_ERR_BACKEND` and release after target B memory is deregistered
after source metadata consumption and before its post.

`CpuFatalLatchRejectsQueuedTransfer` and `CpuFatalLatchRejectsQueuedNotification`
use fresh normal `source`/`target` roles. A deliberate CUDA configuration error
rejects B before enqueue while A remains behind a test-only stream delay.
Pending cancellation must return `NIXL_ERR_REPOST_ACTIVE` without losing A's
handle; after GPU error completion, release succeeds. Target sentinel payloads
and epochs must remain unchanged. These are CPU pre-enqueue failure tests, not
arbitrary device-fault injection. Their expected error logs are not benchmark
failures when the assertions and both process exits pass.
