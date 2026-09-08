# GPUNETIO QP-progress GoogleTest harness

`gpunetio_qp_progress_gtest` is a paired, environment-configured integration
executable. Running its paired cases needs
one source GPU and two target GPUs, a numeric target IPv4 address, unique OOB
ports, and a shared directory used only to exchange serialized NIXL metadata.
Control/verification acknowledgements use TCP, outside the measured interval.

The GPUNETIO GTest parent wires this directory. The local `meson.build` builds
the paired executable and registers it with Meson; absent role configuration is
an explicit GoogleTest skip before any CUDA query.

## Required environment

Build with CUDA, DOCA, GoogleTest/GoogleMock and the normal NIXL development
dependencies available. Use the same compiler options for both comparison arms.
The root build disables tests for `buildtype=release`, so use an optimized
`debugoptimized` build with assertions enabled:

```bash
meson setup build --buildtype=debugoptimized -Doptimization=3 -Ddebug=false \
  -Db_ndebug=false -Denable_plugins=GPUNETIO -Dbuild_tests=true \
  -Dbuild_examples=false -Dbuild_docs=false -Dnixl_cuda_arch_list=90 \
  -Dcpp_args=-Wno-error=maybe-uninitialized --wrap-mode=forcefallback
ninja -C build src/core/libnixl.so src/plugins/gpunetio/libplugin_GPUNETIO.so \
  test/gtest/plugins/gpunetio/gpunetio_qp_progress_gtest
export QP_TEST="$PWD/build/test/gtest/plugins/gpunetio/gpunetio_qp_progress_gtest"
export NIXL_PLUGIN_DIR="$PWD/build/src/plugins/gpunetio"
export LD_LIBRARY_PATH="$(find "$PWD/build/src" -name 'lib*.so' -printf '%h\n' \
  | sort -u | paste -sd:):${LD_LIBRARY_PATH:-}"
git rev-parse HEAD
sha256sum "$QP_TEST" "$NIXL_PLUGIN_DIR/libplugin_GPUNETIO.so"
```

Architecture 90 and the GCC warning override describe the tested H20 build;
adapt these to the GPU/compiler in use. Include the DOCA library directory in
`LD_LIBRARY_PATH` if it is not already in the system loader configuration.

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
  "$QP_TEST" --gtest_filter=QpProgress.PerformanceMixed2MiBAnd4KiB

# Terminal/host with one visible source GPU. Use the target's numeric IPv4.
NIXL_QP_PROGRESS_ROLE=source \
  "$QP_TEST" --gtest_filter=QpProgress.PerformanceMixed2MiBAnd4KiB
```

The performance case runs one 20-warmup/100-measured pair. Run three fresh
source/target process pairs externally for independent repeats.
It writes `qp_progress_performance.json` into the coordinate directory. The
JSON reports p50/p99 transfer-window latency, actual payload and marker bytes,
and separate wall time. It does not claim CQ timing or backend-internal timing.
Both endpoints must print `PASSED`, not `SKIPPED`. Keep all three JSON files per
arm. Compare medians of the three per-run metrics, reporting small-transfer and
bulk-transfer tails plus aggregate rate; an isolation win is not automatically
a throughput win. Profiled runs must not enter the performance comparison.

`OutstandingDescriptorsAndAttachedStreamOrder` uses 513 descriptors per
request (512 data plus an epoch marker) with descriptor merging disabled. It
posts seven requests per peer, then releases the alternating delayed/fast A
requests before performing 129 sequential B-peer reuse posts. The delay is a
test-only CUDA kernel enqueued before the attached transfer; it is not a backend
option or a production hook. The target verifies every payload after a
system-acquire of its exact epoch marker, then checks data-coupled notifications
plus one standalone notification per peer. With descriptor merging disabled,
the initial 14 requests occupy 28 of the 32 ring entries (513 descriptors use
two entries each). After those requests are released, the harness performs one
matched 513-descriptor READ into the reused B source buffer and GPU-verifies its
peer-distinct payload and epoch. `SingleQpReadWriteControl` provides the paired
one-active-QP WRITE and READ
control performance path. It re-posts one prepared WRITE and one prepared READ
handle across changing payload epochs, with one active QP at a time, and emits
source API p50/p99 windows separately from the mixed WRITE performance JSON.
Set `NIXL_QP_PROGRESS_CONTROL_BYTES=2097152` to repeat the same control at 2 MiB.
`RemoteDeregisterReturnsBackendError` requires its fresh fault role and checks
bounded `NIXL_ERR_BACKEND` and release after target B memory is deregistered
after source metadata consumption and before its post.
