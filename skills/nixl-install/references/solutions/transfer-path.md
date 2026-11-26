# Transfer-path diagnosis

## Investigate if needed

Start with existing workload evidence. Do not modify a production workload
only to add diagnostics; use an authorized reproduction or later run. Use
documentation from the deployed release.

These are branches, not a checklist. Run only the named checks and stop when
evidence supports one fix, rules out NIXL, or completes path verification. An
earlier installation, loading, backend creation, or integration initialization
failure belongs to its installation use case.

1. **Failure or timeout:** use [Check 1](#check-1-did-nixl-serve-the-request)
   and [Check 2](#check-2-where-did-the-request-fail). Add
   [Check 3](#check-3-which-backend-and-path-ran) only when that layer matters.
2. **Slow transfer or path verification:** use
   [Check 1](#check-1-did-nixl-serve-the-request) and
   [Check 3](#check-3-which-backend-and-path-ran).
3. **Fallback, recomputation, or suspicious success:** start with
   [Check 1](#check-1-did-nixl-serve-the-request). Add
   [Check 2](#check-2-where-did-the-request-fail) only when NIXL reports an
   error. Add [Check 3](#check-3-which-backend-and-path-ran) when NIXL served
   the request and backend or transport fallback is suspected, or when an error
   points to that layer.
4. **Incorrect data:** use [Check 1](#check-1-did-nixl-serve-the-request) and
   [Check 5](#check-5-was-the-data-correct).

Use [Check 4](#check-4-what-differs-from-an-isolated-transfer) only when the
branch checks leave competing remedies.

### Evidence and sources

#### Check 1: Did NIXL serve the request?

Inspect integration configuration, logs, source, and existing request, byte,
error, or [telemetry](../../../../fern/docs/pages/user-guide/telemetry.md)
evidence. Request success or unrelated activity does not prove that NIXL
served this request. If it skipped NIXL without a NIXL error, use
[Fix 1](#fix-1-integration-skipped-nixl-or-failed-during-transfer).

#### Check 2: Where did the request fail?

Map the observed status or log to registration, metadata exchange, preparation,
posting, or completion. Use the returned NIXL status or typed Python exception.
For `NIXL_ERR_NOT_FOUND`, use `check_remote_metadata` without descriptors to
detect missing or invalidated agent metadata; when partial metadata is used,
pass the affected descriptors to detect missing or invalidated descriptor
metadata. Then use the error log to separate no common backend from
unregistered buffers.
For registration failures, compare the buffer's memory type with the
initialized backend's supported types and parameters. Inspect allocator, device
exposure, placement, topology, and host prerequisites only when evidence points
there. Use the [architecture guide](../../../../fern/docs/pages/getting-started/architecture.md),
[status reference](../../../../fern/docs/pages/resources/troubleshooting.md),
[backend guide](../../../../fern/docs/pages/user-guide/backends/index.md), and
[Python API](../../../../src/api/python/_api.py). Use
[Fix 3](#fix-3-memory-registration-or-device-access-failed) for registration
or device failures, [Fix 2](#fix-2-wrong-backend-or-lower-level-path) for
backend or peer-path failures, and
[Fix 1](#fix-1-integration-skipped-nixl-or-failed-during-transfer) for metadata
or integration-ordering failures.

#### Check 3: Which backend and path ran?

When available, `query_xfer_backend` or `queryXferBackend` identifies the NIXL
backend; see the [Python API](../../../../src/api/python/_api.py),
[C++ API](../../../../src/api/cpp/nixl.h), and integration source. Availability
or creation does not prove selection. For INFO evidence in an authorized
reproduction or later run, set `NIXL_LOG_LEVEL=INFO` before NIXL loads; the
default `WARN` level does not emit it. When the deployed release provides it,
`Selected backend:` is process evidence; a missing line is not.

Use diagnostics from the backend library loaded by the workload for the
lower-level path. For UCX, use only the
[device and protocol diagnostics](../../../../benchmark/nixlbench/README.md#network-backend-issues)
and installed UCX documentation; do not run NIXLBench from this guide. In an
authorized run, set `NIXL_UCX_PROTO_INFO=y` before process start when the
deployed release supports it; otherwise set `UCX_PROTO_INFO=y`. This identifies
the protocol for the transfer size; `tcp` means TCP, while other tokens require
the installed UCX documentation. `ucx_info -d` shows available devices, not the
selected path. Use
shared [Conflicting UCX copies](../shared/runtime-validation.md#conflicting-ucx-copies)
only when evidence points there. For LIBFABRIC, use its
[plugin guide](../../../../src/plugins/libfabric/README.md). When emitted,
`Created ... rails using provider=` at NIXL `INFO` identifies the initialized
provider; `fi_info` only advertises available providers.

Without selection evidence, leave the backend or path unverified. Device
counters support only device or link attribution when unrelated traffic is
excluded. Use [Fix 2](#fix-2-wrong-backend-or-lower-level-path) when evidence
identifies the wrong layer.

#### Check 4: What differs from an isolated transfer?

Prefer an existing comparable probe. Otherwise run an authorized,
non-interfering, already-working [Python example](../../../../examples/python/README.md)
in a disposable process with a bounded payload and timeout. Skip it if it
cannot match the workload's peers, backend, memory types, direction, and
devices. Do not install, build, edit, or debug it. A passing test proves only
its configuration; use
[Fix 4](#fix-4-workload-and-isolated-transfer-differ) only for a supported
difference.

#### Check 5: Was the data correct?

Compare the affected request with a known-correct result when available.
Completion and byte counts do not prove payload correctness. After transfer
completion, check descriptor offsets, lengths, direction, buffer
synchronization, notifications, overlapping writes, and remote metadata.
Attribute transport corruption only when a matched comparison reproduces it on
the same path. Use
[Fix 5](#fix-5-transfer-data-is-wrong) only when evidence identifies the faulty
layer.

## Fix

Apply only the smallest remedy supported by the evidence. Record current
configuration, environment, and deployment values before an authorized
change. Host, driver, kernel, network, and privileged changes require their
owner's authorization.

### Fix 1: Integration skipped NIXL or failed during transfer

If no NIXL error occurred, correct only the connector setting, peer role,
address, timeout, fallback, or metadata exchange identified by the deployed
integration. Refresh or reload peer metadata only when the expected agent is
missing or stale. Correct descriptor, ordering, or request-state handling only
for the matching status or typed exception. If the release exposes no
supported control, report an integration limitation instead of patching its
source.

### Fix 2: Wrong backend or lower-level path

Change backend selection only after backend evidence, and change provider,
transport, device, or link configuration only after lower-level evidence. Use
the installed API or integration's `backends` selection for the request, or
initialize only the intended backend, consistently on every peer. If neither
control is exposed, report an integration limitation. Use the
[Python API](../../../../src/api/python/_api.py), selected
[backend documentation](../../../../fern/docs/pages/user-guide/backends/index.md),
and installed backend-library documentation. Re-check both layers in the real
workload.

### Fix 3: Memory registration or device access failed

Align allocation and registration with the initialized backend's documented
memory types and parameters. Correct container or device exposure, placement,
topology, or host prerequisites only when evidence identifies that layer.
Because multi-backend registration can succeed when only one backend accepts a
buffer, restrict registration to the intended backend and compare the memory
type with `get_backend_mem_types`. For UCX, `VRAM memory is detected as host by
UCX` indicates that the loaded UCX lacks accelerator-memory support; hand that
library mismatch to the matching installation use case.

### Fix 4: Workload and isolated transfer differ

Align only the supported difference in libraries, versions, environment,
allocator, devices, addressing, or peer configuration. Use release-matched
documentation; do not guess a compatibility matrix or copy another image's
recipe.

### Fix 5: Transfer data is wrong

Correct only the descriptor, synchronization, notification, overlap, or
metadata handling identified by [Check 5](#check-5-was-the-data-correct).
Change the backend or transport only when Check 5 attributes the fault there.

If the real workload uses the intended path correctly but remains slow, this
reference does not prescribe general performance tuning without bottleneck
evidence.

## Verify if possible

After a fix, rerun the failing check. Verify only relevant or changed layers.
Leave unavailable evidence unverified; do not install or build solely to
verify. Any new transfer or registration check must be authorized, bounded,
and timed out.

1. Request-correlated integration evidence shows NIXL use without fallback or
   recomputation. When metadata matters, `check_remote_metadata` succeeds for
   the expected peers. Telemetry is supporting evidence only when enabled.
2. When path selection matters, `query_xfer_backend` or `queryXferBackend` on
   the real transfer identifies the intended backend when available. Otherwise
   require equivalent workload-process selection evidence, such as an emitted
   selection line or one initialized backend on every peer with confirmed NIXL
   execution. Leave selection unverified when none is observable. Diagnostics
   from the loaded backend library identify the provider, transport, and
   device.
3. When memory or device access matters, registration succeeds specifically
   for the intended backend, memory type, and devices; success on another
   backend is insufficient.
4. When completion or correctness matters, transfer state reaches `DONE` or
   `NIXL_SUCCESS` before the payload is compared with a known-correct result.
   For performance, compare transfer-scoped bytes and duration with a stated
   expectation; an isolated test is supporting evidence.

Each check proves only its own layer. Request success, NIXL execution, backend
selection, lower-level path, correctness, and performance require separate
evidence.
