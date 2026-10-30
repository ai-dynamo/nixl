# GPU GET prototype dependency audit

This is a static source/API audit for the local Draft; no build or device run was attempted. The distinction between current upstream `main` and the separate #2234 proposal is explicit below.

## Public revisions inspected

- Upstream `main`: `06dd03bac41d7560809695f0a2e81fc31b855e32`.
- Open Draft #2234 head: `d575619860ae450846d8e188b45d626a644f652d`, a separate PUT/completion proposal. It is not this branch's base and is not modified here.
- NIXL `CONTRIBUTING.md` requires issue/design discussion before significant new functionality, GoogleTest coverage for tests, documentation, and DCO sign-off. The GET issue body is prepared separately and not posted.

## What current upstream main provides

1. `src/api/device/gpu/nixl_device.cuh` exposes PUT, transfer-status, atomic-add, and pointer helpers; it has no transfer-GET operation.
2. `src/api/device/gpu/impl/device_dispatch.cuh` dispatches PUT and status queries but no GET direction.
3. Upstream main does not contain the native GPUNETIO view/compat implementation files introduced by the separate #2234 proposal.

## What is specific to #2234's proposed branch

1. Its GPUNETIO compatibility layer implements PUT and polling, but no supported-profile `compat::get`. The legacy DOCA 3.1 `DUMP_AUTO` profile is explicitly rejected by the public DOCA 3.5+ path.
2. Its new `src/plugins/gpunetio/gpunetio_device_memview.cpp` currently rejects non-`VRAM_SEG` views. This limitation is not a file or behavior in upstream main. The validated Engram case uses a registered pinned-Host/`DRAM_SEG` owner, so the public GET design must settle memory-type support explicitly.
3. The retained GET reference calls private `impl::unwrapViews`, `impl::prepareStatus`, `GpunetioDeviceView`, lane/QP state, and `doca_gpu_dev_verbs_get_thread<DOCA_GPUNETIO_VERBS_DUMP_AUTO,...>`. It is inspection-only and not a current-main or supported-SDK implementation.

## Reproduce the comparison

From a clone with remote `upstream` configured, fetch main and the PR head as read-only refs. Do not checkout or rebase this branch onto the PR head:

```bash
git fetch upstream main
git fetch upstream pull/2234/head:refs/remotes/upstream/pr-2234
git rev-parse upstream/main upstream/pr-2234
```

Check current main for the public API and absence of native backend files:

```bash
git grep -n -E '(^|::)get[[:space:]]*\(' upstream/main -- \
  src/api/device/gpu/nixl_device.cuh src/api/device/gpu/impl/device_dispatch.cuh
git ls-tree -r --name-only upstream/main | rg \
  'src/api/device/gpu/impl/gpunetio|gpunetio_device_memview.cpp'
```

No transfer-GET match or native GPUNETIO-view path is expected in the main tree. Inspect the separate #2234 proposal explicitly for its view restriction and GET absence:

```bash
git show upstream/pr-2234:src/plugins/gpunetio/gpunetio_device_memview.cpp | \
  rg -n 'VRAM_SEG|type !='
git grep -n 'compat::get' upstream/pr-2234 -- \
  src/api/device/gpu/impl/gpunetio
```

The legacy internal dependencies are visible in the committed reference file:

```bash
rg -n 'unwrapViews|prepareStatus|GpunetioDeviceView|DUMP_AUTO|LEGACY_DOCA31' \
  examples/device/gpunetio_get_prototype/legacy_doca31_get_reference.md
```

## Minimum public implementation after SDK support is settled

- Add a direction-aware GET entry point over registered memory-view elements and a dispatcher arm. Preserve the default host READ/WRITE path.
- Add a supported-profile GPUNETIO GET adapter and document submission/completion, terminal error/lane behavior, visibility ordering, and status/buffer/MR lifetime rules.
- Support only the remote memory types the SDK path actually validates. A first VRAM-to-VRAM primitive is separable from the additional remote Host/`DRAM_SEG` support needed by the validated owner case.
- Add a parameterized two-process owner/consumer correctness test for payload/guard bytes, offsets, busy/rejected submissions, completion, errors, visibility, and lifetime. Keep model/application layouts and performance claims out.

No C++ build, SDK installation/change, or GPU run was performed. Current upstream main has no GET implementation; the legacy reference is excluded from the build and is not a passing current-main compile.
