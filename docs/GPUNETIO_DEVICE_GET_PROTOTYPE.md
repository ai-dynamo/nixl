# GPUNETIO Device GET: prototype contract and current port blockers

**Status: local Draft for design review.** This branch is based on current upstream `main`, not on #2234's fork head. The reference package remains documentation-only; no current-main API or implementation is added. #2234 remains a separate PUT/completion proposal. This document makes the GET contract reviewable without claiming a working public API or current-SDK build.

## Prototype scope

- One GPU `THREAD` caller, one peer, one GET at a time on a shared QP lane.
- Registered source view is remote; destination and any SDK-required scratch view are local.
- Submission returns accepted/in-progress or a synchronous rejection. Only an accepted request can be polled. Polling yields in-progress, terminal success, or a terminal error; terminal state is cached and an error latches the lane.
- The prototype must state the operation that makes destination bytes visible to subsequent local GPU work. The consumer must not inspect data before that guarantee. `cuFlushGPUDirectRDMAWrites` evidence for the PUT receiver path is not automatically a GET visibility proof.
- View handles, registrations/MRs, source/destination buffers, status, and SDK scratch remain alive until accepted work is terminal. Release is not cancellation. A timeout or unresolved error retains/quarantines those objects until safe QP teardown.
- The ordinary host READ/WRITE path stays unchanged and default-on. Native GPU GET remains opt-in. No Engram row layouts, FP8 scales, deduplication, scheduler, or serving policy enter the NIXL core.

The initial portable two-endpoint correctness case may target registered VRAM on both endpoints once a supported GET adapter exists. The validated Engram source owner uses pinned Host/`DRAM_SEG`; the separate #2234 native-view implementation currently admits VRAM only. Current upstream main does not contain that native GPUNETIO view implementation. Host-source support is an explicit design requirement before a later port can reproduce the Engram owner layout.

## Evidence and exact dependency boundary

Current upstream main inspected for this draft is `06dd03bac41d7560809695f0a2e81fc31b855e32`. Its public GPU header/dispatcher expose PUT and completion helpers but no transfer GET. Upstream main does not contain the GPUNETIO native compat/view files added by the separate #2234 proposal. That proposed branch adds native GPU PUT/completion; its new memview builder rejects every segment type except `VRAM_SEG`, and its compat layer has no GET operation. #2234 is linked for context only and is not this branch's base.

The validated legacy GET reference is retained as a fenced C++ block in `examples/device/gpunetio_get_prototype/legacy_doca31_get_reference.md` for inspection only. It is not included by Meson. It is explicitly gated on `NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31` and `DOCA_GPUNETIO_VERBS_DUMP_AUTO`; it reaches `impl::unwrapViews`, `impl::prepareStatus`, `impl::gpunetio::GpunetioDeviceView`, the private lane/QP, and status-payload internals. Those helpers/layouts are not a public GET contract. Current `#2234` intentionally requires the public DOCA 3.5+ profile and rejects the legacy profile; it also has no `compat::get` implementation. No DOCA 3.5 GET symbol/signature or scratch/visibility behavior is inferred here.

## Minimum public patch after the SDK contract is confirmed

1. Agree on the public direction-aware GET call and the supported execution scope using registered memory-view elements and the existing completion/status abstraction.
2. Add a GET dispatcher arm and GPUNETIO implementation with direction/range/peer validation and a documented accepted/rejected/completed/error state machine.
3. Add a supported DOCA profile adapter only after the exact GET primitive, ticket, scratch/dump, and visibility semantics are confirmed from public SDK headers/docs. Do not port `DUMP_AUTO` by guessing.
4. Add view admission for the explicitly supported remote memory types. Add pinned Host/`DRAM_SEG` only in the narrowly justified follow-up needed by a Host-owner use case.
5. Add a two-process owner/consumer correctness test for payload/guards, offsets, completion, rejected/busy requests, errors, visibility and MR/buffer lifetime. Keep default host transfers unchanged.

This Draft can be reviewed while those compatibility questions are resolved. It is not ready to be described as a current-SDK implementation or merged code.

## Reproduce the dependency audit

From a clone with `upstream` configured, fetch both revisions as remote-tracking refs. This does not switch the worktree or make #2234 the branch base:

```bash
git fetch upstream main
git fetch upstream pull/2234/head:refs/remotes/upstream/pr-2234
git rev-parse upstream/main upstream/pr-2234
git grep -n -E '(^|::)get[[:space:]]*\(' upstream/main -- src/api/device/gpu/nixl_device.cuh src/api/device/gpu/impl
git ls-tree -r --name-only upstream/main | rg 'src/api/device/gpu/impl/gpunetio|gpunetio_device_memview.cpp'
git diff --unified=3 upstream/main upstream/pr-2234 -- src/api/device/gpu/nixl_device.cuh src/api/device/gpu/impl/device_dispatch.cuh src/plugins/gpunetio/gpunetio_device_memview.cpp
git show upstream/pr-2234:src/plugins/gpunetio/gpunetio_device_memview.cpp | rg -n 'VRAM_SEG|type !='
rg -n 'unwrapViews|prepareStatus|GpunetioDeviceView|DUMP_AUTO|LEGACY_DOCA31' examples/device/gpunetio_get_prototype/legacy_doca31_get_reference.md
```

The main-only checks should have no transfer-GET match and no native GPUNETIO-view path. The cross-revision diff identifies files added only by #2234; the explicit VRAM-only check is evaluated against that PR ref, never `HEAD` on this branch.

## Build and test status

No C++ build or GPU run was performed for this package. Current upstream main lacks a public GET entry point and GPUNETIO GET adapter; the local host has no CUDA/DOCA toolchain. The source/API audit is in the committed `docs/GPUNETIO_DEVICE_GET_DEPENDENCY_AUDIT.md`. The legacy reference is an inspection artifact, not a passing current-main compile. Historical H20 READ/GET validation is not current-main or DOCA 3.5 validation, and no PUT PR performance is attributed to GET.
