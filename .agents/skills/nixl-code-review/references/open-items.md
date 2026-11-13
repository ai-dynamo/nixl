# NIXL Code Review Open Items

Use these numbered references when a review finding depends on a fact that is
unknown, not verified for the user's version, or only inferred from the diff.

Current public-source baselines checked on 2026-05-21:

| Project | Public source pin |
| --- | --- |
| NIXL | `main` at `b458bf0cdc1d21dd7d3130a14a09441109906569` |
| Dynamo | `main` at `11e9f37596f22869ea8d164f5b21339f02bd65e6`; earlier public-source check also captured `5bff35311517b0863549de00e1969810f853c6f5` |

## TBD-1: Installed NIXL API And Lifecycle Semantics

Exact behavior for agent creation, backend creation, memory registration,
metadata export/load/invalidation, transfer request creation, posting, polling,
notification, release, error/status mapping, and teardown for the user's NIXL
version.

Needed evidence:

- NIXL commit, release tag, wheel metadata, or container image digest.
- Source/docs matching that version.
- Minimal reproduction or diff context showing the changed call path.

## TBD-2: Backend And Plugin Availability

Exact backend/plugin names, load paths, dynamic vs static plugin behavior,
capabilities, native dependency requirements, and configuration options for the
reviewed environment.

Needed evidence:

- Version-matched NIXL source or build metadata.
- Redacted plugin path/config evidence from the environment under review.
- Test or runtime evidence showing which backend path is expected.

## TBD-3: Framework Connector Boundary

Exact NIXL-facing behavior for Dynamo, vLLM, SGLang, or a custom integration at
the reviewed version.

Needed evidence:

- Framework commit, release, wheel metadata, or image tag.
- Source/docs for the connector path touched by the diff.
- Startup/config excerpts only when needed and redacted.

## TBD-4: Concurrency, Progress, And Notification Guarantees

Exact ordering, locking, progress-thread, cancellation, notification, and
handle-reuse guarantees for the changed path.

Needed evidence:

- NIXL core/backend source for the changed path.
- Tests or code paths showing whether the operation can run concurrently,
  block, abort, or be reposted.

## TBD-5: Observability And Telemetry Controls

Exact logging, telemetry, trace, metric, and exporter behavior for the reviewed
version and runtime.

Needed evidence:

- Version-matched docs/source for log levels, config keys, telemetry exporters,
  and runtime costs.
- Before/after logs or metrics showing user-visible diagnostic value.

## TBD-6: Validation Gate For The Touched Surface

The right test, benchmark, smoke check, or CI gate for the changed NIXL surface.

Needed evidence:

- Existing repo test conventions and CI labels.
- Minimal test that exercises the changed behavior without hiding the NIXL path.
- Hardware/backend requirements for tests that need GPUs, RDMA, GDS, or storage.

## TBD-7: Project-Specific Review Style

Any repo-specific code review convention not covered by this skill or the
linked Dynamo reference.

Needed evidence:

- Target repository `AGENTS.md`, contribution guide, existing reviewer comments,
  or project-specific review guide.
