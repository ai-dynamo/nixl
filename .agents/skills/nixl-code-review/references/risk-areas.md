# NIXL Code Review Risk Areas

Use this reference after identifying the touched surface in the diff. Facts below
are current-source baselines; match them to the user's version before treating
them as authoritative for a specific review.

## Quick Index

- `Source-Backed Baselines`: pinned references for review claims.
- `Core API And Transfer Lifecycle`: request, metadata, status, and cleanup
  risks.
- `Backend And Plugin Changes`: plugin availability, parameters, and fallback
  risks.
- `Python, FFI, And Downstream Rust Integrations`: binding and lifetime risks.
- `NIXL-Facing Integrations`: Dynamo, vLLM, SGLang, and custom runtime risks.
- `Observability`: error, log, and telemetry review checks.
- `Tests And Benchmarks`: behavior coverage and performance evidence.
- `Diff Scope`: unrelated churn and compatibility-note checks.

## Source-Backed Baselines

- NIXL is described as a point-to-point transfer library for AI inference
  frameworks, with abstractions over CPU/GPU memory and storage through a
  modular plugin architecture. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/nixl.md>.
- NIXL docs describe transfer agents, memory sections, backend interfaces, and
  metadata handlers as the key design entities. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/nixl.md#design>.
- Public docs describe a lifecycle of creating an agent/backend, registering
  memory, exchanging metadata, optionally making connections, creating a
  transfer request, posting it, checking status, and releasing/tearing down.
  Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/nixl.md#example-procedure>.
- Public docs state that one transfer handle can have only one active transfer
  at a time. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/nixl.md#transfer>.
- The backend guide describes backend capability indicators, connection
  management, memory management, metadata management, transfer operations, and
  notification handling. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/BackendGuide.md>.
- The backend guide states that there is no ordering guarantee across transfer
  requests and no NIXL locking mechanism for a specific memory region; the user
  is responsible for avoiding overlapping transfers that target the same memory
  region without application-level coordination.
  Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/BackendGuide.md#transfer-operations>.
- The backend guide says `releaseXferReq` should not block and should provide
  non-blocking behavior to the user even if a backend uses blocking internals.
  Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/BackendGuide.md#transfer-operations>.
- The Dynamo reference review style emphasizes changed-code review, simplicity,
  systems-level thinking, concurrency scrutiny, structured tracing, test value,
  and minimal diff scope. Source:
  <https://github.com/ai-dynamo/dynamo/blob/11e9f37596f22869ea8d164f5b21339f02bd65e6/.agents/skills/graham-code-review/SKILL.md>.

## Core API And Transfer Lifecycle

Review for:

- Memory registration and deregistration paired with the right lifetime.
- Local and remote descriptors staying valid until transfer completion and
  cleanup.
- Metadata export/load/invalidation updated with any agent lifecycle change.
- Transfer handles not reposted while active; if the diff changes handle reuse,
  cite the exact source version or use `TBD-1`.
- Polling/status paths preserving errors and timeout/cancellation behavior.
- Teardown paths releasing backend, memory, metadata, request, and notification
  state without hiding failures.

High-risk smells:

- New code assumes a descriptor is valid because a pointer or address exists.
- A request handle is cached or shared without visible ownership.
- Cleanup happens only on the success path.
- Return/status values are collapsed into a generic success/failure value.

## Backend And Plugin Changes

Review for:

- Capability checks before using local, remote, notification, or memory-type
  features.
- Backend/plugin names, paths, and build options sourced for the user's version
  or marked `TBD-2`.
- Dynamic plugin loading errors propagated with enough context.
- Backend options validated rather than stringly passed through.
- Non-blocking transfer API expectations preserved at the NIXL-facing boundary.
- Abort/release paths that do not stall progress or leak backend request state.

High-risk smells:

- Assuming `UCX`, `GDS`, `POSIX`, or another backend exists without package,
  build, or runtime evidence.
- Treating local storage and remote network backends as interchangeable.
- Adding a plugin fallback that changes semantics without a test.

## Python, FFI, And Downstream Rust Integrations

Review for:

- Object lifetime across language boundaries.
- Exceptions and return codes mapped without losing actionable detail.
- GIL, thread, async, or callback assumptions made explicit and tested.
- Native resources released when Python wrappers or downstream integration
  wrapper objects are dropped.
- Type conversions preserving device IDs, memory types, lengths, offsets, and
  metadata handles.

Use `TBD-1` or `TBD-4` when exact binding semantics are not verified for the
reviewed version.

## NIXL-Facing Integrations

Review for:

- The integration's NIXL version and connector source match the diff under
  review (`TBD-3` when unknown).
- Backend selection, side-channel metadata, and connector configuration remain
  compatible with the target runtime.
- Framework retry, cancellation, shutdown, and error propagation do not mask
  NIXL statuses.
- KV-transfer, disaggregated prefill, storage, or custom runtime paths preserve
  ownership of registered buffers and remote metadata.
- Changes do not assume that behavior from current `main` applies to the user's
  installed Dynamo, vLLM, SGLang, or custom integration version.

## Observability

Review for:

- Error messages include backend/plugin, agent, operation, and source context
  that helps diagnose the failure without leaking secrets.
- Hot-path logs are not too noisy or expensive.
- Structured fields are preferred when the target project already uses them.
- Telemetry/logging controls are source-backed for the reviewed version or
  marked `TBD-5`.
- New logs do not reveal tokens, private hostnames, internal IPs, or proprietary
  model/application details.

## Tests And Benchmarks

Review for:

- Tests exercise the changed NIXL behavior, not just wrapper code.
- Error paths and cleanup paths have coverage when the diff changes lifecycle or
  backend selection.
- GPU, RDMA, storage, or multi-node requirements are explicit and tied to the
  right CI/manual gate (`TBD-6` when unknown).
- Benchmarks include hardware, backend, input shape, before/after numbers, and
  enough repetitions to support a performance claim.

## Diff Scope

Review for:

- Behavioral changes separated from formatting, generated code, or dependency
  churn.
- NIXL-facing integration changes kept separate from unrelated framework
  refactors.
- Documentation and examples updated only when they correspond to changed
  behavior.
- Source pins or compatibility notes updated when a version assumption changes.
