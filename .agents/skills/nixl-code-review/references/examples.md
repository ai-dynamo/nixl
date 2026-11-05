# NIXL Code Review Finding Examples

These are illustrative review-comment shapes, not findings in the current repo.
Replace paths, line numbers, and suggested fixes with evidence from the actual
diff. Do not report an example unless the reviewed change really exhibits the
problem.

## Transfer Handle Reuse

```markdown
- [Blocker] src/.../transfer.cc:123 - This reposts the same transfer handle
  before the previous post reaches a terminal state. NIXL's public docs say one
  transfer handle can have only one active transfer at a time, so this can
  corrupt or race the buffer state. Evidence:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/nixl.md#transfer>.
  Suggested direction: gate repost on completion or allocate independent request
  handles for concurrent transfers.
```

## Backend Assumption Without Evidence

```markdown
- [Major] src/.../backend_select.py:88 - The new path assumes `UCX` is available
  and silently falls back to a different backend when creation fails. Backend
  availability is build/runtime specific (`TBD-2`), and silent fallback changes
  transfer semantics. Suggested direction: surface the backend creation error
  with NIXL version, requested backend, and plugin/config evidence, then add a
  test for the missing-backend branch.
```

## Blocking Release Path

```markdown
- [Major] src/plugins/.../backend.cpp:211 - `releaseReqH` waits for completion
  before returning on the abort path. The backend guide says release should
  present non-blocking behavior to the user and can return error while cleanup
  continues. Evidence:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/BackendGuide.md#transfer-operations>.
  Suggested direction: make release/abort asynchronous at the API boundary and
  keep polling/error state explicit.
```

## Metadata Invalidated Only Locally

```markdown
- [Major] lib/.../agent_lifecycle.rs:144 - Removing an agent invalidates the
  local cache but does not propagate or load the remote metadata invalidation
  path touched by this PR. NIXL docs describe remote metadata invalidation as
  the mechanism that disconnects backends and purges cached remote values.
  Evidence:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/nixl.md#addingremoving-agents-dynamic-scaling>.
  Suggested direction: either call the version-matched invalidation API or
  document why this integration owns invalidation elsewhere.
```

## Observability Regression

```markdown
- [Minor] src/.../plugin_loader.cpp:57 - This replaces a plugin-load error with
  a generic "backend unavailable" message. Reviewers need the requested backend,
  plugin path/config source, and original loader error to diagnose deployment
  failures. Suggested direction: preserve the loader error and redact only
  sensitive path segments if needed.
```

## Test Does Not Exercise NIXL

```markdown
- [Major] tests/.../test_connector.py:42 - The new test mocks the connector
  method that performs registration and transfer setup, so it would pass even if
  this PR breaks descriptor construction. Suggested direction: add a minimal
  integration or fake-backend test that reaches the changed descriptor path, or
  mark the required NIXL/hardware gate as `TBD-6` if unavailable.
```
