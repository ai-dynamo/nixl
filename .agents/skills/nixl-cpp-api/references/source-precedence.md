# Source Precedence

Use this file when the user asks for copy-paste NIXL C++ code, when the
installed version is unknown, or when an API detail may be version-specific.

## Source Order

1. User's installed NIXL headers, library, build tree, or source checkout.
2. Version-matched upstream NIXL source/docs for that installed version.
3. Fallback orientation snapshot listed below.

The fallback snapshot is not the source of truth for an unknown installation.
Use it only to orient the investigation and mark version-sensitive behavior
as unresolved pending source evidence until checked against the user's installed
runtime/source.

## Fallback Snapshot

- Repository: <https://github.com/ai-dynamo/nixl>
- Commit: `b293d9bf2d192b321ee24b1988cf1b6b51875331`
- Checked source date: 2026-05-24

Use the snapshot as a starting point for source navigation only. Prefer finding
the corresponding files in the user's installed headers or version-matched
checkout:

- C++ public API: `src/api/cpp/nixl.h`
- C++ public types: `src/api/cpp/nixl_types.h`
- C++ agent config: `src/api/cpp/nixl_params.h`
- C++ descriptors: `src/api/cpp/nixl_descriptors.h`
- C++ examples: `examples/cpp/nixl_example.cpp`,
  `examples/cpp/nixl_etcd_example.cpp`, `examples/cpp/telemetry_reader.cpp`
- Architecture docs: `docs/nixl.md`, `docs/BackendGuide.md`,
  `docs/telemetry.md`

Fallback-scoped claim families in this skill should start from those files:

- `nixl.h`: `nixlAgent`, backend, memory, transfer, notification, metadata,
  memory-view, and cleanup APIs.
- `nixl_types.h`: memory types, operation/status enums, optional arguments,
  handles, and telemetry types.
- `nixl_params.h`: `nixlAgentConfig` fields and defaults.
- `nixl_descriptors.h`: descriptor and descriptor-list classes.
- `examples/cpp/`: example lifecycle ordering and cleanup shape.
- `docs/nixl.md` and `docs/BackendGuide.md`: architecture, metadata,
  backend capability, transfer, notification, and teardown context.

Do not copy line-specific claims from this snapshot into an answer unless the
same behavior is confirmed in the user's installed source.

## Minimal Evidence To Ask For

Prefer the smallest evidence that resolves the uncertainty:

```bash
git -C <nixl-source> rev-parse HEAD
git -C <nixl-source> show HEAD:src/api/cpp/nixl.h | sed -n '1,140p'
find "$NIXL_ROOT" -name nixl.h -print -o -name 'libnixl*' -print
```

If the user has only a binary install, ask for the install prefix, headers,
library path, and build command that currently compiles or fails. If they use a
framework-managed connector, ask for the framework source/config that owns the
NIXL C++ API call path before rewriting metadata, listener, or buffer logic.

## Reporting Rule

In every answer, state one of:

- `Source: installed NIXL <version/path/commit>`
- `Source: version-matched upstream <commit/tag>`
- `Source: fallback snapshot only; installed-version evidence unresolved`
- `Source: version unresolved`

Do not present backend parameters, storage metadata, partial metadata ordering,
memory-view support, notification support, retry/cancellation semantics, exact
error behavior, or cleanup ordering as universal unless the user's installed
source proves it.

## Unresolved Facts

Do not maintain a static unresolved-fact ledger in this skill. Resolve unknowns
online from installed source/runtime evidence first, then version-matched
upstream source/docs. If still unresolved, state the gap directly:

```text
Unresolved pending source evidence: <fact>. Needed: <installed source/runtime
probe/docs that would resolve it>.
```
