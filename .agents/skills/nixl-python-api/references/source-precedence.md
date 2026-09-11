# Source Precedence

Use this file when the user asks for readiness-dependent NIXL Python API
examples or edits, when the installed version is unknown, or when an API detail
may be version-specific.

## Source Order

1. User's installed NIXL package, wheel, or source checkout.
2. Version-matched upstream NIXL source/docs for that installed version.
3. Fallback orientation snapshot listed below.

The fallback snapshot is not the source of truth for an unknown installation.
Use it only to orient the investigation and mark version-sensitive behavior
as unresolved pending source evidence until checked against the user's installed
runtime/source.

## Fallback Snapshot

- Repository: <https://github.com/ai-dynamo/nixl>
- Commit: `b458bf0cdc1d21dd7d3130a14a09441109906569`
- Checked source date: 2026-05-21

Use the snapshot as a starting point for source navigation only. Prefer finding
the corresponding files in the user's installed package or version-matched
checkout:

- Python API surface: `src/api/python/_api.py`
- Python bindings: `src/bindings/python/nixl_bindings.cpp`
- Python examples: `examples/python/`

Fallback-scoped claim families in this skill should start from those files:

- `_api.py`: Python agent configuration, plugin helpers, memory descriptors,
  descriptor serialization, metadata, notifications, and transfer wrappers.
- `nixl_bindings.cpp`: native status/exception behavior and Python binding
  exposure.
- `examples/python/`: lifecycle examples and example ordering.

Do not copy line-specific claims from this snapshot into an answer unless the
same behavior is confirmed in the user's installed source.

## Minimal Evidence To Ask For

Prefer the smallest evidence that resolves the uncertainty:

```bash
python -c "import nixl; print(getattr(nixl, '__file__', 'no file')); print(getattr(nixl, '__version__', 'no version'))"
python -m pip show nixl
python -m pip list | grep -E -i '^(nixl|nixl-cu)'
```

If the user has a source checkout, ask for the commit and the matching
`src/api/python/_api.py` file. If they use a framework-managed connector, ask
for the framework source/config that owns the NIXL Python API call path before
rewriting metadata or listener logic.

## Reporting Rule

In every answer, state one of:

- `Source: installed NIXL <version/path/commit>`
- `Source: version-matched upstream <commit/tag>`
- `Source: fallback snapshot only; installed-version evidence unresolved`
- `Source: version unresolved`

Do not present backend parameters, storage metadata, partial metadata ordering,
notification support, retry/cancellation semantics, or exact error behavior as
universal unless the user's installed source proves it.

## Unresolved Facts

Do not maintain a static unresolved-fact ledger in this skill. Resolve unknowns
online from installed source/runtime evidence first, then version-matched
upstream source/docs. If still unresolved, state the gap directly:

```text
Unresolved pending source evidence: <fact>. Needed: <installed source/runtime
probe/docs that would resolve it>.
```
