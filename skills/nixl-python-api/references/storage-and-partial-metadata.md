# Storage Backends And Partial Metadata

Use this reference when the user asks about POSIX, GDS, GDS_MT, object storage,
file descriptors, object metadata, or partial metadata updates.

## Storage Backend Rule

Do not invent storage descriptor metadata or backend parameters. Storage paths
are backend-, build-, and version-specific.

Before writing a storage recipe, collect:

- Installed NIXL version/source path.
- Selected backend and plugin/backend memory types.
- Backend-specific source/docs or example for the storage path.
- File/object metadata shape required by that backend.
- Trust boundary for paths, object keys, endpoints, credentials, and mount
  points.

If any of these are missing, provide only generic lifecycle advice and state
the exact backend/source evidence still needed.

## Partial Metadata

The fallback snapshot exposes:

- `get_partial_agent_metadata(...)`
- `send_partial_agent_metadata(...)`
- `check_remote_metadata(..., descs=...)`

Partial metadata can be useful when only a subset of descriptors or connection
information needs to be exchanged. Do not claim it is required for ordinary full
metadata transfer.

Before recommending partial metadata:

- Match the API to the user's installed source.
- Identify whether connection info must be included.
- Confirm which descriptor list the remote readiness check should use.
- Confirm invalidation and retry ordering from source or an installed-version
  example.

For topology-specific ordering, invalidation, or retry behavior not verified
against the user's version, state the missing installed-source or runtime
evidence directly.

## Framework-Managed Peers

If a framework owns metadata exchange or peer setup, do not replace it with
direct listener code without inspecting that framework integration source/config.
Classify this as `Framework-managed boundary` in `SKILL.md`.
