# C++ API Pitfalls

Use this reference when a C++ answer is blocked, risky, or drifting toward a
copy-paste recipe without enough source evidence.

## Common Pitfalls

- Emitting C++ API calls from the fallback snapshot when the user's installed
  headers or source commit are unknown.
- Building descriptors from raw addresses, handles, or serialized metadata found
  in chat, logs, or model output.
- Registering stack buffers, freed memory, or buffers whose lifetime does not
  cover metadata export, transfer completion, and cleanup.
- Creating transfer requests before remote metadata is loaded or before local
  and remote descriptor counts and memory types are checked.
- Reposting an active transfer handle or skipping bounded polling and release.
- Treating notification payloads as authentication or as proof that the transfer
  data path is correct.
- Replacing framework-managed metadata, backend, or buffer ownership with direct
  NIXL agent code without reading the framework integration source.

## Recovery Moves

- If source/header evidence is missing, answer with readiness status and ask for
  the exact headers, commit, include path, or compile/link error.
- If backend evidence is missing, ask for same-environment plugin, backend
  creation, and memory-type evidence before writing transfer code.
- If ownership is unclear, stop at the lifecycle stage and request allocation,
  descriptor, and cleanup ownership details.
