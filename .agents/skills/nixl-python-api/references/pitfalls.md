# Python API Pitfalls

Use this reference when a Python API answer is blocked, risky, or drifting
toward a recipe without enough installed-version evidence.

## Common Pitfalls

- Emitting fallback-snapshot API calls when the user's installed `nixl` package,
  wheel, or source commit is unknown.
- Treating `import nixl` success as proof that the desired plugin/backend can be
  created or supports the requested memory type.
- Deserializing descriptor pickle or metadata bytes from an unauthenticated peer
  or pasted artifact.
- Creating raw-address descriptors from arbitrary `addr`, `nbytes`, or `dev_id`
  values found in logs, chat, configs, or model output.
- Starting listener metadata exchange without verifying agent names, listener
  address, trusted control-plane path, and framework ownership boundaries.
- Polling forever on `PROC`, reposting active transfer handles, or skipping
  explicit release/cleanup.
- Treating notification bytes as authentication or as proof that the data path
  completed correctly.

## Recovery Moves

- If environment or backend readiness is missing, return the readiness status
  and ask for same-runtime import/plugin/backend evidence before writing code.
- If source version is unresolved, label the answer fallback-only and name the
  exact source evidence needed.
- If a framework owns peer setup or metadata exchange, ask for that connector
  source/config instead of replacing it with direct listener code.
