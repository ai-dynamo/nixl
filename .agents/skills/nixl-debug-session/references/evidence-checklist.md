# Evidence Checklist

Use this checklist after the "First 2 Minutes" probes when a NIXL debug session
needs layer-specific evidence. Collect only what can confirm or falsify the
current hypothesis.

- **Framework connector** (`TBD-2`): installed framework version, minimal
  redacted config excerpt, startup command, connector logs, and source/docs
  matching that version.
- **Backend/plugin configuration** (`TBD-3`): source-backed backend/plugin
  names, minimal allowlisted/redacted environment excerpts, static plugin path
  or config evidence, and load errors. Run dynamic plugin-list probes only after
  importability is proven and plugin paths are trusted; otherwise record
  `TBD-3`.
- **Transfer lifecycle** (`TBD-1`): minimal script or test that reaches the
  failing NIXL call, plus return codes, exceptions, request state, and cleanup
  behavior. Trim reproduction scripts to the failing NIXL path; do not paste
  proprietary serving or model code into the worklog.
- **Hardware/runtime** (`TBD-4`): GPU visibility, driver/runtime evidence,
  device access, and whether the issue reproduces on one node or multiple
  nodes.
- **Container/Kubernetes** (`TBD-4`): image tag plus digest or pulled-on date
  when available, entrypoint, mounted devices, redacted pod spec excerpt,
  relevant resource requests/limits, and runtime logs.
- **Network/storage** (`TBD-4`): node placement, anonymized interface or storage
  path evidence, and source-backed transport expectations.
- **Performance** (`TBD-1`, `TBD-5`): source-backed benchmark or
  micro-benchmark plan, one isolated metric per probe, and baseline numbers only
  when matched to the user's version and environment.
- **Logs/telemetry** (`TBD-5`): source-backed log locations, verbosity controls,
  tracing settings, counters, or telemetry fields for the installed version.
