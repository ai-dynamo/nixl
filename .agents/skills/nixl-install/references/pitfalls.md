# Install Pitfalls

Use this reference when an install diagnosis is drifting toward a premature fix
instead of separating evidence by layer.

## Common Pitfalls

- Treating `pip list` from one interpreter as proof for the framework process
  that failed under another interpreter, container, or pod.
- Recommending reinstall, driver changes, or Kubernetes manifest edits before
  identifying the failing layer.
- Treating missing `nvcc` as a NIXL install failure without CUDA runtime or
  driver evidence.
- Running dynamic plugin probes while `NIXL_PLUGIN_DIR` or plugin paths are
  untrusted, user-writable, temporary, or copied from logs.
- Treating a plugin file on disk as proof that the plugin can load, create a
  backend, and support the required memory type.
- Naming framework connector flags from latest docs without matching the user's
  installed vLLM, Dynamo, or SGLang version.
- Continuing into transfer debugging after install/importability or connector
  readiness has already failed.

## Recovery Moves

- First prove the failing command and the probe command use the same Python,
  container, pod, and framework entrypoint.
- If plugin paths are untrusted, stop at static evidence and classify plugin
  readiness as `TBD-3` or `Blocked`.
- If the framework starts but the first transfer fails, close the install doctor
  with an install/connector readiness status and hand off to the right follow-up
  workflow.
