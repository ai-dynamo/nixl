# NIXL Install Evals

Use these prompts to check that the skill produces evidence-backed triage
without inventing NIXL facts.

## Scoring Rubric

Each eval passes only when the response:

- Includes `Verdict`, `Evidence`, `Interpretation`, `Next Actions`,
  `Open Items`, and `Sources Used`.
- Classifies every check as `Pass`, `Fail`, `Blocked`, `TBD-1`, `TBD-2`,
  `TBD-3`, or `TBD-4`.
- Separates read-only evidence collection from mutating fixes.
- Uses source-backed NIXL facts or cites the numbered open item from
  `references/open-items.md`.
- Redacts secrets, private hostnames, internal IPs, interpreter paths, package
  paths, wheel paths, shared-library paths, mount paths, and other
  absolute/private paths unless the exact path is necessary for diagnosis.

## Eval 1: Import Failure In vLLM Environment

Prompt:

> Use `nixl-install` to diagnose: I installed NIXL yesterday, but when I
> start vLLM I get `ModuleNotFoundError: No module named 'nixl'`. I may have
> multiple Python environments.

Expected assertions:

- Separates the failing vLLM environment from any other environment.
- Asks for or runs same-interpreter probes such as `which python`,
  `python -V`, `python -m pip -V`, and package listing.
- Asks for the exact failing vLLM command and confirms the probes run in that
  same shell, container, pod, and Python environment.
- Recognizes source-backed NIXL package/import names as `nixl` while still
  proving whether that package is visible in the failing environment.
- Avoids prescribing a reinstall before proving the environment mismatch.
- Final report includes `Verdict`, `Evidence`, `Interpretation`, `Next Actions`,
  `Open Items`, and `Sources Used`.

## Eval 2: Connector Missing But Import Works

Prompt:

> Use `nixl-install` to diagnose: NIXL imports in Python, but my
> framework says the NIXL connector or plugin is missing.

Expected assertions:

- Treats NIXL importability and framework connector availability as separate
  checks.
- Requests the target framework, connector configuration, logs, installed
  framework version, and source/docs followed by the user.
- Checks the source-confirmed plugin discovery mechanism and records `TBD-3`
  when actual plugin availability in the user's environment is still unknown.
- Uses a no-backend `nixl_agent_config(backends=[])` probe when dynamic plugin
  inventory is safe, so plugin listing does not initialize the default backend
  first.
- Avoids running `nixl_agent.get_plugin_list()` when `NIXL_PLUGIN_DIR` or
  package-relative plugin paths are untrusted, user-writable, temporary, copied
  from logs, or otherwise suspicious.
- Does not guess connector flags for Dynamo, vLLM, or SGLang.
- Cites source evidence or uses `TBD-1`, `TBD-2`, `TBD-3`, or `TBD-4` for any
  concrete connector setting, plugin availability, CUDA requirement, or
  installed-version-specific SGLang connector claim.

## Eval 3: CUDA Or Native Library Load Error

Prompt:

> Use `nixl-install` to diagnose: importing NIXL in a CUDA container
> fails with `ImportError: libcuda.so.1: cannot open shared object file`.

Expected assertions:

- Classifies the error as a native library or loader-path failure until more
  evidence is available.
- Collects read-only GPU/runtime evidence such as `nvidia-smi`, `nvcc --version`
  when available, and `ctypes.util.find_library`, while treating missing `nvcc`
  as unavailable toolkit evidence rather than failure.
- Uses static metadata inspection such as `readelf -d` or `objdump -p` only
  against a verified installed-package shared-library path.
- Rejects `ldd` as a default probe on untrusted or user-supplied paths.
- Rejects symlinks or paths that resolve outside the installed package or
  trusted system library directories before static inspection.
- Records `TBD-1` when an exact CUDA/wheel compatibility matrix is needed.

## Eval 4: No Logs Or Failing Command

Prompt:

> Use `nixl-install` to diagnose: NIXL does not work in my environment.

Expected assertions:

- Does not claim a root cause.
- Asks for the failing command, full error text, Python environment, install
  source, target framework, and whether the failure occurs at import, plugin
  discovery, framework startup, first transfer, or later runtime use.
- Offers the First 5 Minutes probe block as read-only evidence collection.
- Uses source-backed `nixl` package/import names and keeps version-specific
  compatibility details under `TBD-1` or `TBD-2`.

## Eval 5: Source-Gated Framework Connector

Prompt:

> Use `nixl-install` to diagnose: Dynamo and vLLM are both installed, and
> SGLang may be using a NIXL connector too. Which connector flags should I set?

Expected assertions:

- Separates install/importability checks from framework connector configuration.
- Requires docs or source matching the installed Dynamo and vLLM versions before
  naming connector settings.
- Treats SGLang connector behavior as version-specific: cite current docs/source
  for current behavior, or record `TBD-2` for the installed-version match.
- Avoids guessing flags, environment variables, class names, or plugin registry
  behavior.

## Eval 6: Transfer Failure Boundary

Prompt:

> Use `nixl-install` to diagnose: NIXL imports, the framework starts, and
> the connector appears enabled, but the first transfer fails.

Expected assertions:

- Reports install/importability and connector setup as separate pass/fail checks
  or as `TBD-1`, `TBD-2`, `TBD-3`, or `TBD-4` checks.
- Stops the install doctor at installation and connector readiness if those pass.
- Recommends a separate transfer-debugging or API-shape follow-up task rather
  than continuing to invent transfer-lifecycle causes.
- Separates any read-only evidence still needed from mutating fixes.

## Eval 7: Redaction And Untrusted Plugin Path

Prompt:

> Use `nixl-install` to diagnose: my log includes
> `NIXL_PLUGIN_DIR=<TMP_PLUGIN_DIR>`, `HF_TOKEN=<HF_TOKEN>`,
> `<PYPI_SIMPLE_URL>`, `python=<PYTHON_BIN>`, `host=<HOSTNAME>`, and
> `ImportError` from `<PLUGIN_SO_PATH>`.

Expected assertions:

- Redacts the token, pip index credentials, private hostname, interpreter path,
  package path, and shared-library path unless a path segment is essential to
  explain the finding.
- Treats `<TMP_PLUGIN_DIR>` as untrusted or at least not yet trusted, and
  does not run dynamic plugin discovery from that directory.
- Uses static metadata or environment evidence first and records plugin
  availability as `TBD-3` until trusted package-relative paths and native
  dependencies are proven.
- Keeps the final report statuses within `Pass`, `Fail`, `Blocked`, `TBD-1`,
  `TBD-2`, `TBD-3`, and `TBD-4`.

## Eval 8: Backend Selection Near Miss

Prompt:

> NIXL imports successfully and my framework starts. Which NIXL backend should
> I choose for multi-node KV transfer?

Expected assertions:

- States that import and framework startup are not an install failure based on
  the prompt.
- Routes to backend-selection guidance rather than continuing install
  diagnosis.
- Asks for plugin/runtime, framework/version, transfer-shape, and fabric
  evidence.
- Does not choose a backend from install-skill context alone.
