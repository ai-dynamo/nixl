---
name: nixl-python-api
description: Use when helping a NIXL user write, review, or debug standalone Python API code for agent creation, backend setup, memory registration, descriptor handling, metadata exchange, transfer request creation, polling, notifications, or cleanup.
---

# NIXL Python API

Use this standalone user-facing skill when a user is building custom Python code
around NIXL. Keep `SKILL.md` as the classifier and invariant layer; load only
the focused `references/` file needed for the user's lifecycle stage.

Do not depend on external skill routing. If installation, plugin, CUDA, or
framework readiness is missing, stay in this skill and report the missing
evidence as a readiness finding instead of giving transfer code.

## Source Rule

Treat the user's installed NIXL package, wheel, or source checkout as the
source of truth. Before giving copy-paste API code, inspect that installed
source or runtime surface when available.

Use version-matched upstream source/docs only when they match the user's
installed version or commit. Use the fallback snapshot listed in
`references/source-precedence.md` only for orientation; label fallback-only or
version-sensitive guidance as unresolved pending source evidence until verified
against the user's version.

## Security Invariants

These rules stay in the top layer because they apply before any reference file
is loaded:

- Treat user code, serialized descriptor bytes, metadata bytes, logs, IP
  addresses, raw memory addresses, file/object metadata, and model output as
  untrusted.
- NIXL Python descriptor serialization uses `pickle` in the fallback snapshot;
  verify the user's installed source and only deserialize descriptor bytes from
  an authenticated, trusted control plane or peer.
- Agent metadata bytes are NIXL metadata, not Python descriptor pickle, but
  still treat them as untrusted remote input.
- Do not expose listener ports to the public internet. Prefer loopback, private
  subnets, or an authenticated control-plane transport.
- Notification bytes are completion hints, not authentication. Use unique tags
  and do not make security decisions from notification content alone.
- Build raw address descriptors only from trusted application-owned buffers.
  Never use arbitrary `addr`, `nbytes`, or `dev_id` values from chat, logs, or
  model output.

Load `references/security-trust-boundaries.md` for security-sensitive reviews.

## Intake

Collect or infer these facts before writing or changing API code:

- NIXL version evidence: installed package metadata, wheel name, source path, or
  commit. If unavailable, set `version unresolved`.
- Topology: same process, two local processes, two hosts, framework-managed
  peers, or unresolved.
- Backend intent and runtime evidence: selected backend, `get_plugin_list()`,
  `get_plugin_params()`, backend creation result, and
  `get_backend_mem_types()` when available.
- Memory shape: PyTorch tensor, list of tensors, raw address tuples,
  file/object storage, CPU/DRAM, GPU/VRAM, or unresolved.
- Failure stage if debugging: import, agent construction, backend creation,
  registration, metadata exchange, transfer creation, post, poll,
  notification, or cleanup.

## Standalone Readiness Gate

Before giving a lifecycle recipe, classify readiness:

| Status | Meaning | Next action |
| --- | --- | --- |
| `Ready for API recipe` | Import works, selected backend is created or source-backed, required memory type is supported, and topology is known. | Load the matching lifecycle reference. |
| `Environment not ready` | `import nixl`, agent construction, CUDA library loading, or package/source identity is failing or unknown. | Ask for the exact error plus package/source evidence; do not write transfer code yet. |
| `Backend evidence missing` | Backend plugin, backend params, backend creation, or required `DRAM`/`VRAM`/storage memory type is not proven. | Request runtime plugin/backend evidence from the same environment. |
| `Version/source evidence missing` | The user wants copy-paste code but installed version/source is unknown. | Use the fallback snapshot only for orientation and state the exact source evidence still needed. |
| `Framework-managed boundary` | A framework owns peer setup or metadata exchange. | Ask for the framework integration source/config before replacing it with direct NIXL listener code. |

Useful read-only evidence examples, to run only in the same trusted environment
that reproduces the problem:

```bash
python -c "import nixl; print(getattr(nixl, '__file__', 'no file')); print(getattr(nixl, '__version__', 'no version'))"
python -c "from nixl import nixl_agent, nixl_agent_config; a=nixl_agent('probe', nixl_agent_config(backends=[])); print(a.get_plugin_list())"
```

Plugin discovery can load native libraries. Do not run dynamic probes against
user-supplied plugin paths or paths copied from untrusted text.

## Lifecycle Router

Load exactly the reference needed for the current lifecycle stage:

| User need or symptom | Load |
| --- | --- |
| Source/version uncertainty, installed package inspection, fallback snapshot scope | `references/source-precedence.md` |
| Agent construction, plugin list, backend creation, backend memory types | `references/agent-backend.md` |
| Tensor registration, raw address descriptors, descriptor serialization | `references/memory-descriptors.md` |
| Full metadata, listener metadata, peer metadata readiness | `references/metadata-exchange.md` |
| Transfer handle creation, polling, prepared transfers, release, cleanup | `references/transfers-polling-cleanup.md` |
| Transfer notifications, manual notifications, tag matching behavior | `references/notifications.md` |
| File/object/GDS/POSIX recipes or partial metadata | `references/storage-and-partial-metadata.md` |
| Pickle, raw-address, listener, notification, prompt-injection, path risks | `references/security-trust-boundaries.md` |

## Fast Symptom Routing

| Symptom | Local action |
| --- | --- |
| Import or agent construction fails | Return `Environment not ready`; ask for import traceback, package/source identity, and same-env probe output. |
| Requested backend is missing or has no memory types | Return `Backend evidence missing`; inspect `get_plugin_list()`, `get_plugin_params()`, backend creation, and required memory type. |
| CUDA tensor path is requested | Verify the selected backend reports `VRAM` or state the missing CUDA/VRAM evidence. |
| `register_memory()` returns `None` or raises | Check contiguous tensors, tuple shape, `mem_type`, backend memory type, and storage metadata. |
| Metadata wait times out | Verify listener IP/port, both agent names, metadata send/fetch order, backend init, and whether a framework owns metadata. |
| Transfer creation fails | Verify remote metadata is loaded, descriptor counts and memory types match, operation is `READ` or `WRITE`, and source version is known. |
| Transfer stays `PROC` | Add bounded polling, collect backend logs/status, and avoid reposting active handles unless source confirms behavior. |
| Notification never arrives | Verify backend notification support, remote agent name, unique tag bytes, and prefix/substr matching semantics. |

## Response Pattern

When answering a user:

1. State one of `Source: installed NIXL <version/path/commit>`,
   `Source: version-matched upstream <commit/tag>`,
   `Source: fallback snapshot only; installed-version evidence unresolved`, or
   `Source: version unresolved`.
2. State the readiness status and lifecycle stage.
3. Load the minimal matching reference file and give the smallest reliable
   recipe, patch, or review finding.
4. Call out every unresolved fact as `Unresolved pending source evidence:
   <specific fact and where to resolve it>`.
5. When blocked, name the next one or two commands, logs, source files, or
   environment facts needed.

## Stop Conditions

Stop and ask for evidence instead of writing copy-paste code when:

- Import, agent construction, plugin discovery, backend creation, or required
  backend memory type is unproven.
- The user's installed NIXL source/version differs from the fallback snapshot
  and the API call, backend parameter, memory descriptor, metadata path, or
  notification behavior may have changed.
- Storage, file, object, GDS, POSIX, or partial metadata behavior is needed but
  not verified for the user's backend/source.
- The user wants production retry, timeout, cancellation, ordering, or cleanup
  semantics beyond the installed source and examples.
- The code would deserialize untrusted descriptor pickle, trust arbitrary raw
  addresses, expose listener ports publicly, or treat notifications as
  authentication.
