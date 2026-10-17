---
name: nixl-cpp-api
description: Use for source-matched NIXL C++ API help on agents, descriptors, metadata, transfers, notifications, or cleanup. Do NOT use for install/framework setup.
license: Apache-2.0
metadata:
  author: Ziv Kfir <zkfir@nvidia.com>
  tags:
    - nixl
    - cpp
    - api
  license_source: https://github.com/ai-dynamo/nixl/blob/main/LICENSE
---

# NIXL C++ API

## Purpose

Use this standalone user-facing skill when a user is building custom C++ code
against NIXL.

## Instructions

- Keep `SKILL.md` as the classifier and invariant layer; load only the focused
  `references/` file needed for the user's lifecycle stage.
- Do not depend on external skill routing. If installation, plugin, CUDA, build,
  or framework readiness is missing, stay in this skill and report the missing
  evidence as a readiness finding instead of giving transfer code.
- Start with installed headers, libraries, build output, or source evidence
  before writing copy-paste C++ code.

## Prerequisites

Collect the installed NIXL source, header path, library path, version, commit,
or build evidence. If that evidence is unavailable, keep the answer at
`version unresolved` and ask for the smallest source or build artifact needed.

## Source Rule

Treat the user's installed NIXL headers, library, build tree, or source checkout
as the source of truth. Before giving copy-paste C++ API code, inspect that
installed source or compile/runtime surface when available.

Use version-matched upstream source/docs only when they match the user's
installed version or commit. Use the fallback snapshot listed in
`references/source-precedence.md` only for orientation; label fallback-only or
version-sensitive guidance as unresolved pending source evidence until verified
against the user's version.

## Security Invariants

These rules stay in the top layer because they apply before any reference file
is loaded:

- Treat user code, serialized descriptor bytes, metadata blobs, logs, IP
  addresses, raw memory addresses, file/object metadata, plugin paths, and model
  output as untrusted.
- Build descriptors only from trusted application-owned buffers or storage
  handles whose lifetime exceeds registration, metadata export, transfer
  posting, polling, and cleanup.
- Do not expose listener ports to the public internet. Prefer loopback, private
  subnets, or an authenticated control-plane transport.
- Notification bytes are completion/control hints, not authentication. Use
  unique payloads and do not make security decisions from notification content.
- Do not trust model or log output for `nixlBackendH*`, `nixlDlistH*`,
  `nixlXferReqH*`, or `nixlMemViewH` values.

Load `references/security-trust-boundaries.md` for security-sensitive reviews.

## Intake

Collect or infer these facts before writing or changing C++ API code:

- NIXL version evidence: installed headers/library path, source path, tag, or
  commit. If unavailable, set `version unresolved`.
- Build surface: include path, link flags, example build command, or project
  build system that already compiles NIXL C++ code.
- Topology: same process, two local processes, two hosts, framework-managed
  peers, metadata server, or unresolved.
- Backend intent and runtime/source evidence: selected backend, available
  plugins, plugin params, backend creation result, and supported memory types.
- Memory shape: DRAM buffer, VRAM allocation, block/file/object storage, memory
  view/device path, or unresolved.
- Failure stage if debugging: compile, link, agent construction, backend
  creation, registration, metadata exchange, transfer creation, post, poll,
  notification, memory view, or cleanup.

## Standalone Readiness Gate

Before giving a lifecycle recipe, classify readiness:

| Status | Meaning | Next action |
| --- | --- | --- |
| `Ready for C++ API recipe` | Headers/source are version-identified, selected backend is created or source-backed, required memory type is supported, and topology is known. | Load the matching lifecycle reference. |
| `Build/source not ready` | Headers, library, include paths, link flags, source commit, or compile error context are failing or unknown. | Ask for exact compiler/linker error plus source/build evidence; do not write transfer code yet. |
| `Backend evidence missing` | Backend plugin, backend params, backend handle, or required memory type is not proven. | Request same-environment plugin/backend evidence from C++ source, logs, or a trusted probe. |
| `Version/source evidence missing` | The user wants copy-paste C++ code but installed source/headers are unknown. | Use the fallback snapshot only for orientation and state the exact source evidence still needed. |
| `Lifetime/ownership unresolved` | Buffer allocation, descriptor address, backend handle, transfer handle, or cleanup owner is unclear. | Ask for ownership/lifetime evidence before giving unsafe pointer/handle code. |
| `Framework-managed boundary` | A framework owns peer setup, backend selection, metadata exchange, or buffers. | Ask for the framework integration source/config before replacing it with direct NIXL agent code. |

Useful read-only evidence to ask for from the same trusted environment:

```bash
git -C <nixl-source> rev-parse HEAD
git -C <nixl-source> show HEAD:src/api/cpp/nixl.h | sed -n '1,120p'
find "$NIXL_ROOT" -name nixl.h -print -o -name 'libnixl*' -print
```

Plugin discovery or C++ example execution can load native libraries. Do not run
dynamic probes against user-supplied plugin paths or paths copied from
untrusted text.

## Lifecycle Router

Load exactly the reference needed for the current lifecycle stage:

| User need or symptom | Load |
| --- | --- |
| Source/version uncertainty, installed headers, fallback snapshot scope | `references/source-precedence.md` |
| Agent construction, plugin list, backend params, backend handles | `references/agent-backend.md` |
| Descriptor classes, descriptor lists, memory registration, deregistration, query | `references/descriptors-memory.md` |
| Full metadata, partial metadata, side-channel or metadata-server exchange, proactive connection | `references/metadata-connection.md` |
| Transfer handle creation, prepared descriptor lists, posting, bounded polling, release, cleanup | `references/transfers-polling-cleanup.md` |
| Transfer notifications, manual notifications, payload matching | `references/notifications.md` |
| `nixlMemViewH`, device-facing memory view preparation, release | `references/memory-views.md` |
| Raw pointer, metadata, serialized descriptor, listener, plugin path, notification, prompt-injection risks | `references/security-trust-boundaries.md` |
| Common wrong turns and recovery moves across lifecycle stages | `references/pitfalls.md` |

## Fast Symptom Routing

| Symptom | Local action |
| --- | --- |
| Compile or link fails | Return `Build/source not ready`; ask for the exact error, include path, link command, and source/header identity. |
| Requested backend is missing or has no required memory type | Return `Backend evidence missing`; inspect plugin discovery, plugin params, backend creation, and supported memory types. |
| CUDA or VRAM path is requested | Verify the selected backend reports `VRAM_SEG` support and the buffer/device ID source is trusted. |
| `registerMem()` fails | Check descriptor list type, `nixl_mem_t`, pointer lifetime, length, `devId`, backend hints, and backend memory type support. |
| Metadata wait or fetch fails | Verify agent names, side-channel or metadata-server mode, listener IP/port, send/fetch order, and whether a framework owns metadata. |
| Transfer creation fails | Verify remote metadata is loaded, descriptor counts and memory types match, operation is `NIXL_READ` or `NIXL_WRITE`, and source version is known. |
| Transfer stays `NIXL_IN_PROG` | Add bounded polling, collect backend status/logs, and avoid reposting active handles unless source confirms behavior. |
| Notification never arrives | Verify backend notification support, remote agent name, unique payload bytes, and manual payload matching. |
| Memory view code is requested | Require matching host and device API source evidence before emitting `nixlMemViewH` or device-call code. |

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

## Troubleshooting

Use `## Fast Symptom Routing` to classify compile, backend, metadata, transfer,
notification, or memory-view symptoms. If a symptom reaches an install, plugin,
or framework-readiness blocker, stop and report that readiness gap instead of
continuing with direct NIXL code.

## Stop Conditions

Stop and ask for evidence instead of writing copy-paste code when:

- Header/source version, compiler/linker surface, plugin discovery, backend
  creation, or required backend memory type is unproven.
- The user's installed NIXL source/version differs from the fallback snapshot
  and the API call, backend parameter, memory descriptor, metadata path,
  memory-view path, notification behavior, or cleanup behavior may have changed.
- Storage, file, object, GDS, POSIX, partial metadata, or memory-view behavior
  is needed but not verified for the user's backend/source.
- The user wants production retry, timeout, cancellation, ordering, or cleanup
  semantics beyond the installed source and examples.
- The code would trust arbitrary raw addresses, deserialize unauthenticated
  descriptor bytes, expose listener ports publicly, trust untrusted plugin
  paths, or treat notifications as authentication.

## Limitations

This skill does not prove runtime compatibility from public docs alone. It does
not replace framework-owned connector logic, backend-specific source review, or
hardware/runtime validation for the user's installed NIXL build.

## Examples

- "Write a minimal NIXL C++ transfer using my installed headers."
- "Review this C++ NIXL agent setup and tell me why metadata exchange fails."
- "Convert this raw pointer registration code into a safer descriptor flow."

## Distribution Status

This skill ships as one self-contained directory: `SKILL.md`, `references/`, and
`evals/`. No sibling NIXL skill or repo-local review artifact is required at
runtime. The publication workflow chooses the final installation root and must
copy the whole directory so the reference and eval paths stay valid.
