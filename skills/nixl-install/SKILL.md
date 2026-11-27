---
name: nixl-install
description: Investigate NIXL installation, integration, configuration, and runtime issues in software that uses NIXL, such as vLLM or SGLang. Use for setup, compatibility, backend, or transfer-path problems. Do not use for developing or testing NIXL itself.
license: Apache-2.0
metadata:
  author: NVIDIA NIXL Team <sronen@nvidia.com>
  version: "0.1.0"
  tags:
    - nixl
    - installation
    - troubleshooting
  license_source: https://github.com/ai-dynamo/nixl/blob/main/LICENSE
---

# NIXL Install

## Purpose

Help an AI agent investigate and address NIXL installation, integration,
configuration, and runtime issues. For documented use cases, apply
evidence-backed guidance. Do not use this skill to develop or test NIXL itself;
a custom build is in scope only when software that uses NIXL will load it.

## Prerequisites

Access to the target environment or relevant user-provided evidence.

Identify the deployed or target NIXL package versions and, when available, the
build commit as needed to read matched NIXL evidence; routing does not require
a known version. Read this skill's internal files from the installed skill
revision, and linked NIXL source, documentation, and package evidence from the
matching tag or commit. Prefer repository-relative links in a matching local
checkout. Otherwise, open the same repository-root path in the
[NIXL GitHub repository](https://github.com/ai-dynamo/nixl) at that ref.

## Instructions

### 1. Route and select use cases

Route files contain applicability data. Finish routing before opening any
solution or shared file.

Read the route file for each relevant or uncertain use case before selecting.

1. [Framework-supported NIXL](references/routing/framework-supported.md) —
   install, load, or initialize the NIXL package expected by the software.
2. [Chosen or custom NIXL build](references/routing/custom-artifact.md) — build,
   install, or load a specific NIXL build required by the software.
3. [Transfer-path diagnosis](references/routing/transfer-path.md) — diagnose a
   failed, incorrect, skipped, unexpectedly slow, or unexpected-path NIXL
   transfer, or verify its effective path.
4. [LIBFABRIC on AWS EFA](references/routing/libfabric-efa.md) — make the NIXL
   LIBFABRIC backend available, selected, or verifiable on AWS EFA.

One selected use case is the normal case. Select another only when its route
also matches. If applicability is unclear, inspect existing context,
configuration, logs, metrics, and traces with passive checks first. Do not run
transfers, benchmarks, or diagnostics solely for routing. Ask the user only
when a required routing fact or choice cannot be determined another way.

If multiple use cases are selected, order them by relevance and technical
dependencies.

#### No matching route

If no use case is selected or routing remains unresolved, go directly to
[Report](#4-report) and stop.

### 2. Process selected use cases

Process selected use cases one at a time, in the chosen order. Do not open an
unselected solution or a later selected solution before reaching it.

1. [Framework-supported NIXL](references/solutions/framework-supported.md)
2. [Chosen or custom NIXL build](references/solutions/custom-artifact.md)
3. [Transfer-path diagnosis](references/solutions/transfer-path.md)
4. [LIBFABRIC on AWS EFA](references/solutions/libfabric-efa.md)

For each selected use case:

1. Read its solution file.
2. When the solution names a shared check, read the shared file's opening
   rules, only the named sections, and [Evidence boundaries](references/shared/runtime-validation.md#evidence-boundaries).
3. Complete the [Shared workflow](#3-shared-workflow).
4. Continue with the next selected use case only if it still applies.

### 3. Shared workflow

Apply this workflow to the current selected use case. Findings from one use
case may inform another.

1. **Investigate.** Use the loaded use-case data and inspect relevant target
   artifacts, configuration, logs, runtime evidence, source, and
   version-matched documentation. Proceed autonomously with read-only checks.
   Do not restart or modify a production workload only to add diagnostics; use
   an authorized reproduction or a later run.
   Continue meaningful independent checks when another check fails, and stop
   only when further investigation would not change the attribution or remedy.
   A selected solution branch defines the relevant checks. Once it identifies
   one supported remedy, do not append unrelated checks or a benchmark.
2. **Attribute.** Name the first supported divergence and attribute only what
   the evidence supports to NIXL, the integrating software, the host
   environment, or unknown. Attribute a component to the layer that installs,
   selects, or controls it. Use unknown when control is unproven. Distinguish
   observation, inference, and unconfirmed hypotheses. Do not exclude other
   layers until evidence does. If no cause or remedy is supported, mark it
   unresolved and skip Fix.
3. **Fix.** Before applying an authorized remedy, record the prior state needed
   to revert it. Apply the supported remedy to the requested target and verify
   the result. If the user asked only for diagnosis, an explanation, or a
   proposed fix, report the remedy and how to verify it.
4. **Verify.** Verify the changed layer and practical dependent layers against
   the exact target when possible. Report only what was executed and observed.
   Do not infer later layers from an earlier success. If verification is
   blocked or fails, mark that outcome unresolved and continue with the
   remaining use cases when possible.

### 4. Report

Provide one short, concise report when routing stops before
[Process selected use cases](#2-process-selected-use-cases) or all
selected use cases finish. Read the
[report template](references/shared/report-template.md) and follow only its
[No use case](references/shared/report-template.md#no-use-case) rule if no use case was selected or routing remains unresolved.
For selected use cases, read and follow the template when the report involves
changes, unresolved or failed checks, multiple use cases, or non-obvious
attribution; otherwise answer directly.

Report success only for a completed investigation or a requested fix that was
verified.
