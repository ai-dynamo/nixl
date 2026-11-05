# Debug Worklog Template

Use this template when creating a new NIXL debug-session worklog.

```markdown
# Debug: <issue title>

**Date**: <YYYY-MM-DD>
**Source**: <GitHub issue / Linear ticket / pasted report / command>
**Status**: investigating
**Owner**: TBD
**Environment Summary**: TBD
**Primary Suspected Layer**: TBD

## Source Metadata

- Source ID:
- Source URL:
- Reported date:
- Reporter:
- Related commits, PRs, sessions, emails, or docs:

## Problem

<Concise symptom, expected behavior, actual behavior, and impact.>

## Reproduction

### Current Reproduction Command

~~~bash
TBD
~~~

### Steps

1. TBD

### Reproduction Status

- `not-started`, `blocked`, `reproduced`, `not-reproduced`, or `fixed`

## Environment

| Area | Evidence | Source |
| --- | --- | --- |
| NIXL version/commit | TBD | TBD |
| Python environment | TBD | TBD |
| Framework/integration | TBD | TBD |
| Backend/plugin/config | TBD | TBD |
| Hardware/CUDA/runtime | TBD | TBD |
| Container/Kubernetes | TBD | TBD |
| Network/storage path | TBD | TBD |

## Investigation Log

### <timestamp>

- Hypothesis:
- Command/source:
- Evidence:
- Interpretation:
- Next step:

## Findings

- TBD

## Root Cause

TBD

## Fix Plan

TBD

## Verification

| Check | Status | Evidence |
| --- | --- | --- |
| Original reproduction | TBD | TBD |
| Targeted unit/integration test | TBD | TBD |
| Framework or example smoke test | TBD | TBD |
| Regression risk check | TBD | TBD |

## Open Items

- TBD
- External blockers, owners, and requested evidence: TBD
```
