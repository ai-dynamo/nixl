# Phase 37: Terminology Normalization and Quality Audit - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-07
**Phase:** 37-Terminology Normalization and Quality Audit
**Areas discussed:** Terminology rules, Audit scope, Fern build validation

---

## Terminology Rules

| Option | Description | Selected |
|--------|-------------|----------|
| REQUIREMENTS list only | Stick to QS-01 (plug-in, etcd, backend caps) | |
| Expanded term list | Pre-define additional rules: KV cache, NIXLBench, InfiniBand | ✓ |
| You decide | Claude builds list during audit | |

**User's choice:** Expanded term list
**Notes:** User also selected all 4 specific rules (KV cache, NIXLBench, backend caps, InfiniBand) and added "Align to terminology throughout the docs."

---

## Audit Scope

| Option | Description | Selected |
|--------|-------------|----------|
| New pages only | Audit only benchmarking pages from Phases 32-36 | |
| New pages + cross-references | Audit new pages + existing pages that link to/from benchmarking | |
| Full site audit | Audit all docs pages | |
| Custom | User: "Just align the new pages to the previous ones (v1.0)" | ✓ |

**User's choice:** Align new pages to v1.0 conventions
**Notes:** v1.0 pages already audited — just align new pages TO them.

---

## Fern Build Validation

| Option | Description | Selected |
|--------|-------------|----------|
| Fix inline | Run fern check, fix issues directly | ✓ |
| Flag for review | Log failures, don't fix | |
| You decide | Claude handles as part of workflow | |

**User's choice:** Fix inline
**Notes:** Final quality gate — everything must pass.

---

## Claude's Discretion

- Whether to run fern docs dev for visual preview
- Minor style fixes beyond explicit terminology list
- Audit step ordering

## Deferred Ideas

- Full site terminology audit — not needed
- Automated terminology linting CI — out of scope
- Style guide document — future consideration
