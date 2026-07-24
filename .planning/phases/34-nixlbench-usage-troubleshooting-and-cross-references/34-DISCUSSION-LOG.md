# Phase 34: NIXLBench Usage, Troubleshooting, and Cross-References - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-07
**Phase:** 34-NIXLBench Usage, Troubleshooting, and Cross-References
**Areas discussed:** Page split, Usage content scope, Troubleshooting scope, Backend examples

---

## Page Split

| Option | Description | Selected |
|--------|-------------|----------|
| Combined usage.md | Usage + troubleshooting in one page, troubleshooting as section at bottom | ✓ |
| Separate troubleshooting.md | usage.md for guide, troubleshooting.md for failure modes | |
| You decide | Claude picks based on content volume | |

**User's choice:** Combined usage.md
**Notes:** Keeps NIXLBench nav at 3 pages (index.md, build.md, usage.md)

---

## Usage Content Scope — CLI Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Core flags only | ~10-15 essential flags in a single table | |
| Grouped flag tables | Core + memory/transfer + performance (3 tables, ~30 flags) | |
| You decide | Claude determines right level for 4 communication patterns | ✓ |

**User's choice:** You decide
**Notes:** None

---

## Usage Content Scope — ETCD

| Option | Description | Selected |
|--------|-------------|----------|
| Warning callout + link | <Warning> for 60s barrier, link to ETCD page, brief Docker setup | ✓ |
| Full ETCD section | Dedicated subsection covering setup, required vs optional | |
| You decide | Claude balances based on NB-05 | |

**User's choice:** Warning callout + link
**Notes:** Matches NB-05 requirement

---

## Troubleshooting Scope

| Option | Description | Selected |
|--------|-------------|----------|
| ROADMAP 4 + runtime essentials | 4 required modes + library-not-found + ETCD cleanup | |
| ROADMAP 4 only | Strict 4 failure modes from success criteria | |
| You decide | Claude picks based on developer needs | ✓ |

**User's choice:** You decide
**Notes:** None

---

## Backend Examples

| Option | Description | Selected |
|--------|-------------|----------|
| Pattern-focused examples | 4 patterns using UCX + 1-2 storage examples (GDS, OBJ) | ✓ |
| Per-backend examples | Brief example for each major backend | |
| You decide | Claude picks approach | |

**User's choice:** Pattern-focused examples
**Notes:** Link to backend User Guide pages for backend-specific flags

---

## Claude's Discretion

- CLI flag table scope
- Runtime issues in troubleshooting (ETCD cleanup, library-not-found)
- Config file example inclusion
- NVSHMEM worker type mention

## Deferred Ideas

- Full CLI reference (70+ flags) — out of scope for v1.1
- Per-backend example pages — deferred
- Performance tuning guide — not in ROADMAP scope
