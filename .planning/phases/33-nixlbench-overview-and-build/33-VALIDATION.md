---
phase: 33
slug: nixlbench-overview-and-build
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-07
---

# Phase 33 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Fern CLI (`fern check`) |
| **Config file** | `fern/fern.config.json` |
| **Quick run command** | `cd fern && fern check` |
| **Full suite command** | `cd fern && fern check` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd fern && fern check`
- **After every plan wave:** Run `cd fern && fern check`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 33-01-01 | 01 | 1 | NB-01, NB-02 | structural | `test -f docs/development/benchmarking/nixlbench/index.md && test -f docs/development/benchmarking/nixlbench/build.md` | pending |
| 33-01-02 | 01 | 1 | NB-01 | content | `grep -q 'title:' docs/development/benchmarking/nixlbench/index.md` | pending |
| 33-01-03 | 01 | 1 | NB-02 | content | `grep -q '<Tabs>' docs/development/benchmarking/nixlbench/build.md` | pending |
| 33-01-04 | 01 | 1 | NB-02 | cross-link | `grep -q 'building-nixl' docs/development/benchmarking/nixlbench/build.md` | pending |
| 33-01-05 | 01 | 1 | NB-01, NB-02 | fern | `cd fern && fern check` | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements. `fern check` is already available.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Visual rendering of Tabs component | NB-02 | Requires visual inspection in browser | Run `cd fern && fern docs dev`, navigate to NIXLBench build page, verify Docker/Native tabs render correctly |

---

## Validation Sign-Off

- [x] All tasks have automated verify commands
- [x] Sampling continuity: fern check runs after every task
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 5s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
