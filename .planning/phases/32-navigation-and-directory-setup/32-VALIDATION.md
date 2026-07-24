---
phase: 32
slug: navigation-and-directory-setup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 32 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | fern check (Fern CLI validation) |
| **Config file** | `fern/docs.yml` + `docs/index.yml` |
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

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 32-01-01 | 01 | 1 | NAV-01 | — | N/A | integration | `grep -c 'section: Benchmarking' docs/index.yml` | N/A | ⬜ pending |
| 32-01-02 | 01 | 1 | NAV-01 | — | N/A | integration | `grep -c 'section: NIXLBench' docs/index.yml` | N/A | ⬜ pending |
| 32-01-03 | 01 | 1 | NAV-01 | — | N/A | integration | `grep -c 'section: KVBench' docs/index.yml` | N/A | ⬜ pending |
| 32-02-01 | 02 | 1 | NAV-02 | — | N/A | existence | `test -d docs/development/benchmarking/nixlbench && test -d docs/development/benchmarking/kvbench` | N/A | ⬜ pending |
| 32-02-02 | 02 | 1 | NAV-02 | — | N/A | existence | `test -f docs/development/benchmarking/nixlbench/index.md` | N/A | ⬜ pending |
| 32-02-03 | 02 | 1 | NAV-02 | — | N/A | existence | `test -f docs/development/benchmarking/nixlbench/usage.md` | N/A | ⬜ pending |
| 32-02-04 | 02 | 1 | NAV-02 | — | N/A | existence | `test -f docs/development/benchmarking/kvbench/index.md` | N/A | ⬜ pending |
| 32-02-05 | 02 | 1 | NAV-02 | — | N/A | existence | `test -f docs/development/benchmarking/kvbench/commands.md` | N/A | ⬜ pending |
| 32-03-01 | 02 | 1 | NAV-03 | — | N/A | integration | `cd fern && fern check` | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.*

No test framework installation needed — validation uses `fern check` (already available) and shell commands (`grep`, `test`).

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
