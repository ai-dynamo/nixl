# Phase 37: Terminology Normalization and Quality Audit - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Batch grep pass across all new benchmarking pages (Phases 32-36 output) for terminology drift, validate CLI flag tables against `--help` output, confirm `fern check` passes, and audit cross-links. Fix all issues inline. No new content — this is quality assurance only.

</domain>

<decisions>
## Implementation Decisions

### Terminology Rules
- **D-01:** Expanded terminology standardization list (aligning new pages to v1.0 docs conventions):
  - `plug-in` (not `plugin`) — consistent with existing NIXL docs
  - `etcd` (lowercase) in prose text; `ETCD` acceptable in CLI flag values and code contexts
  - `KV cache` (two words, space) in prose text; `kvcache` only in code/CLI contexts (e.g., `kvcache` subcommand)
  - `NIXLBench` (camelCase, one word) — not `NIXL Bench`, `nixlbench` (except in CLI commands), or `Nixlbench`
  - `KVBench` (camelCase, one word) — same pattern as NIXLBench
  - Backend names ALL CAPS in tables and CLI contexts (UCX, GDS, POSIX, etc.)
  - `InfiniBand` (camelCase) — not `Infiniband`, `infiniband`, or `IB` (except in compound terms like `ibverbs`)
  - All other terms: align to whatever the existing v1.0 docs use

### Audit Scope
- **D-02:** Audit only the new benchmarking pages created in Phases 32-36. Do NOT audit existing v1.0 docs — those were already audited. The goal is to align new pages TO the v1.0 conventions, not to re-audit the whole site.
- **D-03:** Cross-links from new pages to existing pages should be verified (links resolve correctly). Cross-links from existing pages to new pages are not expected to exist yet (no existing page was modified to link to benchmarking).

### CLI Flag Validation
- **D-04:** Validate CLI flag tables against actual `--help` output for both tools:
  - NIXLBench: run `nixlbench --help` (or check README if binary not available)
  - KVBench: run `python main.py [cmd] --help` for each subcommand
  - Flag the `--etcd_endpoints` (NIXLBench, underscores) vs `--etcd-endpoints` (KVBench, hyphens) difference — per QS-03, this is intentional and must be documented correctly in each tool's pages
- **D-05:** Fix any mismatches between documented flags and actual `--help` output directly in the markdown files.

### Fern Build Validation
- **D-06:** Run `fern check` from the `fern/` directory. Fix any failures inline — this is the final quality gate. All new pages must pass `fern check` with no errors.
- **D-07:** Optionally run `fern docs dev` to preview rendered output and catch visual issues (broken tables, missing images, malformed MDX). Claude's discretion on whether a visual preview is needed.

### Cross-Link Audit
- **D-08:** Verify all first-mention inline links for backend names resolve to correct User Guide backend pages.
- **D-09:** Verify ETCD links point to `docs/user-guide/etcd-metadata-exchange.md`.
- **D-10:** Verify NIXLBench ↔ KVBench cross-links work in both directions (KVBench overview links to NIXLBench, NIXLBench doesn't need to link back).

### Claude's Discretion
- Whether to run `fern docs dev` for visual preview
- Whether to fix minor style inconsistencies beyond the explicit terminology list (e.g., sentence vs fragment in table cells)
- Ordering of audit steps (grep first vs fern check first)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pages to audit
- `docs/user-guide/benchmarking/nixlbench/index.md` — NIXLBench overview (Phase 33)
- `docs/user-guide/benchmarking/nixlbench/build.md` — NIXLBench build (Phase 33)
- `docs/user-guide/benchmarking/nixlbench/usage.md` — NIXLBench usage + troubleshooting (Phase 34)
- `docs/user-guide/benchmarking/kvbench/index.md` — KVBench overview (Phase 35)
- `docs/user-guide/benchmarking/kvbench/build.md` — KVBench build (Phase 35)
- `docs/user-guide/benchmarking/kvbench/commands.md` — KVBench commands, config, examples (Phase 36)

### Navigation config
- `docs/index.yml` — Verify all benchmarking nav entries resolve correctly

### Terminology reference (v1.0 baseline)
- `docs/user-guide/backends/ucx.md` — Backend page terminology pattern (v1.0 baseline)
- `docs/user-guide/etcd-metadata-exchange.md` — etcd terminology in existing docs
- `docs/user-guide/building-nixl/index.md` — Existing Developer Guide terminology

### CLI validation sources
- `benchmark/nixlbench/README.md` §Command Line Options — NIXLBench flag reference
- `benchmark/kvbench/README.md` §Command Line Arguments — KVBench flag reference
- `benchmark/kvbench/main.py` — KVBench CLI entry point for `--help` validation

### Requirements
- `.planning/REQUIREMENTS.md` — QS-01 (terminology), QS-02 (no duplication), QS-03 (CLI validation), QS-04 (Fern MDX)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- v1.0 docs pages — Terminology baseline to grep against for consistency
- `fern check` command — Built-in Fern validation for MDX/navigation issues

### Established Patterns
- v1.0 already established: `plug-in`, lowercase `etcd` in prose, backend ALL CAPS, `InfiniBand`
- Existing cross-link pattern: first-mention inline links for backend names → User Guide backend pages

### Integration Points
- All 6 new benchmarking pages + `docs/index.yml` are the audit targets
- No existing pages are modified in this phase

</code_context>

<specifics>
## Specific Ideas

- User wants new pages aligned TO v1.0 conventions, not a re-audit of the whole site
- Expanded terminology list beyond REQUIREMENTS QS-01: added KV cache, NIXLBench/KVBench casing, InfiniBand, backend caps
- Fix all issues inline — no separate report/review step

</specifics>

<deferred>
## Deferred Ideas

- Full site terminology audit across all v1.0 + v1.1 pages — not needed, v1.0 already audited
- Automated terminology linting CI check — interesting but out of scope
- Style guide document for future contributors — could be useful but not in this milestone

</deferred>

---

*Phase: 37-terminology-normalization-and-quality-audit*
*Context gathered: 2026-04-07*
