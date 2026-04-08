---
phase: 34-nixlbench-usage-troubleshooting-and-cross-references
plan: 01
subsystem: docs
tags: [nixlbench, fern-mdx, etcd, benchmarking, troubleshooting, cross-links]

# Dependency graph
requires:
  - phase: 33-nixlbench-overview-and-build
    provides: NIXLBench index.md and build.md pages with established Fern MDX patterns and backend link format
provides:
  - Complete NIXLBench usage and troubleshooting page with etcd coordination, communication patterns, storage examples, CLI tables, and troubleshooting
affects: [37-terminology-normalization-and-quality-audit]

# Tech tracking
tech-stack:
  added: []
  patterns: [first-mention backend cross-linking, Warning callout for critical timing constraints, Symptoms/Resolution troubleshooting format]

key-files:
  created: []
  modified: [docs/development/benchmarking/nixlbench/usage.md]

key-decisions:
  - "All 11 backend names linked on first mention in the etcd Coordination paragraph for compact cross-referencing"
  - "CLI tables split into Core Configuration (6 flags) and Memory/Transfer Configuration (12 flags) per plan spec"
  - "Troubleshooting uses Symptoms/Resolution format for scannable problem-solving"
  - "TOML config file and NVSHMEM worker type noted briefly after CLI tables per discretion recommendations"

patterns-established:
  - "Symptoms/Resolution troubleshooting format for developer-facing docs"
  - "First-mention backend cross-linking consolidated in introductory paragraph"

requirements-completed: [NB-03, NB-04, NB-05, NB-06]

# Metrics
duration: 2min
completed: 2026-04-07
---

# Phase 34 Plan 01: NIXLBench Usage and Troubleshooting Summary

**Complete usage guide with etcd coordination, four communication patterns (pairwise, many-to-one, one-to-many, TP), GDS and OBJ storage examples, 18-flag CLI reference, and four troubleshooting sections with cross-links to all backend User Guide pages**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-07T20:16:16Z
- **Completed:** 2026-04-07T20:18:20Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Replaced stub usage.md with 249-line complete Fern MDX page covering all NIXLBench usage scenarios
- Cross-linked all 10 backend names (UCX, GDS, GDS_MT, POSIX, GPUNETIO, Mooncake, Libfabric, HF3FS, OBJ, GUSLI) to their User Guide pages on first mention
- Warning callout for the 60-second etcd join window barrier with link to etcd metadata exchange page
- Four communication patterns with multi-node examples for pairwise and single-command examples for the other three
- Four troubleshooting sections covering etcd connection failures, build failures, CUDA/GPU not found, and backend library missing

## Task Commits

Each task was committed atomically:

1. **Task 1: Author usage.md with all sections** - `9ed05e4b` (feat)
2. **Task 2: Verify backend cross-links and terminology compliance** - no commit (verification-only, no fixes needed)

## Files Created/Modified
- `docs/development/benchmarking/nixlbench/usage.md` - Complete NIXLBench usage and troubleshooting page replacing Phase 32 stub

## Decisions Made
- Consolidated all 10 backend first-mention links into the etcd Coordination paragraph rather than spreading them across sections -- provides a dense but scannable cross-reference hub
- Included TOML config file note and NVSHMEM worker type mention as brief paragraphs after CLI tables per RESEARCH discretion recommendations
- Used "On host 1 (initiator):" / "On host 2 (target):" format for pairwise multi-node example per D-07

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- usage.md is complete and fern check passes with zero errors
- All backend cross-links verified; terminology (etcd lowercase, no plugin) compliant
- Ready for Phase 37 terminology normalization and quality audit

---
*Phase: 34-nixlbench-usage-troubleshooting-and-cross-references*
*Completed: 2026-04-07*
