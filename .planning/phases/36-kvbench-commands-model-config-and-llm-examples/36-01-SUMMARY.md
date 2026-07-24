---
phase: 36-kvbench-commands-model-config-and-llm-examples
plan: 01
subsystem: docs
tags: [kvbench, fern, cli-reference, yaml-schema, deepseek, llama, benchmarking]

# Dependency graph
requires:
  - phase: 35-kvbench-overview-and-build
    provides: KVBench overview and build pages (index.md, build.md)
provides:
  - Complete KVBench commands.md page with command reference, model config guide, and LLM examples
affects: [37-terminology-normalization-and-quality-audit]

# Tech tracking
tech-stack:
  added: []
  patterns: [two-column CLI argument tables, annotated YAML examples, end-to-end copy-paste examples]

key-files:
  created: []
  modified:
    - docs/development/benchmarking/kvbench/commands.md

key-decisions:
  - "Documented README CLI arguments only; backend-specific args (GDS, GPUNETIO, HF3FS, OBJ) are passthrough to NIXLBench and documented on respective backend pages"
  - "Llama 3.1 example shows command invocation without fabricated output values, with note referencing DeepSeek R1 output format"
  - "Used --etcd-endpoints (hyphens) throughout KVBench docs per README convention, with cross-tool note about NIXLBench underscores"

patterns-established:
  - "MLA vs MHA/GQA field table pattern: common fields first, then architecture-specific fields in separate tables"
  - "End-to-end LLM example pattern: model arch YAML + model config YAML + plan command + output + profile command"

requirements-completed: [KB-03, KB-04, KB-05]

# Metrics
duration: 3min
completed: 2026-04-07
---

# Phase 36 Plan 01: KVBench Commands, Model Config, and LLM Examples Summary

**Complete KVBench commands.md page (494 lines) with all 5 subcommand references, CLI argument tables, dual YAML schema documentation, and DeepSeek R1 / Llama 3.1 end-to-end examples**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-07T21:10:52Z
- **Completed:** 2026-04-07T21:13:55Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Replaced 7-line stub with 494-line complete page covering all 4 major sections
- Documented all 5 KVBench subcommands (plan, profile, kvcache, ct-perftest, sequential-ct-perftest) with usage examples and output formats
- Created 5 CLI argument group tables validated against args.py Click definitions
- Documented both model architecture YAML schemas (MLA for DeepSeek R1, MHA/GQA for Llama 3.1) and model config YAML schema (strategy/runtime/system)
- Provided DeepSeek R1 end-to-end examples with both block and layer access patterns showing output differences
- Provided Llama 3.1 70B end-to-end example with block access pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Author Command Reference and CLI Argument Tables** - `5b503326` (feat)
2. **Task 2: Author Model Configuration Guide and LLM Examples** - `ed7bc08f` (feat)

## Files Created/Modified
- `docs/development/benchmarking/kvbench/commands.md` - Complete KVBench commands, model config, and LLM examples page (494 lines, replacing 7-line stub)

## Decisions Made
- Documented only the CLI arguments present in the README tables (matching D-04 grouping). Backend-specific arguments from args.py (GDS batch settings, GPUNETIO device list, HF3FS pool size, OBJ/S3 credentials) are passthrough to NIXLBench and documented on respective backend pages.
- For Llama 3.1, showed only the command invocation without fabricated output values. Added a Note component explaining the output follows the same format as DeepSeek R1. This avoids misleading developers with unverified numbers.
- Used the actual YAML file values from the examples/ directory for model config examples. The README examples show different ISL/page_size values achieved via CLI overrides.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- KVBench documentation is complete across all 3 pages (index.md, build.md, commands.md)
- Ready for Phase 37 terminology normalization and quality audit
- fern check passes with 0 errors

---
*Phase: 36-kvbench-commands-model-config-and-llm-examples*
*Completed: 2026-04-07*
