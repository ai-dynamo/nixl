---
phase: 32-navigation-and-directory-setup
plan: 01
subsystem: docs/navigation
tags: [fern, navigation, index.yml]
key-files:
  modified: [docs/index.yml]
  created: []
metrics:
  tasks: 1
  commits: 1
  files_changed: 1
---

# Plan 32-01 Summary: Add Benchmarking Navigation

## What Was Built

Added a `section: Benchmarking` block to `docs/index.yml` under Developer Guide, after "Building a Backend Plugin". The section contains two nested sub-sections (NIXLBench and KVBench), each with `collapsed: open-by-default` and paths to their respective index.md files plus child pages.

## Commits

| # | Hash | Description |
|---|------|-------------|
| 1 | 6b612d2e | docs(32-01): add Benchmarking navigation section to Developer Guide |

## Deviations

None.

## Self-Check: PASSED

- [x] `section: Benchmarking` appears in docs/index.yml (1 match)
- [x] `section: NIXLBench` appears in docs/index.yml (1 match)
- [x] `section: KVBench` appears in docs/index.yml (1 match)
- [x] All 4 page paths declared
- [x] Benchmarking section has no `path:` attribute (no landing page)
- [x] YAML parses without errors
