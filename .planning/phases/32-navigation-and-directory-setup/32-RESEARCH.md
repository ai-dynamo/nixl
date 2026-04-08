# Phase 32: Navigation and Directory Setup — Research

**Researched:** 2026-04-07
**Confidence:** HIGH
**Scope:** Structural only — navigation tree changes and stub file creation

## RESEARCH COMPLETE

## 1. Current Navigation Structure

`docs/index.yml` contains 6 top-level sections: Getting Started, User Guide, Developer Guide, Examples, API Reference, Resources.

The Developer Guide currently has 2 entries:
1. `section: Building NIXL from Source` (line 55) — collapsed subsection with `path:` to index.md + child pages
2. `page: Building a Backend Plugin` (line 67) — single page entry

New Benchmarking section goes **after** line 68 (`path: development/building-a-backend-plugin.md`), still inside the Developer Guide `contents:` block.

## 2. Section/Page Patterns in docs/index.yml

### Pattern A: Section without landing page (Getting Started)
```yaml
- section: Getting Started
  contents:
    - page: Overview
      path: getting-started/overview.md
```
No `path:` on the section itself. Used for the top-level "Benchmarking" parent section per CONTEXT.md D-01.

### Pattern B: Section with landing page (Building NIXL from Source)
```yaml
- section: Building NIXL from Source
  collapsed: open-by-default
  path: user-guide/building-nixl/index.md
  contents:
    - page: Docker
      path: user-guide/building-nixl/docker.md
```
Has `path:` pointing to index.md, `collapsed: open-by-default`. Used for NIXLBench and KVBench nested sections per CONTEXT.md D-02.

### Pattern C: Single page
```yaml
- page: Building a Backend Plugin
  path: development/building-a-backend-plugin.md
```

## 3. Directory Naming Discrepancy

**Critical finding:** CONTEXT.md D-05 says `docs/development/benchmarking/` but ROADMAP.md success criteria #2 and REQUIREMENTS.md NAV-02 say `docs/development/benchmarks/`. ARCHITECTURE.md also uses `benchmarks/`.

**Resolution:** Follow CONTEXT.md (user decisions), which says `benchmarking/`. The ROADMAP and REQUIREMENTS were written before the discuss-phase captured user preferences. CONTEXT.md D-01 explicitly says `section: Benchmarking` (not "Benchmarks"), and D-05 explicitly says `docs/development/benchmarking/`. The directory name `benchmarking/` is consistent with the section label `Benchmarking`.

## 4. Stub File Requirements

Per CONTEXT.md D-06 through D-08, exactly 4 stub files:

| File | Title (D-08) | Purpose |
|------|--------------|---------|
| `docs/development/benchmarking/nixlbench/index.md` | NIXLBench | Section index (overview + build) |
| `docs/development/benchmarking/nixlbench/usage.md` | NIXLBench Usage and Troubleshooting | Usage + troubleshooting |
| `docs/development/benchmarking/kvbench/index.md` | KVBench | Section index (overview + build) |
| `docs/development/benchmarking/kvbench/commands.md` | KVBench Commands and Examples | Commands, model config, examples |

Per D-07: each stub needs `title:` and `description:` frontmatter plus a placeholder sentence. No bare/empty files.

## 5. Frontmatter Pattern

Existing pages use YAML frontmatter:
```yaml
---
title: Building NIXL from Source
description: Build NIXL from source -- C++ library, Python bindings, Rust bindings, or Docker container.
---
```

Stubs must follow this exact pattern.

## 6. Expected docs/index.yml Changes

Insert after line 68 (Building a Backend Plugin path), still inside Developer Guide contents:

```yaml
      - section: Benchmarking
        contents:
          - section: NIXLBench
            collapsed: open-by-default
            path: development/benchmarking/nixlbench/index.md
            contents:
              - page: Usage and Troubleshooting
                path: development/benchmarking/nixlbench/usage.md
          - section: KVBench
            collapsed: open-by-default
            path: development/benchmarking/kvbench/index.md
            contents:
              - page: Commands and Examples
                path: development/benchmarking/kvbench/commands.md
```

Key observations:
- Benchmarking parent section has **no `path:`** (matches D-01, like Getting Started pattern)
- NIXLBench and KVBench sub-sections each have `path:` to their index.md (matches D-02)
- `collapsed: open-by-default` on both sub-sections (matches existing pattern and D-02)
- Child page paths are relative to `docs/` (same as all other paths in index.yml)

## 7. Fern Check Considerations

- All paths declared in `docs/index.yml` must resolve to actual files under `docs/`
- Files must have valid frontmatter (empty files may fail)
- `fern check` is run from the `fern/` directory
- The `collapsed` value `open-by-default` is valid (used twice already in index.yml)

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `fern check` rejects `collapsed: open-by-default` on nested sections | LOW | Already used in current config |
| Path mismatch between index.yml and filesystem | LOW | Use exact paths from D-06 |
| Empty stub content triggers fern warnings | LOW | D-07 requires frontmatter + placeholder |
| YAML indentation error in index.yml | MEDIUM | Verify indentation matches existing patterns (6 spaces for contents inside Developer Guide) |

## 9. Validation Architecture

### Verification Approach
1. **File existence check:** All 4 stub files exist with non-zero content
2. **Directory existence check:** Both `nixlbench/` and `kvbench/` directories exist under `docs/development/benchmarking/`
3. **YAML syntax check:** `docs/index.yml` parses without errors
4. **Frontmatter check:** Each stub has `title:` and `description:` fields
5. **fern check:** Run `fern check` from `fern/` directory — must pass with zero errors

### Acceptance Criteria Mapping
- NAV-01 → index.yml contains Benchmarking section with NIXLBench and KVBench sub-sections
- NAV-02 → Directories and files exist at declared paths
- NAV-03 → `fern check` passes
