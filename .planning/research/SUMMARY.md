# Research Summary — NIXLBench and KVBench Documentation (v1.1)

## Executive Summary

This milestone adds documentation for two developer benchmark tools — NIXLBench (C++ binary) and KVBench (Python CLI) — to the existing NIXL Fern docs site under the Developer Guide. Both tools are already in the codebase with comprehensive READMEs, but the READMEs are not suitable for direct conversion: they use GitHub-flavored Markdown conventions incompatible with Fern MDX, mix content types that should be separated into distinct pages, and duplicate information already covered by the existing docs site (building NIXL from source, backend configuration, ETCD setup). The recommended approach is to reauthor content from scratch using the READMEs as source material — not to copy-paste them.

The correct architecture is 9 new MDX pages organized into two subdirectories under `docs/development/benchmarks/`, integrated into the Developer Guide section of `docs/index.yml`. All Fern components needed are already live on the site — no new custom components are required.

---

## Stack Additions

- No new tooling required — all Fern components (`<Tabs>`, `<Note>`, `<Warning>`, `<Tip>`, fenced code blocks with `title=`) are already live on the production site
- Do NOT use `<CodeBlocks>` for single-language examples (NIXLBench is C++ CLI only, KVBench is Python CLI only)
- Use `text` language tag for terminal output blocks (kvcache table, latency output) to prevent false syntax highlighting
- `<Tabs>` for Docker vs. native build (both tools have exactly two install paths)
- `<Warning>` for ETCD 60s barrier timeout and KVBench `CUDA_VISIBLE_DEVICES` requirement

---

## Feature Table Stakes (P1 — this milestone)

**NIXLBench:**
- Overview page
- Building page (Docker + native combined, with `<Tabs>`)
- CLI reference (core flags + storage flags + config file flags — grouped, not one mega-table)
- ETCD coordination callout
- Backend-specific examples (8 backends)
- Troubleshooting

**KVBench:**
- Overview page (NIXLBench subprocess dependency in paragraph one — mandatory)
- Building page (Docker + Python venv)
- Command reference (all 5 commands: plan, profile, kvcache, ct-perftest, sequential-ct-perftest)
- Model configuration guide
- LLM architecture examples (DeepSeek R1, Llama 3.1)
- CTP examples

**P2 (include if scope allows):** KVBench GDS tutorial (source: `benchmark/kvbench/docs/tutorial-gds.md`), KVBench extension guide (source: `benchmark/kvbench/docs/adding-a-new-model-architecture.md`)

---

## Architecture

- **9 new files** under `docs/development/benchmarks/nixlbench/` (5 pages) and `docs/development/benchmarks/kvbench/` (4 pages)
- **1 modified file:** `docs/index.yml` — two new `section:` blocks under Developer Guide after "Building a Backend Plugin"
- No changes to `fern/docs.yml`, no existing page modifications, no new components
- Authoring order: `docs/index.yml` first → NIXLBench pages → KVBench pages

---

## Watch Out For

1. **Content duplication** — READMEs contain full Docker/CUDA/UCX/NIXL/ETCD walkthroughs already in existing docs. Audit before writing; substitute cross-links.
2. **KVBench misdescribed as standalone** — `profile` invokes `nixlbench` as a subprocess. State this in the first paragraph of KVBench overview.
3. **Monolithic pages** — NIXLBench has 70+ flags, KVBench has 30+ args. Enforce page split by creating `docs/index.yml` entries before writing content.
4. **Navigation YAML** — Copy "Building NIXL from Source" section pattern exactly (`section:` + `path:` + `contents:`).
5. **Stale CLI tables** — Run `nixlbench --help` and `python main.py [cmd] --help` before marking CLI reference phases complete. Known discrepancy: NIXLBench uses `--etcd_endpoints` (underscores); KVBench uses `--etcd-endpoints` (hyphens).
6. **Terminology drift** — `plugin` → `plug-in`, `ETCD` → `etcd` in prose; batch normalization pass after all drafts.
7. **Backend cross-references** — 11 backends, each should link to its User Guide page on first mention.

---

## Suggested Phase Structure (6 phases, starting at Phase 32)

| Phase | Name | Goal |
|-------|------|------|
| 32 | Navigation Setup and Directory Structure | Create `docs/development/benchmarks/` tree and `docs/index.yml` entries before any content |
| 33 | NIXLBench Core Pages | Overview, Build (Docker + native), Usage guide |
| 34 | NIXLBench CLI Reference and Backend Examples | All flags validated against `--help`, backend cross-refs |
| 35 | KVBench Core Pages | Overview (NIXLBench dependency stated), Build, Command reference |
| 36 | KVBench Model Config and LLM Examples | Model config guide, DeepSeek R1 + Llama examples, CTP examples |
| 37 | Terminology Normalization and Cross-Reference Audit | Batch grep pass, all backend names hyperlinked, `fern check` clean |

---

## Confidence: HIGH

All source material is in the codebase. All Fern platform patterns are verified live on the production site. This is an authoring and content organization task throughout — no phases require additional research.
