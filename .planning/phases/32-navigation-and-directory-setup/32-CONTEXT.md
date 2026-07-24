# Phase 32: Navigation and Directory Setup - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Add Benchmarks subsections to `docs/index.yml` and create the directory scaffolding with stub files before any content is authored. This phase is structural only — no real content is written. Success = `fern check` passes with stubs in place.

</domain>

<decisions>
## Implementation Decisions

### Navigation Structure

- **D-01:** A single `section: Benchmarking` block is inserted into `docs/index.yml` under Developer Guide, after the `"Building a Backend Plugin"` entry. It has **no `path:` (no section index page)** — consistent with the "Getting Started" section pattern where the section itself has no landing page.

- **D-02:** Inside `Benchmarking`, two nested sections: `section: NIXLBench` and `section: KVBench`. Each has a `path:` pointing to its own `index.md` (which serves as the tool overview + build page). Each section uses `collapsed: open-by-default`.

- **D-03:** Each tool section has exactly **2 child pages** (not counting the section index):
  - NIXLBench: `index.md` (overview + build) + `usage.md` (usage + troubleshooting) = 2 declared pages under the section
  - KVBench: `index.md` (overview + build) + `commands.md` (commands, model config, examples) = 2 declared pages under the section

- **D-04:** Total new `docs/index.yml` entries: 4 pages + 2 section nodes = 6 filesystem paths declared (all must have stubs by end of this phase).

### Directory Layout

- **D-05:** New directories: `docs/development/benchmarking/nixlbench/` and `docs/development/benchmarking/kvbench/`. The shared `benchmarking/` intermediate directory groups both tools under `docs/development/`.

- **D-06:** New stub files (6 total):
  ```
  docs/development/benchmarking/nixlbench/index.md
  docs/development/benchmarking/nixlbench/usage.md
  docs/development/benchmarking/kvbench/index.md
  docs/development/benchmarking/kvbench/commands.md
  ```
  (4 content files — the 2 section nodes in `docs/index.yml` point to `index.md` files, not additional separate files)

### Stub File Content

- **D-07:** Each stub contains minimal Fern frontmatter (`title:` and `description:` fields) plus a single placeholder sentence. No bare-file (zero-byte) stubs — those may trigger `fern check` warnings.

- **D-08:** `title:` values for the 4 stubs:
  - `nixlbench/index.md` → `"NIXLBench"`
  - `nixlbench/usage.md` → `"NIXLBench Usage and Troubleshooting"`
  - `kvbench/index.md` → `"KVBench"`
  - `kvbench/commands.md` → `"KVBench Commands and Examples"`

### Claude's Discretion

- Exact `collapsed:` value for NIXLBench/KVBench nested sections (use `open-by-default` to match existing subsection pattern unless `fern check` rejects it)
- Whether to add a `description:` annotation to the Benchmarking section node in `docs/index.yml`

</decisions>

<specifics>
## Specific Ideas

- User confirmed: "Developer Guide → Benchmarking → {NIXLBench, KVBench}" — not two separate top-level sections, one parent Benchmarking section
- User confirmed: no section index/landing page for Benchmarking itself
- User confirmed: 2 pages per tool (not 4-5 as originally planned in ARCHITECTURE.md)

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Navigation config
- `docs/index.yml` — existing navigation structure; all new entries must follow its section/page/path patterns. NIXLBench+KVBench entries go under Developer Guide after line 68 ("Building a Backend Plugin").

### Existing subsection pattern (reference implementation)
- `docs/index.yml:53-65` — "Building NIXL from Source" section pattern: `section:` + `collapsed: open-by-default` + `path:` to index.md + `contents:` list
- `docs/index.yml:1-6` — "Getting Started" section pattern (no `path:`, no index page) — use this for the `Benchmarking` parent section

### Architecture decisions
- `.planning/research/ARCHITECTURE.md` — Integration architecture, original file list, nav diff (note: scope has changed to 4 pages from 9)
- `.planning/REQUIREMENTS.md` — NAV-01, NAV-02, NAV-03 are the acceptance criteria for this phase

### Fern platform reference
- `fern/docs.yml` — Top-level Fern config; not modified in this phase but read to understand `products:` path references

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/user-guide/building-nixl/index.md` — Section index page pattern with frontmatter; template for nixlbench/index.md and kvbench/index.md stubs
- `docs/development/building-a-backend-plugin.md` — Single-page Developer Guide entry pattern

### Established Patterns
- All pages in `docs/` carry `title:` and `description:` frontmatter — stubs must follow this
- Subsections under Developer Guide use `collapsed: open-by-default`
- Section nodes without a `path:` (like "Getting Started") are valid in Fern — no landing page needed

### Integration Points
- `docs/index.yml` is the only file modified in this phase (plus creating 4 new stub files and 2 new directories)
- No changes to `fern/docs.yml`, `fern/fern.config.json`, or existing content files

</code_context>

<deferred>
## Deferred Ideas

- CLI reference page for NIXLBench (70+ flags) — scoped out by user in milestone requirements
- Backend-specific example pages for NIXLBench — scoped out
- KVBench CTP examples page — scoped out
- KVBench GDS tutorial and extension guide — P2, deferred

</deferred>

---

*Phase: 32-navigation-and-directory-setup*
*Context gathered: 2026-04-07*
