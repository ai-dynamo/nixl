# Architecture Research

**Domain:** Documentation site content integration — Fern docs platform (NIXL v1.1 milestone)
**Researched:** 2026-04-07
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
docs/index.yml                          (navigation source of truth)
    |
    ├── Getting Started/                (docs/getting-started/)
    ├── User Guide/                     (docs/user-guide/)
    ├── Developer Guide/                (docs/user-guide/building-nixl/,
    │                                    docs/development/)
    │       └── [NEW] Benchmarks/       (docs/development/benchmarks/)
    │               ├── NIXLBench/
    │               └── KVBench/
    ├── Examples/                       (docs/examples/)
    ├── API Reference/                  (docs/api-reference/, docs/development/)
    └── Resources/                      (docs/resources/)
```

The Fern platform reads `fern/docs.yml` (site config: colors, layout, navbar) and `docs/index.yml` (navigation tree + page paths). All content lives under `docs/`. The navigation tree in `docs/index.yml` is the single integration point for adding new sections.

### Component Responsibilities

| Component | Responsibility | Current State |
|-----------|----------------|---------------|
| `fern/docs.yml` | Site-wide config: branding, layout, products, versions | No changes needed for this milestone |
| `docs/index.yml` | Navigation tree; maps section labels to file paths | Needs two new subsection blocks added to Developer Guide |
| `docs/user-guide/building-nixl/` | Pattern reference: a section with index.md + child pages | Existing pattern to mirror |
| `docs/development/` | Houses non-building developer content (backend plugin, SB API) | Will receive `benchmarks/` subdirectory |

## Recommended Project Structure

The correct location for both benchmark tool documentation is under `docs/development/benchmarks/` as subsections within the existing Developer Guide.

```
docs/
├── development/
│   ├── building-a-backend-plugin.md    (existing)
│   ├── sb-api-reference.md             (existing)
│   └── benchmarks/                     (NEW directory)
│       ├── nixlbench/                  (NEW directory)
│       │   ├── index.md                (NEW — overview + quick start)
│       │   ├── build.md                (NEW — Docker + native build)
│       │   ├── usage.md                (NEW — ETCD setup, basic usage, config file)
│       │   ├── cli-reference.md        (NEW — full flag reference by category)
│       │   └── backend-examples.md     (NEW — per-backend command examples)
│       └── kvbench/                    (NEW directory)
│           ├── index.md                (NEW — overview, KVBench vs CTP commands)
│           ├── build.md                (NEW — Docker + Python venv install)
│           ├── commands.md             (NEW — plan, profile, kvcache, ct-perftest, sequential-ct-perftest)
│           └── llm-examples.md         (NEW — DeepSeek R1, LLaMA configs, CTP YAML examples)
```

### Structure Rationale

- **`docs/development/benchmarks/`:** Keeps benchmark tools together under one parent, matching the mental model that these are developer-facing tools rather than end-user guides. Avoids polluting the `development/` root with many flat files.
- **Subdirectory per tool (`nixlbench/`, `kvbench/`):** Each tool has enough content (5+ pages) to warrant its own subdirectory. Mirrors the pattern of `building-nixl/` which uses `index.md` + child pages.
- **`index.md` per tool:** Fern renders a subsection landing page when a `section:` block in `docs/index.yml` has a `path:` pointing to an `index.md`. This pattern is already established with `user-guide/building-nixl/index.md`.
- **Not a new top-level section:** NIXLBench and KVBench are developer tools, not user-facing product features. They belong in Developer Guide alongside "Building from Source" and "Building a Backend Plugin". A new top-level section would fragment the navigation unnecessarily.
- **Not flat pages in `development/`:** Five pages for NIXLBench plus four for KVBench (nine total) would clutter the development root. The subdirectory grouping keeps them scannable.

## Navigation Structure

The following diff to `docs/index.yml` (the Developer Guide section block) shows the exact changes needed:

```yaml
# ==================== Developer Guide ====================
- section: Developer Guide
  contents:
    - section: Building NIXL from Source          # existing
      collapsed: open-by-default
      path: user-guide/building-nixl/index.md
      contents:
        - page: Docker
          path: user-guide/building-nixl/docker.md
        - page: "NIXL C++ (Meson)"
          path: user-guide/building-nixl/nixl-cpp.md
        - page: Python Bindings
          path: user-guide/building-nixl/python-bindings.md
        - page: Rust Bindings
          path: user-guide/building-nixl/rust-bindings.md
    - page: Building a Backend Plugin              # existing
      path: development/building-a-backend-plugin.md
    # ---- NEW BELOW ----
    - section: NIXLBench
      collapsed: open-by-default
      path: development/benchmarks/nixlbench/index.md
      contents:
        - page: Building NIXLBench
          path: development/benchmarks/nixlbench/build.md
        - page: Usage Guide
          path: development/benchmarks/nixlbench/usage.md
        - page: CLI Reference
          path: development/benchmarks/nixlbench/cli-reference.md
        - page: Backend Examples
          path: development/benchmarks/nixlbench/backend-examples.md
    - section: KVBench
      collapsed: open-by-default
      path: development/benchmarks/kvbench/index.md
      contents:
        - page: Building KVBench
          path: development/benchmarks/kvbench/build.md
        - page: Commands
          path: development/benchmarks/kvbench/commands.md
        - page: LLM Architecture Examples
          path: development/benchmarks/kvbench/llm-examples.md
```

## Architectural Patterns

### Pattern 1: Section with Landing Index

**What:** A `section:` block in `docs/index.yml` includes both a `path:` (pointing to `index.md`) and a `contents:` list of child pages. The `index.md` serves as the section overview — it introduces the tool and links to child pages.

**When to use:** Any multi-page topic with 3+ child pages. Both NIXLBench (5 pages) and KVBench (4 pages) qualify.

**Trade-offs:** Requires maintaining an extra `index.md` per tool, but gives users a scoped entry point and allows Fern to render the section as a collapsible nav group with a clickable parent.

**Example (existing):**
```
docs/user-guide/building-nixl/index.md   ← section landing
docs/user-guide/building-nixl/docker.md  ← child page
```

### Pattern 2: Flat Page in Section

**What:** A `page:` entry directly under a `section:` in `docs/index.yml`, with no child pages.

**When to use:** Single-topic pages that do not need sub-navigation. "Building a Backend Plugin" is the existing example.

**Trade-offs:** Simple, but does not scale if the page grows. Do not apply this pattern to NIXLBench or KVBench — both are large enough to warrant the multi-page approach.

### Pattern 3: Content Sourced from Upstream READMEs (not recommended)

**What:** Some projects symlink or copy `benchmark/*/README.md` directly into `docs/`.

**When to use:** Never for this project.

**Trade-offs:** The upstream READMEs (`benchmark/nixlbench/README.md`, `benchmark/kvbench/README.md`) use raw GitHub Markdown conventions (bare anchor links, GitHub-flavored callouts, relative paths to source files). They are not valid MDX and will break the Fern build. Content must be reauthored as standalone Fern-compatible Markdown pages. Additionally, upstream READMEs mix installation, usage, and reference content on a single page — the docs site separates these into distinct pages for navigability.

## Data Flow

### Navigation Resolution (Fern Build)

```
fern/docs.yml
    ↓ (references)
docs/index.yml
    ↓ (path: entries resolve to)
docs/development/benchmarks/nixlbench/*.md
docs/development/benchmarks/kvbench/*.md
    ↓ (rendered as)
HTML pages at nixl.docs.buildwithfern.com/docs/...
```

Fern resolves all `path:` values in `docs/index.yml` relative to the `docs/` root. There is no import or include mechanism — each file is a self-contained page.

### Cross-Page Reference Pattern

Internal links use Fern's absolute path convention `/docs/<section>/<page>` (not relative filesystem paths). For example, a NIXLBench page referencing the ETCD User Guide page uses:

```markdown
[Metadata Exchange with ETCD](/docs/user-guide/etcd-metadata-exchange)
```

KVBench pages referencing NIXLBench should use:

```markdown
[NIXLBench](/docs/development/benchmarks/nixlbench)
```

## Integration Points

### Files to Modify

| File | Change | Notes |
|------|--------|-------|
| `docs/index.yml` | Add two `section:` blocks under Developer Guide | Insert after `building-a-backend-plugin.md` entry; both use `collapsed: open-by-default` to match existing style |

### New Files to Create

| File | Content Source | Notes |
|------|---------------|-------|
| `docs/development/benchmarks/nixlbench/index.md` | NIXLBench README Features + Quick Start sections | Overview, key capabilities, link to child pages |
| `docs/development/benchmarks/nixlbench/build.md` | NIXLBench README Building section | Docker build (recommended) + native build |
| `docs/development/benchmarks/nixlbench/usage.md` | NIXLBench README Usage section | ETCD setup, basic usage, config file format, multi-node patterns |
| `docs/development/benchmarks/nixlbench/cli-reference.md` | NIXLBench README Command Line Options section | Full flag reference, organized by category (core, memory, performance, storage, per-backend) |
| `docs/development/benchmarks/nixlbench/backend-examples.md` | NIXLBench README Backend-Specific Examples section | Per-backend command examples (UCX, GPUNETIO, GDS, POSIX, OBJ, GUSLI, etc.) |
| `docs/development/benchmarks/kvbench/index.md` | KVBench README Overview + Table of Contents | Overview, two command categories (KVBench vs CTP), supported LLM architectures |
| `docs/development/benchmarks/kvbench/build.md` | KVBench README Building section | Docker + Python venv install instructions |
| `docs/development/benchmarks/kvbench/commands.md` | KVBench README Command Descriptions + CLI Arguments | All five commands: plan, profile, kvcache, ct-perftest, sequential-ct-perftest |
| `docs/development/benchmarks/kvbench/llm-examples.md` | KVBench README Examples + Developer Guides content | DeepSeek R1 block/layer examples, CTP YAML configs, matrix generation, Slurm examples |

### No Changes Required

| File | Reason |
|------|--------|
| `fern/docs.yml` | Site config is complete; no new products, versions, or nav tabs needed |
| `fern/components/` | No new custom MDX components needed |
| `docs/assets/` | No new images or fonts needed |
| Any existing docs page | No existing content needs modification; benchmark pages are additive |

## Anti-Patterns

### Anti-Pattern 1: Top-Level Benchmarks Section

**What people do:** Add a "Benchmarks" entry at the same level as Getting Started, User Guide, Developer Guide.

**Why it's wrong:** NIXLBench and KVBench are developer tools for measuring NIXL performance — they are not user-facing product features. Elevating them to top-level implies equal importance to core documentation, which fragments the navigation hierarchy for all users. The PROJECT.md explicitly specifies Developer Guide as the target section.

**Do this instead:** Place both tools as subsections of Developer Guide, after "Building a Backend Plugin".

### Anti-Pattern 2: Flat Nine-Page Dump into `development/`

**What people do:** Add all nine new `.md` files directly into `docs/development/` without subdirectories.

**Why it's wrong:** The `development/` directory currently holds two files (`building-a-backend-plugin.md`, `sb-api-reference.md`). Adding nine more files destroys the scannability of the directory. The navigation grouping in `docs/index.yml` would also become a flat nine-entry list under Developer Guide with no visual hierarchy.

**Do this instead:** Group by tool under `docs/development/benchmarks/nixlbench/` and `docs/development/benchmarks/kvbench/`.

### Anti-Pattern 3: Single-Page Per Tool

**What people do:** Combine all NIXLBench content into one `nixlbench.md` file (mirroring the README structure).

**Why it's wrong:** The NIXLBench README is ~700 lines covering features, system requirements, building, usage, CLI reference, and backend examples. A single-page dump produces a doc that is hard to navigate and impossible to deep-link. The CLI reference alone covers nine backend-specific flag groups. Splitting into focused pages enables bookmarking, search result precision, and progressive disclosure.

**Do this instead:** Use the five-page split (index, build, usage, cli-reference, backend-examples).

### Anti-Pattern 4: Copying Raw README Verbatim

**What people do:** Copy-paste the README markdown directly into the docs file.

**Why it's wrong:** The READMEs contain GitHub-flavored Markdown that is not fully Fern/MDX compatible: bare anchor-only links (`[text](#heading)`), HTML comments, and relative paths to source files (`../../src/plugins/gusli/README.md`). These will either silently break or cause build errors. The READMEs also lack Fern frontmatter (`title:`, `description:`), which all existing docs pages include.

**Do this instead:** Reauthor content as Fern-compatible MDX with frontmatter, replacing relative links with absolute `/docs/` paths, and converting GitHub-flavored callouts to Fern `<Note>` components.

## Build Order Considerations

The nine new files can be created in any order since they have no build-time dependencies on each other. However, the recommended authoring order is:

1. Create directory structure (`docs/development/benchmarks/nixlbench/`, `docs/development/benchmarks/kvbench/`)
2. Update `docs/index.yml` with navigation entries (ensures nav is defined before content, allowing incremental preview)
3. Author NIXLBench pages (index → build → usage → cli-reference → backend-examples)
4. Author KVBench pages (index → build → commands → llm-examples)

The `docs/index.yml` change is the single file that gates Fern's ability to render any of the new pages. It should be the first substantive change made.

## Sources

- Observed: `docs/index.yml` — existing navigation tree structure (lines 52–68 for Developer Guide pattern)
- Observed: `docs/user-guide/building-nixl/index.md` — section landing page pattern with frontmatter
- Observed: `docs/development/building-a-backend-plugin.md` — flat page pattern with frontmatter
- Observed: `benchmark/nixlbench/README.md` — NIXLBench content scope (~700 lines, covering features, build, usage, CLI, backend examples)
- Observed: `benchmark/kvbench/README.md` — KVBench content scope (~430 lines, covering KVBench + CTP commands)
- Observed: `benchmark/kvbench/docs/` — supplementary KVBench developer guides (tutorial-gds.md, creating-a-model-config.md, adding-a-new-model-architecture.md, ct-perftest.md) — candidate content for llm-examples.md
- Observed: `fern/docs.yml` — Fern product/version config, confirms `docs/index.yml` is the navigation source

---
*Architecture research for: NIXL documentation site — NIXLBench and KVBench integration*
*Researched: 2026-04-07*
