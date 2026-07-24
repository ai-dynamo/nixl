# Phase 35: KVBench Overview and Build - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Author the KVBench overview page and build page as Fern-compatible MDX. The overview states that KVBench's `profile` command invokes `nixlbench` as a subprocess and explains the two command categories (KVBench commands vs CTP commands). The build page covers Docker (reusing NIXLBench container) and Python venv install. No command reference or examples — those are Phase 36.

</domain>

<decisions>
## Implementation Decisions

### Page Structure
- **D-01:** Separate pages: `index.md` (overview) and `build.md` (build instructions). Consistent with NIXLBench Phase 33 pattern. This updates Phase 32's nav structure — KVBench section now has 3 child pages (`index.md`, `build.md`, `commands.md`) instead of 2. Phase 32's `docs/index.yml` entries need updating to add a `build.md` entry.

### Overview Content
- **D-02:** The first paragraph states that KVBench generates and runs `nixlbench` commands, with an inline link to the NIXLBench section. This satisfies requirement KB-01 ("states in its first paragraph that profile invokes nixlbench as a subprocess").
- **D-03:** The overview presents the two command categories (KVBench commands: plan/profile/kvcache and CTP commands: ct-perftest/sequential-ct-perftest) with brief one-line descriptions. Claude's discretion on visual layout (grouped bullets, two sections, etc.).
- **D-04:** Mention supported LLM architectures (DeepSeek R1, Llama 3.1, and others) on the overview page as a feature highlight. No deep explanation — that's Phase 36.

### Build Page
- **D-05:** Use `<Tabs>` component for Docker vs Python venv build paths, consistent with ROADMAP KB-02.
- **D-06:** Docker tab links to the NIXLBench build page rather than repeating Docker build steps. Brief note: "KVBench is included in the NIXLBench Docker container" + link to NIXLBench build page. No duplication.
- **D-07:** Python venv tab is self-contained — shows the `venv` + `pip`/`uv` install steps directly since they're short and specific to KVBench.
- **D-08:** No system requirements section on the build page (KVBench's only requirements are Python 3.12+ and optional PyTorch for GPU benchmarks — mention inline).

### Claude's Discretion
- Visual layout for the two command categories (grouped bullets vs two sections)
- Whether to include a "Supported Models" subsection on overview or just a brief mention
- Exact frontmatter `description:` text for each page
- Whether to include a "Next steps" link at the bottom of overview pointing to build page

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source material
- `benchmark/kvbench/README.md` — Primary source for KVBench overview, build instructions, command descriptions, and examples
- `benchmark/kvbench/README.md` §Overview (line ~26) — Two command categories explanation
- `benchmark/kvbench/README.md` §Building (line ~38) — Docker and Python build steps
- `benchmark/kvbench/pyproject.toml` — Python dependencies and project metadata

### Model examples (for overview mention)
- `benchmark/kvbench/examples/model_deepseek_r1.yaml` — DeepSeek R1 model architecture config
- `benchmark/kvbench/examples/model_llama_3_1_70b.yaml` — Llama 3.1 70B model architecture config

### Existing doc patterns
- `docs/development/benchmarking/nixlbench/index.md` — NIXLBench overview page pattern (Phase 33 output)
- `docs/development/benchmarking/nixlbench/build.md` — NIXLBench build page pattern (Phase 33 output)

### Navigation config
- `docs/index.yml` — Must be updated to add `build.md` entry under KVBench section (Phase 32 only declared `index.md` + `commands.md`)

### Cross-link targets
- NIXLBench section pages — link target for the subprocess dependency explanation
- NIXLBench build page — link target for Docker build instructions

### Requirements
- `.planning/REQUIREMENTS.md` — KB-01 (overview with nixlbench dependency), KB-02 (build with Docker + venv Tabs)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- NIXLBench `index.md` and `build.md` (Phase 33 output) — Template for consistent page structure
- `<Tabs>` Fern component — used in NIXLBench build page, reuse for Docker/Python tabs
- Fern frontmatter pattern: `title:` + `description:` on all pages

### Established Patterns
- Cross-link to existing docs rather than duplicating (carried from Phase 33 D-05)
- First-mention inline links for backend names (carried from Phase 33 D-09)
- Problem-first narrative on overview pages (carried from Phase 33 D-02)
- Rewrite README prose for Fern doc style (carried from Phase 33 D-07)

### Integration Points
- `docs/index.yml` — Must be updated to reflect 3 pages under KVBench instead of 2
- New files: `docs/development/benchmarking/kvbench/index.md`, `docs/development/benchmarking/kvbench/build.md`
- The stub `index.md` from Phase 32 will be replaced with real content

</code_context>

<specifics>
## Specific Ideas

- User wants parallel structure with NIXLBench (separate overview + build pages)
- NIXLBench dependency stated naturally in first paragraph, not as a callout box
- Docker build links to NIXLBench build page — zero duplication

</specifics>

<deferred>
## Deferred Ideas

- KVBench command reference (all 5 subcommands with CLI tables) — Phase 36
- Model configuration YAML schema documentation — Phase 36
- LLM architecture examples (DeepSeek R1, Llama 3.1) — Phase 36
- CTP examples and matrix format documentation — Phase 36
- KVBench GDS tutorial (`benchmark/kvbench/docs/tutorial-gds.md`) — P2, deferred per REQUIREMENTS.md
- Adding new model architecture guide — P2, deferred per REQUIREMENTS.md

</deferred>

---

*Phase: 35-kvbench-overview-and-build*
*Context gathered: 2026-04-07*
