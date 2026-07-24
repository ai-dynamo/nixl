# Phase 33: NIXLBench Overview and Build - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Author the NIXLBench overview page and build page as Fern-compatible MDX. The overview explains what NIXLBench is and why developers would use it. The build page covers Docker and native build paths. No usage guide or troubleshooting — those are Phase 34.

</domain>

<decisions>
## Implementation Decisions

### Page Structure
- **D-01:** Separate pages: `index.md` (overview) and `build.md` (build instructions). This updates Phase 32's nav structure — NIXLBench section now has 3 child pages (`index.md`, `build.md`, `usage.md`) instead of 2. Phase 32's `docs/index.yml` entries need updating to add a `build.md` entry.
- **D-02:** The overview page (`index.md`) uses a problem-first narrative: lead with what problem NIXLBench solves (benchmarking distributed data transfers across backends), then list features (backends, communication patterns, memory types, ETCD coordination). System requirements are NOT on the overview page.

### Build Presentation
- **D-03:** The build page uses `<Tabs>` component for Docker vs Native build paths, consistent with ROADMAP success criteria NB-02.
- **D-04:** Docker build section shows essentials only: basic `build.sh` invocation + 2-3 most common options (e.g., `--build-type`, `--arch`). Link to the README for the full options table rather than reproducing all options.
- **D-05:** NIXL prerequisite is handled with a single sentence and link: "NIXLBench requires a NIXL installation — see [Building NIXL from Source](link)." No repeated NIXL build steps. No Prerequisites callout box — just inline text.
- **D-06:** System requirements (hardware + software) go on the build page, not the overview. Presented before the build tabs.

### Content Depth and Tone
- **D-07:** Adapt the README's information and structure but rewrite prose for the Fern docs site style — shorter paragraphs, consistent tone with existing NIXL docs, no GitHub-flavored constructs.
- **D-08:** The overview features list should be concise — grouped bullets (backends, patterns, memory, coordination) not a deep technical breakdown. Save detail for the build and usage pages.

### Cross-Linking Strategy
- **D-09:** Link each backend name (UCX, GDS, Mooncake, etc.) to its corresponding User Guide backend page on first mention per page. This satisfies requirement NB-06.
- **D-10:** Link to the "Metadata Exchange with ETCD" User Guide page when ETCD coordination is mentioned (but detailed ETCD callouts are Phase 34's scope).

### Claude's Discretion
- Exact frontmatter `description:` text for each page
- Whether to include a "Next steps" link at the bottom of overview pointing to build page
- Ordering of features in the overview bullets
- Exact phrasing of the NIXL prerequisite sentence

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source material
- `benchmark/nixlbench/README.md` — Primary source for NIXLBench features, system requirements, build instructions, and `build.sh` options
- `benchmark/nixlbench/contrib/build.sh` — Docker build script; verify documented options against actual flags
- `benchmark/nixlbench/contrib/Dockerfile` — Docker build context; verify base image and build stages

### Existing doc patterns (templates)
- `docs/user-guide/building-nixl/index.md` — Section index page pattern with frontmatter and sub-page links
- `docs/user-guide/building-nixl/docker.md` — Docker build page pattern; reference for Tabs component usage and tone
- `docs/development/building-a-backend-plugin.md` — Developer Guide single-page pattern

### Navigation config
- `docs/index.yml` — Must be updated to add `build.md` entry under NIXLBench section (Phase 32 only declared `index.md` + `usage.md`)

### Cross-link targets
- `docs/user-guide/building-nixl/` — "Building NIXL from Source" pages (link target for NIXL prerequisite)
- `docs/user-guide/backends/` — Individual backend pages (UCX, GDS, etc.) for first-mention inline links

### Requirements
- `.planning/REQUIREMENTS.md` — NB-01 (overview content), NB-02 (build page with Tabs, link to existing build docs)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/user-guide/building-nixl/docker.md` — Established pattern for `<Tabs>` Docker vs native build presentation
- `docs/user-guide/building-nixl/index.md` — Section index page with sub-page links; template for overview structure
- Fern frontmatter pattern: `title:` + `description:` used consistently across all existing pages

### Established Patterns
- All docs pages use YAML frontmatter with `title:` and `description:`
- No bare anchor links or HTML comments in Fern MDX
- Backend names are linked to their User Guide pages on first mention (existing convention in user-guide pages)
- `<Tabs>` component used for multi-path instructions (Docker/native/pip)

### Integration Points
- `docs/index.yml` — Must be updated to reflect 3 pages under NIXLBench instead of 2
- New files created: `docs/development/benchmarking/nixlbench/index.md`, `docs/development/benchmarking/nixlbench/build.md`
- The stub `index.md` from Phase 32 will be replaced with real content

</code_context>

<specifics>
## Specific Ideas

- User wants the overview to be problem-first narrative, not a feature dump
- User wants minimal duplication — NIXL build steps are a sentence + link, not repeated
- User wants build.sh essentials only on the page, link to README for full reference
- Content should be rewritten for doc site tone, not copy-pasted from README

</specifics>

<deferred>
## Deferred Ideas

- Full `build.sh` options table (70+ flags) — link to README instead
- NIXLBench CLI reference page — out of scope for v1.1 per REQUIREMENTS.md
- Backend-specific deep-dive examples — deferred per REQUIREMENTS.md
- System requirements could eventually get their own section if content grows

</deferred>

---

*Phase: 33-nixlbench-overview-and-build*
*Context gathered: 2026-04-07*
