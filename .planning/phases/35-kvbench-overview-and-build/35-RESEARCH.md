# Phase 35: KVBench Overview and Build - Research

**Researched:** 2026-04-07
**Domain:** Fern MDX documentation authoring (KVBench overview and build pages)
**Confidence:** HIGH

## Summary

Phase 35 creates two Fern-compatible MDX pages for KVBench: an overview page (`index.md`) and a build page (`build.md`). The source material is well-defined in `benchmark/kvbench/README.md`, and the output pattern is established by the Phase 33 NIXLBench pages. The primary work is content authoring -- rewriting README prose into Fern doc style, structuring the overview around the two command categories and NIXLBench subprocess relationship, and creating a tabbed Docker/Python venv build page.

The navigation config (`docs/index.yml`) must be updated to add a `build.md` entry under the KVBench section. The existing Phase 32 stub `index.md` will be replaced with real content.

**Primary recommendation:** Follow the NIXLBench page pattern exactly (frontmatter, cross-linking, Tabs component) and use `benchmark/kvbench/README.md` as the single source of truth for content.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Separate pages: `index.md` (overview) and `build.md` (build instructions). Consistent with NIXLBench Phase 33 pattern. Phase 32's `docs/index.yml` entries need updating to add a `build.md` entry.
- **D-02:** The first paragraph states that KVBench generates and runs `nixlbench` commands, with an inline link to the NIXLBench section. This satisfies requirement KB-01.
- **D-03:** The overview presents the two command categories (KVBench commands: plan/profile/kvcache and CTP commands: ct-perftest/sequential-ct-perftest) with brief one-line descriptions. Claude's discretion on visual layout.
- **D-04:** Mention supported LLM architectures (DeepSeek R1, Llama 3.1, and others) on the overview page as a feature highlight. No deep explanation -- that's Phase 36.
- **D-05:** Use `<Tabs>` component for Docker vs Python venv build paths, consistent with ROADMAP KB-02.
- **D-06:** Docker tab links to the NIXLBench build page rather than repeating Docker build steps. Brief note: "KVBench is included in the NIXLBench Docker container" + link to NIXLBench build page. No duplication.
- **D-07:** Python venv tab is self-contained -- shows the `venv` + `pip`/`uv` install steps directly since they're short and specific to KVBench.
- **D-08:** No system requirements section on the build page (KVBench's only requirements are Python 3.12+ and optional PyTorch for GPU benchmarks -- mention inline).

### Claude's Discretion
- Visual layout for the two command categories (grouped bullets vs two sections)
- Whether to include a "Supported Models" subsection on overview or just a brief mention
- Exact frontmatter `description:` text for each page
- Whether to include a "Next steps" link at the bottom of overview pointing to build page

### Deferred Ideas (OUT OF SCOPE)
- KVBench command reference (all 5 subcommands with CLI tables) -- Phase 36
- Model configuration YAML schema documentation -- Phase 36
- LLM architecture examples (DeepSeek R1, Llama 3.1) -- Phase 36
- CTP examples and matrix format documentation -- Phase 36
- KVBench GDS tutorial -- P2, deferred per REQUIREMENTS.md
- Adding new model architecture guide -- P2, deferred per REQUIREMENTS.md
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| KB-01 | KVBench overview page states in its first paragraph that `profile` invokes `nixlbench` as a subprocess (NIXLBench dependency), and covers what KVBench is and its two command categories (KVBench commands vs CTP commands) | Source content in `benchmark/kvbench/README.md` lines 26-32; NIXLBench page pattern in `docs/development/benchmarking/nixlbench/index.md`; cross-link target is `./nixlbench` relative path |
| KB-02 | KVBench build page covers Docker build (re-using NIXLBench container) and Python venv install, using `<Tabs>` for the two paths | Source content in `benchmark/kvbench/README.md` lines 38-56; Tabs pattern in `docs/development/benchmarking/nixlbench/build.md`; Docker tab links to NIXLBench build page, Python tab shows venv+uv steps |
</phase_requirements>

## Architecture Patterns

### Fern MDX Page Structure (Established Pattern)

Every doc page follows this structure, verified from the NIXLBench Phase 33 output:

```markdown
---
title: Page Title
description: One-sentence summary for SEO and navigation.
---

First paragraph introduces the tool/concept with inline cross-links on first mention.

## Section Heading

Content...

## Next Steps

- **[Link text](./relative-path)** -- Brief description
```

[VERIFIED: `docs/development/benchmarking/nixlbench/index.md`]

### Cross-Link Conventions

- Use relative Fern doc paths: `./nixlbench/build` (not full filesystem paths)
- Backend names link to their User Guide pages on first mention: `[UCX](/docs/user-guide/backends/ucx)`
- etcd links to: `/docs/user-guide/etcd-metadata-exchange`
- NIXLBench section root: referenced as a relative sibling path from KVBench pages

[VERIFIED: `docs/development/benchmarking/nixlbench/index.md` line 6-11]

### Tabs Component Pattern

The `<Tabs>` component is used in the NIXLBench build page and works as follows:

```markdown
<Tabs>
<Tab title="Docker">

Content for Docker tab...

</Tab>
<Tab title="Python venv">

Content for Python venv tab...

</Tab>
</Tabs>
```

Key rules:
- Blank line after `<Tab title="...">` and before `</Tab>`
- Markdown content inside tabs renders normally
- Code blocks inside tabs work without issues

[VERIFIED: `docs/development/benchmarking/nixlbench/build.md` lines 27-110]

### Navigation Config Pattern (docs/index.yml)

Current KVBench section (Phase 32 output):
```yaml
- section: KVBench
  collapsed: open-by-default
  path: development/benchmarking/kvbench/index.md
  contents:
    - page: Commands and Examples
      path: development/benchmarking/kvbench/commands.md
```

Must be updated to add `build.md`:
```yaml
- section: KVBench
  collapsed: open-by-default
  path: development/benchmarking/kvbench/index.md
  contents:
    - page: Building KVBench
      path: development/benchmarking/kvbench/build.md
    - page: Commands and Examples
      path: development/benchmarking/kvbench/commands.md
```

This mirrors the NIXLBench section structure where build comes before usage/commands.

[VERIFIED: `docs/index.yml` lines 79-84]

### Anti-Patterns to Avoid
- **GitHub-flavored anchor links (`[text](#heading)`):** Fern does not support these. Use Fern relative doc links instead.
- **HTML comments (`<!-- ... -->`):** Fern MDX does not strip HTML comments; they render as visible text.
- **Bare HTML tags:** Only Fern-provided components (`<Tabs>`, `<Tab>`, `<Warning>`, `<Note>`, etc.) are allowed.
- **Duplicating content from other pages:** Always cross-link instead (D-06 mandates this for Docker build steps).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tab switching UI | Custom HTML/JS tabs | `<Tabs>` / `<Tab>` Fern component | Fern renders these natively; custom HTML breaks |
| Callout boxes | Custom blockquote styling | `<Warning>`, `<Note>` Fern components | Consistent styling, accessible |
| Navigation structure | Manual sidebar links | `docs/index.yml` entries | Fern generates sidebar from this config |

## Common Pitfalls

### Pitfall 1: pyproject.toml says Python >=3.10 but README and docs say 3.12+
**What goes wrong:** Conflicting Python version requirements confuse users.
**Why it happens:** `pyproject.toml` has `requires-python = ">=3.10"` but the project README and NIXLBench build page both state Python 3.12+.
**How to avoid:** Follow the user decision D-08 which says "Python 3.12+" -- this aligns with the NIXLBench build page and is the documented minimum. Do not reference pyproject.toml's `>=3.10`.
**Warning signs:** Mentioning Python 3.10 anywhere in the docs.

[VERIFIED: `benchmark/kvbench/pyproject.toml` line 19 shows `>=3.10`; NIXLBench build.md line 23 shows "Python: 3.12 or later"]

### Pitfall 2: Forgetting the Blank Lines Around Tab Content
**What goes wrong:** Markdown inside `<Tab>` tags doesn't render as expected (headings become plain text, lists break).
**Why it happens:** Fern MDX requires blank lines after opening `<Tab>` and before closing `</Tab>` for markdown parsing to activate.
**How to avoid:** Always include blank lines as shown in the Tabs pattern above.
**Warning signs:** Content inside tabs appearing as unformatted text.

### Pitfall 3: Wrong Cross-Link Path Format
**What goes wrong:** Links to NIXLBench pages return 404 in the rendered docs.
**Why it happens:** Using filesystem-relative paths instead of Fern doc paths.
**How to avoid:** Use Fern doc link format: `/docs/development/benchmarking/nixlbench/build` (no `.md` extension in some contexts) or relative `./nixlbench/build`. Check existing cross-links in NIXLBench pages for the exact format used.
**Warning signs:** Any link containing `.md` extension or filesystem-style paths.

[VERIFIED: NIXLBench index.md uses `./nixlbench/build` format]

### Pitfall 4: Describing Commands in Too Much Detail on Overview Page
**What goes wrong:** Overview page becomes a mini-reference, duplicating Phase 36 content.
**Why it happens:** The README has extensive command documentation that's easy to over-include.
**How to avoid:** Stick to D-03: brief one-line descriptions per command. Full CLI tables and examples are Phase 36 scope.
**Warning signs:** Any CLI argument tables or multi-line command examples on the overview page.

## Code Examples

### KVBench Overview Page (index.md) Structure

```markdown
---
title: KVBench
description: [one-line description]
---

[First paragraph: KVBench generates and runs nixlbench commands for KV cache
transfer benchmarking. The `profile` command invokes [NIXLBench](link) as a
subprocess. Inline link to NIXLBench section.]

## Command Categories

### KVBench Commands
- **plan** -- one-line description
- **profile** -- one-line description
- **kvcache** -- one-line description

### CTP Commands
- **ct-perftest** -- one-line description
- **sequential-ct-perftest** -- one-line description

## Supported Models

Brief mention of DeepSeek R1, Llama 3.1, and others.

## Next Steps

- **[Building KVBench](./kvbench/build)** -- Docker and Python venv install
- **[Commands and Examples](./kvbench/commands)** -- Full command reference
```

[ASSUMED -- layout is Claude's discretion per CONTEXT.md; planner should finalize]

### KVBench Build Page (build.md) Structure

```markdown
---
title: Building KVBench
description: [one-line description]
---

KVBench requires Python 3.12 or later. For GPU benchmarks, PyTorch is also needed.

<Tabs>
<Tab title="Docker">

KVBench is included in the NIXLBench Docker container. See
[Building NIXLBench](/docs/development/benchmarking/nixlbench/build) for
Docker build instructions.

</Tab>
<Tab title="Python venv">

Clone the repository and set up a virtual environment:

\```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl/benchmark/kvbench
python3 -m venv venv
source venv/bin/activate
pip install uv
uv sync --active
\```

</Tab>
</Tabs>
```

[VERIFIED: build steps from `benchmark/kvbench/README.md` lines 49-56; Docker cross-link pattern from D-06]

## Source Content Mapping

| README Section | Maps To | Phase |
|----------------|---------|-------|
| Overview (lines 26-32) | `index.md` first paragraph + command categories | 35 |
| Supported LLM Architectures (lines 33-36) | `index.md` brief model mention | 35 |
| Building > Docker (lines 40-46) | `build.md` Docker tab (cross-link only) | 35 |
| Building > Python (lines 48-56) | `build.md` Python venv tab | 35 |
| Command Descriptions (lines 160-215) | `commands.md` | 36 (OUT OF SCOPE) |
| Command Line Arguments (lines 79-158) | `commands.md` | 36 (OUT OF SCOPE) |
| Examples (lines 217-405) | `commands.md` | 36 (OUT OF SCOPE) |

## Files to Create/Modify

| File | Action | Notes |
|------|--------|-------|
| `docs/development/benchmarking/kvbench/index.md` | **Replace** stub | Currently a placeholder from Phase 32 |
| `docs/development/benchmarking/kvbench/build.md` | **Create** new | Does not exist yet |
| `docs/index.yml` | **Modify** | Add `build.md` entry under KVBench section, before `commands.md` |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Fern CLI |
| Config file | `fern/fern.config.json` (assumed) |
| Quick run command | `cd fern && fern check` |
| Full suite command | `cd fern && fern check` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| KB-01 | Overview page has correct content structure | manual | Visual review of `index.md` content | Will exist after implementation |
| KB-02 | Build page has Docker/venv tabs | smoke | `cd fern && fern check` | Will exist after implementation |
| QS-04 | Fern-compatible MDX (no build errors) | smoke | `cd fern && fern check` | N/A |

### Sampling Rate
- **Per task commit:** `cd fern && fern check`
- **Per wave merge:** `cd fern && fern check`
- **Phase gate:** Full `fern check` green before `/gsd-verify-work`

### Wave 0 Gaps
None -- existing Fern infrastructure covers all phase requirements. `fern check` validates MDX syntax, frontmatter, and navigation config.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Overview page layout with "Command Categories" and "Supported Models" sections | Code Examples | Low -- layout is Claude's discretion per CONTEXT.md; easy to adjust |
| A2 | Cross-link format `./kvbench/build` for sibling page links | Code Examples | Medium -- may need `/docs/development/benchmarking/kvbench/build` format; verify against NIXLBench pattern |
| A3 | `fern check` is the correct validation command | Validation Architecture | Low -- standard Fern CLI command; used in prior phases |

## Open Questions

1. **Exact Fern cross-link path format for sibling pages**
   - What we know: NIXLBench index.md uses `./nixlbench/build` to link to its build page
   - What's unclear: Whether KVBench index.md should use `./kvbench/build` or just `./build` since they're in the same directory
   - Recommendation: Check what NIXLBench index.md does -- it uses `./nixlbench/build` which suggests the path is relative to the benchmarking section root, not the current file. Implementer should verify with `fern check`.

## Sources

### Primary (HIGH confidence)
- `benchmark/kvbench/README.md` -- Full source content for overview, build steps, command categories
- `benchmark/kvbench/pyproject.toml` -- Python version requirement (>=3.10), dependencies
- `docs/development/benchmarking/nixlbench/index.md` -- NIXLBench overview page pattern (Phase 33 output)
- `docs/development/benchmarking/nixlbench/build.md` -- NIXLBench build page pattern with Tabs component
- `docs/index.yml` lines 79-84 -- Current KVBench navigation config
- `benchmark/kvbench/examples/model_deepseek_r1.yaml` -- DeepSeek R1 model architecture fields
- `benchmark/kvbench/examples/model_llama_3_1_70b.yaml` -- Llama 3.1 70B model architecture fields

### Secondary (MEDIUM confidence)
- None

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no libraries needed; pure content authoring using established Fern MDX patterns
- Architecture: HIGH -- page structure directly mirrors Phase 33 NIXLBench output
- Pitfalls: HIGH -- verified against actual source files and existing pages

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (stable -- Fern MDX patterns unlikely to change)
