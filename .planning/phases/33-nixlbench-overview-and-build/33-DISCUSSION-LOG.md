# Phase 33: NIXLBench Overview and Build - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-07
**Phase:** 33-NIXLBench Overview and Build
**Areas discussed:** Page structure, Build presentation, Content depth, Cross-linking strategy

---

## Page Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Combined index.md | Overview + build in one page, matches Phase 32 D-03 | |
| Separate overview + build.md | index.md = overview only, build.md = build instructions | ✓ |
| You decide | Claude picks based on content volume | |

**User's choice:** Separate overview + build.md
**Notes:** This requires updating Phase 32 nav entries (3 pages instead of 2 under NIXLBench).

---

| Option | Description | Selected |
|--------|-------------|----------|
| Features-first | Lead with what NIXLBench does, then requirements, then Next link | |
| Problem-first | Start with what problem it solves, then features as bullets, then requirements | ✓ |
| You decide | Claude picks layout | |

**User's choice:** Problem-first narrative
**Notes:** None

---

## Build Presentation

| Option | Description | Selected |
|--------|-------------|----------|
| Full options table | Include full build.sh options table from README | |
| Essentials only | Basic invocation + 2-3 common options, link to README for full table | ✓ |
| You decide | Claude determines detail level | |

**User's choice:** Essentials only
**Notes:** None

---

| Option | Description | Selected |
|--------|-------------|----------|
| Link only | Prerequisites callout box with link to Building NIXL from Source | |
| Brief summary + link | 1-2 sentence summary then link | |
| Custom | User: "remove build steps, mention in a sentence with a link to build" | ✓ |

**User's choice:** A sentence with a link, no repeated build steps
**Notes:** User explicitly said "let's remove the build steps. Maybe we can mention in a sentence with a link to build"

---

## Content Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Inline brief list | Short bullet list under overview | |
| Full requirements table | Full hardware + software table from README | |
| Separate section on build page | Put requirements on build page, not overview | ✓ |

**User's choice:** System requirements on the build page
**Notes:** Requirements are needed at build time, not overview time.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Adapt structure, rewrite prose | Use README info but rewrite for Fern doc style | ✓ |
| Minimal rewrite | Keep README prose mostly intact, just reformat | |
| You decide | Claude adapts based on style match | |

**User's choice:** Adapt structure, rewrite prose
**Notes:** None

---

## Cross-Linking Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| First-mention inline links | Link each backend to User Guide page on first mention per page | ✓ |
| Grouped 'See also' section | Collect all backend links at bottom | |
| You decide | Claude picks based on existing patterns | |

**User's choice:** First-mention inline links
**Notes:** Matches requirement NB-06 and existing NIXL docs convention.

---

## Claude's Discretion

- Exact frontmatter description text
- Whether to include "Next steps" link at bottom of overview
- Feature ordering in overview bullets
- Exact phrasing of NIXL prerequisite sentence

## Deferred Ideas

- Full build.sh options table — link to README instead
- NIXLBench CLI reference page — out of scope for v1.1
- Backend-specific deep-dive examples — deferred
