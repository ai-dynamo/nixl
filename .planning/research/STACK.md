# Stack Research

**Domain:** CLI tool documentation on Fern — benchmark tools with complex option tables, multi-step build instructions, and code-heavy examples
**Researched:** 2026-04-07
**Confidence:** HIGH (all findings verified against existing codebase; Fern component inventory drawn from live docs in `docs/`)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Fern MDX | Current (site already live) | Page authoring format | Already validated platform; all tooling and CI in place |
| Standard Markdown tables | N/A | CLI option reference tables | Fern renders GFM tables natively; no component wrapper needed for flat option lists |
| Fern `<Tabs>` + `<Tab>` | Built-in | Build method selection (Docker vs native) | Both tools have two install paths; tabs present alternatives without vertical scrolling or duplicated prose |
| Fern `<CodeBlocks>` | Built-in | Multi-language or multi-variant code blocks | Used on Quick Start for Python/C++/Rust; appropriate when one concept has multiple equivalent invocations |
| Fern callouts (`<Note>`, `<Tip>`, `<Warning>`) | Built-in | Inline contextual guidance | Already used consistently across all v1.0 pages; `<Warning>` for ETCD required/optional distinctions, `<Tip>` for workflow shortcuts |
| Fern `<Frame>` | Built-in | Image captions | Used on GDS example for sequence diagrams; applicable if architecture diagrams are added later |
| MDX snippets (`<Markdown src="..." />`) | Fern experimental | Shared content fragments | Used for backend env-var tables; applicable if CLI option groups appear in multiple pages |

### Supporting Patterns (not separate libraries)

| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| Fenced code blocks with `title=` attribute | Label code blocks with filename or context | Every CLI command block; every config file block — readers scan by title |
| Fenced code block with language `bash` | Shell commands | All `nixlbench` and `main.py` invocations |
| Fenced code block with language `toml` | TOML config files | NIXLBench `--config_file` examples |
| Fenced code block with language `yaml` | YAML config files | KVBench CTP configuration files, model config files |
| Fenced code block with language `text` | Tabular output / benchmark results | Terminal output from `kvcache` and `sequential-ct-perftest` commands — prevents false syntax highlighting |
| H2/H3 section hierarchy | CLI reference structure | Group options by category (Core, Memory, Performance, Backend-Specific) matching README section groupings |
| Front-matter `title` + `description` | Page metadata | Required on every page; Fern uses description for SEO and page subtitle |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `fern check` | Validate docs.yml and MDX before push | Run after adding new pages to index.yml |
| `fern docs dev` | Local preview server | Verify tab rendering and table layout before publishing |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Markdown tables for option reference | `<Accordion>` per option | Only if individual options need long prose descriptions; NIXLBench/KVBench options are terse enough for table rows |
| `<Tabs>` for Docker vs native build | Separate pages per build method | Separate pages only if build paths diverge significantly (>10 steps each); current paths are short enough for tabs |
| `<Note>` / `<Warning>` callouts | Inline bold text | Callouts for must-not-miss operational requirements (ETCD coordination, CUDA_VISIBLE_DEVICES); inline bold for softer hints |
| MDX snippets for shared option groups | Duplicated table content | Snippets only if the same option table appears verbatim on 2+ pages; NIXLBench and KVBench share `--backend` values but the tables differ enough to stay separate |
| `title=` attribute on code blocks | Inline comment headers | `title=` is visually distinct and matches existing site convention; don't mix approaches |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `<CodeBlocks>` for single-language examples | Adds tab UI overhead when there is only one language; NIXLBench is C++ CLI only, KVBench is Python CLI only | Plain fenced code block with `title=` attribute |
| Custom React components (new ones) | `fern/components/` only has `CustomFooter.tsx`; introducing new TSX increases maintenance surface with no DX gain | Fern built-in components cover all benchmark doc needs |
| `<Frame>` as a mandatory wrapper | GDS example uses it for SVG sequence diagrams; benchmark pages have no diagrams in source material | Use `<Frame>` only if architecture diagrams are added; don't wrap code output blocks |
| Option descriptions as prose paragraphs | Benchmark tools have 30+ options each; prose becomes unscannable | Markdown tables grouped by option category |
| Numbered steps written as plain paragraphs | Readers skip steps in prose; build instructions must be sequential and checkable | Use numbered lists (`1.`, `2.`, `3.`) or H3-level step headings for multi-command build sequences |
| A single "all options" mega-table | NIXLBench has 8 backend-specific option groups; one flat table is 50+ rows and unusable | Separate H3-level tables per group (Core, Memory/Transfer, Performance, Storage, per-backend) |

## Stack Patterns by Variant

**NIXLBench (C++ binary, complex CLI):**
- `<Tabs>` with titles "Docker (Recommended)" and "Native Build" for the build section
- H3 tables per option group (Core Configuration, Memory and Transfer, Performance and Threading, Storage Backends, Backend-Specific) — mirrors README structure that readers already know
- `<Warning>` for the ETCD requirement: network backends require ETCD; storage backends do not — this is a common stumbling block
- `<Note>` for the TOML config file equivalence (command-line args map 1:1 to config file keys)
- Backend-specific examples as H2 sections with `bash` code blocks using `title=` (e.g., `title="UCX: GPU-to-GPU"`)

**KVBench (Python CLI with subcommands):**
- `<Tabs>` with titles "Docker" and "Python (venv)" for the build section — KVBench's Docker path re-uses NIXLBench's container, worth a `<Note>` cross-link
- Subcommand reference as H2 sections (`plan`, `profile`, `kvcache`, `ct-perftest`, `sequential-ct-perftest`) rather than one flat list
- Common arguments table at top-level, then per-subcommand tables for subcommand-specific args — matches how Click CLI help is structured
- `<Warning>` for `CUDA_VISIBLE_DEVICES` requirement on CTP tests (already flagged as "Important note" in README)
- `yaml` language tag for all CTP YAML config file examples; `text` for tabular terminal output (kvcache table, ct-perftest results)

## Version Compatibility

| Component | Constraint | Notes |
|-----------|------------|-------|
| Fern MDX components | Current site version | All components (`<Note>`, `<Tip>`, `<Warning>`, `<Tabs>`, `<Tab>`, `<CodeBlocks>`, `<Frame>`, `<Markdown src>`) verified as live on site |
| `experimental.mdx-components` | Set to `./components` in docs.yml | Custom components path is configured; no changes to docs.yml needed for standard built-in components |
| Front-matter `title` + `description` | Required per Fern convention | Every existing page has both; benchmark pages must match |

## Sources

- Existing live docs (`docs/getting-started/quick-start.md`) — verified `<Tabs>`, `<Tab>`, `<CodeBlocks>`, `<Note>`, `<Tip>`, `<Warning>` syntax in production
- Existing live docs (`docs/user-guide/backends/ucx.md`) — verified Markdown table pattern for option reference
- Existing live docs (`docs/resources/environment-variables.md`) — verified large multi-group table structure
- Existing live docs (`docs/examples/gds-direct-storage.md`) — verified `<Frame>`, `<Note>`, `<Tip>` callout usage
- Existing live docs (`docs/user-guide/building-nixl/docker.md`) — verified `<Tip>` and `<Note>` for build guidance
- `fern/docs.yml` — verified `experimental.mdx-components`, layout, and component path
- `fern/snippets/env-vars-ucx.mdx` — verified `<Markdown src="..." />` snippet pattern
- `benchmark/nixlbench/README.md` — source material: 8 CLI option groups, backend-specific sections, TOML config, multi-step native build
- `benchmark/kvbench/README.md` — source material: 5 subcommands, common/per-command args, CTP YAML config, tabular terminal output
- [Fern components overview](https://buildwithfern.com/learn/docs/writing-content/components/overview) — HIGH confidence (verified callout types: Note, Tip, Warning, Error; CodeBlocks, Tabs, Tab confirmed)

---
*Stack research for: NIXLBench and KVBench documentation on Fern*
*Researched: 2026-04-07*
