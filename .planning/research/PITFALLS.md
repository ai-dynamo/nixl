# Pitfalls Research

**Domain:** Adding benchmark/CLI tool documentation (NIXLBench + KVBench) to an existing Fern docs site
**Researched:** 2026-04-07
**Confidence:** HIGH — based on direct inspection of source READMEs, existing docs structure, `docs/index.yml`, and five diagnosed debug files from prior phases

---

## Critical Pitfalls

### Pitfall 1: Duplicating Build Instructions Already in "Building NIXL from Source"

**What goes wrong:**
The NIXLBench Docker build requires cloning the NIXL repo and running `./build.sh` from `benchmark/nixlbench/contrib`. The existing docs already have a full "Building NIXL from Source" section (Docker, C++/Meson, Python Bindings, Rust Bindings) at `user-guide/building-nixl/`. Writers copy the CUDA Toolkit install steps, apt dependency lists, UCX build-from-source block, and Docker installation commands verbatim into the new NIXLBench page instead of linking.

**Why it happens:**
The README contains a complete, self-contained build guide spanning ~350 lines. A first draft that converts it to docs without auditing what already exists in the published site will copy everything. The overlap is not obvious unless you read both documents side-by-side.

**How to avoid:**
Before writing, do an explicit cross-reference audit: for each build step in the README, check whether an equivalent page already exists. The overlap map is:

| NIXLBench README section | Existing doc page |
|---|---|
| Docker Installation (Ubuntu/RHEL) | Link to `user-guide/building-nixl/docker` |
| CUDA Toolkit Installation | Link to `user-guide/building-nixl/nixl-cpp` or Quick Start |
| UCX Build from Source | Link to backend page `user-guide/backends/ucx` |
| Building NIXL (meson setup build) | Link to `user-guide/building-nixl/nixl-cpp` |

The NIXLBench page should document only what is unique: the `etcd-cpp-api` installation, the NIXLBench-specific `meson setup build -Dnixl_path=...` invocation, and the `build.sh` container build options table. Everything else is a link.

**Warning signs:**
- NIXLBench build page exceeds ~80 lines of bash for prerequisites
- The page contains `sudo apt-get install -y build-essential cmake ninja-build`
- The page contains `wget https://developer.download.nvidia.com/compute/cuda/`

**Phase to address:**
NIXLBench page authoring phase — before writing, not during review.

---

### Pitfall 2: Duplicating ETCD Coordination Setup

**What goes wrong:**
NIXLBench uses ETCD for worker coordination, and the README contains a full ETCD setup section including the `docker run quay.io/coreos/etcd` command and a `sudo apt install etcd-server` block. The existing site already has a detailed "Metadata Exchange with etcd" page at `user-guide/etcd-metadata-exchange.md` covering prerequisites, environment variables, and deployment patterns. New NIXLBench docs reproduce ETCD setup inline rather than linking to the existing page.

**Why it happens:**
NIXLBench uses ETCD differently (for worker coordination/barrier synchronization) than the metadata exchange page describes (for agent metadata publication). Writers see a difference in purpose and assume duplication is justified. In practice the setup steps (running an etcd server, pointing clients at port 2379) are identical and readers benefit from a single authoritative source.

**How to avoid:**
The NIXLBench doc should include a short note explaining why ETCD is required (worker coordination/barrier), then link to the existing ETCD page for server setup. The only NIXLBench-specific ETCD content is the `--etcd_endpoints` flag and the 60-second barrier timeout behavior — those belong on the NIXLBench page. Everything else is a cross-reference.

**Warning signs:**
- NIXLBench page contains `docker run ... quay.io/coreos/etcd` command
- NIXLBench page contains `sudo apt install etcd-server`
- The phrase "Start ETCD server" appears as a heading on the NIXLBench page with a full fenced-code block

**Phase to address:**
NIXLBench page authoring phase — define a "link, don't duplicate" policy for ETCD setup content before writing.

---

### Pitfall 3: Terminology Drift from Established Standards

**What goes wrong:**
The existing docs use specific terms established across phases 21–27. Both READMEs and the resulting docs introduce inconsistent variants. The most common drifts from actual source material:

| Established term (existing docs) | README / wrong variant |
|---|---|
| `plug-in` (hyphenated) | `plugin` (no hyphen) in README prose |
| `etcd` (lowercase) | `ETCD` (all caps) in README headings and prose |
| `GPUDirect Storage` | `GDS` used as a standalone noun in README |
| "backend" (noun) | "Backend" (capitalized mid-sentence) |
| `DRAM` / `VRAM` | Same — these are consistent already |

When new pages use the README's capitalization and hyphenation without normalizing to the established style, readers see inconsistency and editors accumulate a growing gap to fix.

**Why it happens:**
The README was written by engineers for engineers; it was never copy-edited for the docs style. Converting it to docs content is a translation task, not just a formatting task. Writers who translate quickly miss style-level decisions.

**How to avoid:**
Run a terminology normalization pass as a dedicated step after first-draft writing, not during it. The rules to apply:
- Replace `plugin` with `plug-in` everywhere in prose (code examples keep `--worker_type nixl` as-is — flag names are not prose)
- Replace `ETCD` (all-caps) with `etcd` in prose and headings; keep `ETCD` only when referencing the `--runtime_type ETCD` flag value
- Replace standalone `GDS` as noun with `GPUDirect Storage (GDS)` on first use per page, then `GDS` thereafter
- Do not capitalize "backend" mid-sentence unless it is part of a proper name like "UCX backend"

**Warning signs:**
- Search new files for `\bplugin\b` (no hyphen) in prose sentences
- Search for `\bETCD\b` outside of code blocks or flag value context
- A heading like "ETCD Coordination Setup" rather than "etcd Coordination Setup"

**Phase to address:**
Terminology normalization pass — a dedicated review step at the end of each page's drafting, before the phase is marked complete.

---

### Pitfall 4: Overly Complex Page Structures That Bury the CLI Reference

**What goes wrong:**
Both NIXLBench and KVBench have large CLI flag surfaces. NIXLBench has 8 flag groups (core, memory, performance, device/network, storage, backend-specific for 7 backends). KVBench has 6 flag groups (common, CLI override, plan-specific, shared benchmark, CTP-specific). When this is all placed on a single "Usage" page or squeezed into a "Getting Started" page, the result is a 2,000+ line page that is exhausting to scan. Readers who need to find `--gds_batch_pool_size` cannot.

**Why it happens:**
The README is linear (top-to-bottom scroll) which works for a GitHub README. Fern docs are navigation-tree based; long single-page docs feel like reading a manual. Writers convert README sections 1:1 to a single page without considering how readers approach docs (goal-directed, not top-to-bottom).

**How to avoid:**
Plan the page hierarchy before writing. Recommended split:
- NIXLBench: Overview, Build, Usage Guide (ETCD setup + basic patterns), CLI Reference (full flags table), Examples (one page, or backend-specific sub-pages linked from the nav)
- KVBench: Overview, Build/Install, Command Reference (plan/profile/kvcache/ct-perftest as sections), Examples

The CLI Reference pages should be pure reference (tables, no prose paragraphs). The Usage Guide page should have short prose + examples. Keep them separate so readers can bookmark the reference.

**Warning signs:**
- A single NIXLBench page exceeds 300 lines in the `.md` file
- The words "Overview", "Build", "Usage", and "CLI Reference" all appear as `##` headings on the same page
- The page's `## Command Line Options` section has 7+ sub-sections with their own `###` headings

**Phase to address:**
Page structure planning phase — define the page hierarchy and nav entries in `docs/index.yml` before writing content.

---

### Pitfall 5: Navigation Conflicts in `docs/index.yml`

**What goes wrong:**
New pages are added to `docs/index.yml` with incorrect nesting, duplicate slugs, or placement outside the intended "Developer Guide" section. Common specific mistakes:
1. Adding NIXLBench/KVBench as top-level sections instead of entries under "Developer Guide"
2. Using a `section:` node for NIXLBench without providing a `path:` (index page) — Fern renders the section title as non-clickable but still expects an index page
3. Placing `collapsed: open-by-default` on NIXLBench sub-pages when they should be on the parent section node
4. Using a file path that does not match the actual file location (e.g., `path: developer-guide/nixlbench.md` when the file is at `docs/developer-guide/nixlbench/overview.md`)

**Why it happens:**
`docs/index.yml` uses a specific Fern schema where `section:` with `contents:` requires either a `path:` (index page) or no path (section-title-only). The existing entries show the correct pattern (e.g., `Building NIXL from Source` uses `path: user-guide/building-nixl/index.md` with `collapsed: open-by-default`), but writers unfamiliar with this pattern write new entries inconsistently.

**How to avoid:**
Copy the exact structure of the existing "Building NIXL from Source" section as the template for NIXLBench and KVBench sections. The pattern is:
```yaml
- section: NIXLBench
  collapsed: open-by-default
  path: developer-guide/nixlbench/index.md
  contents:
    - page: Overview
      path: developer-guide/nixlbench/overview.md
    - page: Build
      path: developer-guide/nixlbench/build.md
```
Verify each `path:` entry actually exists on disk before marking the phase complete.

**Warning signs:**
- `fern docs dev` throws a "page not found" or "could not resolve path" error on startup
- A section heading in the sidebar is not clickable (missing `path:` on a `section:` node)
- The "Developer Guide" section in the sidebar suddenly has no children (path conflict collapsed the tree)

**Phase to address:**
Navigation setup phase — add `docs/index.yml` entries as the first task of each page-set phase, before writing content, so `fern docs dev` can validate the structure.

---

### Pitfall 6: Stale CLI Reference Tables After Source Changes

**What goes wrong:**
The NIXLBench README documents 70+ flags across 8 groups. KVBench documents 30+ arguments across 6 groups. CLI reference tables in the docs become stale as the tools evolve — new flags are added (e.g., `--sequential-ct-perftest` was added to KVBench after `ct-perftest`), defaults change, or flags are renamed. The docs reflect the README at the time of writing, not the current binary.

**Why it happens:**
There is no automated mechanism linking the docs CLI tables to the actual binary. The source of truth is the `--help` output, but documentation is hand-authored from the README. When the README is updated, the docs are not automatically updated.

**How to avoid:**
Two strategies:
1. During authoring, validate the CLI reference by running `nixlbench --help` and `python main.py --help` (and each subcommand) and comparing against the README. Mark any divergence.
2. In the docs, add a note like "Run `nixlbench --help` for the canonical flag list — this page reflects version X.Y." This sets reader expectations and reduces the cost of minor staleness.

Do not copy-paste the CLI tables directly from the README without running `--help` to validate. The README itself has known minor discrepancies (e.g., the KVBench README lists `--etcd-endpoints` while NIXLBench README uses `--etcd_endpoints` — underscore vs hyphen difference between Python CLI and C++ binary).

**Warning signs:**
- CLI reference table was copy-pasted from the README without a `--help` validation step in the task checklist
- The KVBench `--etcd-endpoints` flag (hyphen, Python style) appears in the NIXLBench CLI reference (which uses underscore)
- A flag appears in the table that does not appear in `--help` output

**Phase to address:**
CLI Reference authoring phase — include a `--help` validation step as an explicit checklist item before marking the phase complete.

---

### Pitfall 7: Missed Cross-References to Existing Backend Pages

**What goes wrong:**
NIXLBench and KVBench both support all NIXL backends (UCX, GDS, GDS_MT, POSIX, HF3FS, OBJ, GPUNETIO, Mooncake, Libfabric, GUSLI, AZURE_BLOB). The backend-specific examples in the NIXLBench docs explain how to run the benchmark against each backend, but do not link to the corresponding backend page in "User Guide: NIXL Backends". Readers who hit a GDS configuration error have no obvious path to the GDS backend page where prerequisites and configuration are documented.

**Why it happens:**
The README was written without hyperlinks to the broader NIXL docs ecosystem (it predates the docs site). Converting it to docs without adding cross-references misses the primary advantage of a multi-page docs site over a flat README.

**How to avoid:**
For every backend mentioned in the NIXLBench or KVBench examples, add a cross-reference on first use. The map is:

| Backend in benchmark docs | Link target |
|---|---|
| UCX | `user-guide/backends/ucx` |
| GDS / GPUDirect Storage | `user-guide/backends/gds` |
| GDS_MT | `user-guide/backends/gds-mt` |
| POSIX | `user-guide/backends/posix` |
| HF3FS | `user-guide/backends/hf3fs` |
| OBJ (S3) | `user-guide/backends/obj` |
| GPUNETIO / DOCA GPUNetIO | `user-guide/backends/gpunetio` |
| Mooncake | `user-guide/backends/mooncake` |
| Libfabric | `user-guide/backends/libfabric` |
| GUSLI | `user-guide/backends/gusli` |
| Azure Blob | `user-guide/backends/azure-blob` |

Also link ETCD coordination back to `user-guide/etcd-metadata-exchange` on first mention.

**Warning signs:**
- Backend names appear in the benchmark docs without a hyperlink on first use
- The phrase "see the backend documentation" appears without a link target
- GDS or POSIX examples appear without a prerequisite note pointing to the backend page

**Phase to address:**
Cross-reference audit phase — a dedicated pass after first-draft writing, specifically checking each backend name for a link.

---

### Pitfall 8: KVBench Described as a Benchmark When It Generates Commands for NIXLBench

**What goes wrong:**
KVBench's `profile` command does run NIXLBench internally, but the primary purpose of KVBench is to generate `nixlbench` commands configured for specific LLM architectures. Writing the KVBench docs as if it is a standalone benchmark tool obscures the relationship: KVBench is a command generator and configuration helper that wraps NIXLBench. Readers who only read the KVBench docs may not know they need NIXLBench installed.

**Why it happens:**
The KVBench README title says "A comprehensive utility for generating NIXL Bench commands" — this is accurate. But the `profile` command description says "actually runs the benchmark with nixlbench, collecting performance data" which sounds standalone. Writers see both descriptions and pick the more exciting framing.

**How to avoid:**
The KVBench overview page must explicitly state: KVBench is a Python utility that generates and optionally executes `nixlbench` commands. The `profile` command invokes `nixlbench` as a subprocess — NIXLBench must be installed and on the `PATH`. Link to the NIXLBench build page as a prerequisite.

**Warning signs:**
- The KVBench overview describes KVBench as a "performance testing tool" without mentioning its dependency on NIXLBench
- The KVBench build page does not list NIXLBench as a prerequisite for `profile` command usage
- No link from KVBench pages to the NIXLBench docs

**Phase to address:**
KVBench overview page authoring — the NIXLBench dependency relationship must be established in the first paragraph of the overview, not discovered by readers in a footnote.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|---|---|---|---|
| Copy README directly as doc page | Fast first draft | Terminology drift, duplication, no cross-references — requires a full rewrite later | Never — always normalize in the first pass |
| Put all CLI flags on one page | Simpler nav tree | 2000+ line pages readers cannot scan; CLI reference and usage guide mixed together | Never for this volume of flags |
| Skip `--help` validation, trust README | Saves ~30 minutes | Stale tables, flag name mismatches between tools (underscore vs hyphen) | Never — this is a one-time investment per phase |
| Defer cross-references to "a later cleanup phase" | Faster writing | Cross-references are forgotten; orphaned pages become the norm | Never — add while writing, not after |
| Use same page for overview and build instructions | One fewer page to maintain | Overview pages become 50% build docs; readers who already built NIXL cannot skip to usage | Acceptable only if the build is trivially short (< 10 lines) |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|---|---|---|
| Fern `docs/index.yml` nav | Adding a `section:` with `contents:` but no `path:` causes a non-clickable section title | Always provide `path:` pointing to an index page when using `section:` with children; copy the "Building NIXL from Source" pattern |
| Fern local dev (`fern docs dev`) | Testing edit-this-page, search, and Ask AI features in local dev — all are disabled | Test server-dependent features via `fern generate --docs --preview` (preview links) only; never report a bug from local dev observations alone (confirmed by debug files) |
| KVBench `profile` invoking NIXLBench | Documenting `profile` without noting that NIXLBench binary must be on `PATH` | State the subprocess dependency explicitly; link to NIXLBench install page from KVBench prerequisites |
| ETCD coordination (NIXLBench) | Writing a full ETCD setup guide in the NIXLBench page, duplicating `user-guide/etcd-metadata-exchange` | Link to the existing ETCD page; only document NIXLBench-specific behavior (60s barrier timeout, `--etcd_endpoints` flag) |
| Backend flag values in CLI tables | Mixing Python CLI style (`--etcd-endpoints`, hyphen) with C++ binary style (`--etcd_endpoints`, underscore) in the same reference table | Use the exact flag string from `--help` output; NIXLBench uses underscores, KVBench uses hyphens — keep them separate |

---

## Performance Traps

Not applicable to documentation authoring — this domain has no runtime performance concerns.

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---|---|---|
| Including real AWS credentials or S3 secrets in NIXLBench examples | Credentials committed to the GitLab repo | Use `<access_key>`, `<secret_key>` placeholders in all examples; the README already uses environment variable form (`AWS_ACCESS_KEY_ID`) — follow that pattern |
| Documenting `--obj_secret_key` with an example value | Same as above | Placeholder only; add a note to use environment variables instead of CLI flags for secrets |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---|---|---|
| All backend-specific CLI options on one monolithic page | Readers trying to configure the GUSLI backend must scroll past GDS, POSIX, HF3FS, OBJ, and Azure Blob sections | Group backend-specific options with a clear anchor or separate page per backend group; use Fern `<Tabs>` component for backend variants if on one page |
| KVBench examples only show DeepSeek R1 and LLaMA 3.1 | Readers with other architectures cannot extrapolate | Explicitly document that `--model` accepts any YAML following the model config schema; link to the "Creating a Model Configuration" developer guide |
| NIXLBench GUSLI `--device_list` format undocumented in prose | The `id:type:path` format (e.g., `11:F:./store0.bin`) is cryptic without a schema explanation | Add a Device Types table (`F` = file, `K` = kernel block device, `N` = networked server with `t`/`u` prefix) as shown in the README's GUSLI Device Types section |
| CTP YAML config format not cross-referenced from CLI docs | Readers know the `ct-perftest` command exists but cannot find the YAML config schema | The CLI reference for `ct-perftest` must link to or inline the YAML configuration schema (matrix file format, traffic pattern fields) |

---

## "Looks Done But Isn't" Checklist

- [ ] **NIXLBench build page:** Confirm it does NOT reproduce Docker installation steps, CUDA toolkit install, or UCX build-from-source — these must be links to existing pages.
- [ ] **ETCD setup content:** Confirm `docker run quay.io/coreos/etcd` command does NOT appear on the NIXLBench page — must be a link to `user-guide/etcd-metadata-exchange`.
- [ ] **KVBench overview:** Confirm the first paragraph explicitly states KVBench generates/invokes `nixlbench` commands and links to the NIXLBench pages.
- [ ] **CLI reference validation:** Confirm `--help` output was run for both binaries and compared against the tables before the phase was marked complete.
- [ ] **Terminology pass:** Run `grep -rn '\bplugin\b' docs/developer-guide/` and `grep -rn '\bETCD\b' docs/developer-guide/` — both should return zero matches in prose (code blocks exempt).
- [ ] **Cross-references:** Every backend name (UCX, GDS, GDS_MT, POSIX, HF3FS, OBJ, GPUNETIO, Mooncake, Libfabric, GUSLI, Azure Blob) has a link on first use.
- [ ] **Nav validation:** `fern docs dev` starts without errors after each `docs/index.yml` edit.
- [ ] **Page length check:** No single docs page exceeds ~250 lines (overview, build, usage, and CLI reference are separate files).
- [ ] **Secrets:** No real credentials appear in any example — all use `<placeholder>` or environment variable form.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---|---|---|
| Duplication discovered post-publish | MEDIUM | Audit overlapping content, convert duplicated sections to links, verify no information is lost in the source page before removing it |
| Terminology drift found in review | LOW | Sed-pass on new files only; do not modify existing pages unless a separate terminology phase is scoped |
| Navigation conflict (Fern build failure) | LOW | Check `docs/index.yml` path values against filesystem; Fern error messages include the unresolved path |
| Stale CLI table discovered by user | LOW–MEDIUM | Run `--help`, diff against table, update affected rows; add a version note to the page header |
| KVBench/NIXLBench relationship unclear to readers | MEDIUM | Rewrite KVBench overview page; add a "How it works" diagram or callout box explaining the subprocess relationship |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---|---|---|
| Duplicating build instructions | NIXLBench Build page authoring | Grep new page for `apt-get install build-essential` — must be zero matches |
| Duplicating ETCD setup | NIXLBench Usage page authoring | Grep new page for `quay.io/coreos/etcd` — must be zero matches |
| Terminology drift | Terminology normalization pass (end of each page phase) | `grep -rn '\bplugin\b\|\bETCD\b' docs/developer-guide/` in prose context |
| Overly complex page structure | Navigation planning phase (before writing) | Page count for NIXLBench >= 4 separate files (overview, build, usage, CLI ref) |
| Navigation conflicts | Nav setup as first task of each page phase | `fern docs dev` starts cleanly after each `docs/index.yml` change |
| Stale CLI reference | CLI reference phase — `--help` validation step | Checklist item: "ran `nixlbench --help` and `python main.py [cmd] --help` and compared all flags" |
| Missed backend cross-references | Cross-reference audit (dedicated pass after first draft) | Every backend name is a hyperlink on first use |
| KVBench/NIXLBench relationship | KVBench overview authoring | First paragraph of overview explicitly names NIXLBench as a dependency |

---

## Sources

- Direct inspection: `benchmark/nixlbench/README.md` (v1.1, 2026-04-07)
- Direct inspection: `benchmark/kvbench/README.md` (v1.1, 2026-04-07)
- Direct inspection: `docs/index.yml` (current nav structure)
- Direct inspection: `docs/user-guide/building-nixl/index.md` (existing build docs)
- Direct inspection: `docs/user-guide/etcd-metadata-exchange.md` (existing ETCD docs)
- Direct inspection: `.planning/debug/fern-local-dev-limitations.md` (confirmed: search, edit-this-page, ask-ai all disabled in `fern docs dev`)
- Direct inspection: `.planning/debug/edit-this-page-not-visible.md` and `edit-this-page-round2.md`
- Direct inspection: `.planning/debug/search-bar-no-entry-symbol.md`
- Direct inspection: `.planning/debug/page-actions-toolbar-layout.md`
- Direct inspection: `.planning/PROJECT.md` (milestone context, established conventions)

---
*Pitfalls research for: Adding NIXLBench and KVBench documentation to existing NIXL Fern docs site*
*Researched: 2026-04-07*
