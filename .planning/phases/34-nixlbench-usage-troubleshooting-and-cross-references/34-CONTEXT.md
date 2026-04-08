# Phase 34: NIXLBench Usage, Troubleshooting, and Cross-References - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Author the NIXLBench usage guide covering launching workers, ETCD coordination, all four communication patterns, and interpreting output. Include troubleshooting for common failure modes. Cross-link all backend mentions and ETCD to their existing User Guide pages. No new pages beyond `usage.md` — no separate troubleshooting page.

</domain>

<decisions>
## Implementation Decisions

### Page Structure
- **D-01:** Everything goes in a single `usage.md` page — usage guide at the top, troubleshooting section at the bottom. No separate `troubleshooting.md`. This keeps the NIXLBench nav at 3 pages (index.md, build.md, usage.md).
- **D-02:** Page sections (in order): ETCD Coordination (brief), Communication Patterns (with examples), CLI Options (essential flags), Troubleshooting.

### ETCD Coordination
- **D-03:** Use a `<Warning>` callout for the 60-second join window barrier. Link to the existing "Metadata Exchange with ETCD" User Guide page (`docs/user-guide/etcd-metadata-exchange.md`). Brief Docker one-liner for starting ETCD. Do NOT re-explain ETCD setup in detail. Satisfies requirement NB-05.
- **D-04:** Mention when ETCD is required vs optional (network backends require it; storage backends can run without it for single instances). Keep this concise — 2-3 sentences.

### Usage Examples
- **D-05:** Focus examples on the 4 communication patterns (pairwise, many-to-one, one-to-many, TP) using UCX as the default backend. Show the `--scheme` flag variations with brief explanation of each pattern.
- **D-06:** Add 1-2 storage backend examples (GDS for local storage, OBJ for S3) to demonstrate how storage benchmarks differ from network benchmarks. Link to backend User Guide pages for backend-specific flags rather than documenting them here.
- **D-07:** Show multi-node examples demonstrating initiator/target worker launching (two `nixlbench` commands on separate hosts pointing to the same ETCD server).

### CLI Options
- **D-08:** Claude's discretion on how many flags to document. Should include enough for a developer to run all 4 communication patterns and basic storage benchmarks. Skip per-backend flag tables — those belong in the backend docs or the full CLI reference (deferred).

### Troubleshooting
- **D-09:** Cover the 4 ROADMAP-required failure modes: ETCD connection failures, CUDA/GPU not found, backend library missing, build failures.
- **D-10:** Claude's discretion on whether to add runtime essentials (library-not-found errors, ETCD cleanup after failed runs). These are the two most common runtime issues from the README. Include if they fit naturally; skip if the page is already long enough.

### Cross-Linking
- **D-11:** Link every backend name (UCX, GDS, GDS_MT, POSIX, GPUNETIO, Mooncake, HF3FS, OBJ, GUSLI, Azure Blob) to its corresponding User Guide backend page on first mention per page. Satisfies NB-06. Backend pages exist at `docs/user-guide/backends/{backend}.md`.
- **D-12:** Link to "Metadata Exchange with ETCD" (`docs/user-guide/etcd-metadata-exchange.md`) when ETCD coordination is discussed. Satisfies NB-05.

### Claude's Discretion
- Exact CLI flag table scope (core only vs core + memory/transfer)
- Whether to include ETCD cleanup and library-not-found in troubleshooting
- Ordering of troubleshooting items
- Whether to include a config file example (TOML format)
- Whether to mention NVSHMEM worker type or keep focus on the default NIXL worker

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source material
- `benchmark/nixlbench/README.md` — Primary source for usage examples, CLI flags, ETCD coordination, troubleshooting steps, and backend-specific examples
- `benchmark/nixlbench/README.md` §Usage (line ~390) — Worker launching, ETCD setup, basic examples
- `benchmark/nixlbench/README.md` §Command Line Options (line ~432) — Full CLI flag reference (70+ flags)
- `benchmark/nixlbench/README.md` §Troubleshooting (line ~817) — Build and runtime troubleshooting

### Existing docs to cross-link
- `docs/user-guide/etcd-metadata-exchange.md` — ETCD metadata exchange guide (link target for NB-05)
- `docs/user-guide/backends/ucx.md` — UCX backend page
- `docs/user-guide/backends/gds.md` — GDS backend page
- `docs/user-guide/backends/obj.md` — OBJ (S3) backend page
- `docs/user-guide/backends/posix.md` — POSIX backend page
- `docs/user-guide/backends/gpunetio.md` — GPUNETIO backend page
- `docs/user-guide/backends/mooncake.md` — Mooncake backend page
- `docs/user-guide/backends/libfabric.md` — Libfabric backend page
- `docs/user-guide/backends/hf3fs.md` — HF3FS backend page
- `docs/user-guide/backends/gusli.md` — GUSLI backend page
- `docs/user-guide/backends/azure-blob.md` — Azure Blob backend page
- `docs/user-guide/backends/gds-mt.md` — GDS_MT backend page

### Doc patterns
- `docs/user-guide/building-nixl/docker.md` — Reference for `<Tabs>` and `<Warning>` callout usage in Fern MDX

### Requirements
- `.planning/REQUIREMENTS.md` — NB-03 (usage guide), NB-04 (troubleshooting), NB-05 (ETCD callout + link), NB-06 (backend cross-links)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `<Warning>` Fern component — used in existing docs for important callouts; use for the 60-second ETCD join window
- `<Tabs>` Fern component — available but not needed for this page (no Docker/native split on usage)
- Backend User Guide pages all follow the same pattern — consistent link targets

### Established Patterns
- All backend pages at `docs/user-guide/backends/` use consistent frontmatter and structure
- ETCD metadata exchange page exists as a standalone User Guide page — link to it, don't duplicate
- First-mention inline linking for backend names (carried from Phase 33 D-09)
- Fern frontmatter `title:` + `description:` on all pages

### Integration Points
- `usage.md` replaces the stub file created in Phase 32 at `docs/development/benchmarking/nixlbench/usage.md`
- No changes to `docs/index.yml` needed (usage.md entry already exists from Phase 32)

</code_context>

<specifics>
## Specific Ideas

- User wants the 4 communication patterns to be the centerpiece of the usage examples, not per-backend examples
- ETCD section should be a brief callout + link, not a full setup guide
- Troubleshooting should cover the ROADMAP-required 4 failure modes at minimum

</specifics>

<deferred>
## Deferred Ideas

- Full CLI reference (70+ flags) — out of scope for v1.1 per REQUIREMENTS.md
- Per-backend example pages — deferred per REQUIREMENTS.md
- Performance tuning guide (CPU affinity, network tuning) — not in ROADMAP scope
- Config file format documentation — Claude's discretion whether to include a brief mention

</deferred>

---

*Phase: 34-nixlbench-usage-troubleshooting-and-cross-references*
*Context gathered: 2026-04-07*
