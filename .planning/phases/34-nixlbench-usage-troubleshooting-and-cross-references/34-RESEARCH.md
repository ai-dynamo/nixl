# Phase 34: NIXLBench Usage, Troubleshooting, and Cross-References - Research

**Researched:** 2026-04-07
**Domain:** Technical documentation (Fern MDX) -- NIXLBench usage guide and troubleshooting
**Confidence:** HIGH

## Summary

Phase 34 authors the NIXLBench usage and troubleshooting page (`usage.md`), replacing the stub created in Phase 32. All source material lives in `benchmark/nixlbench/README.md` and is well-structured. The page consolidates usage guide content (ETCD coordination, communication patterns, CLI options, output interpretation) and troubleshooting (four required failure modes) into a single page per user decision D-01. Cross-linking to backend User Guide pages and the ETCD metadata exchange page completes the requirements.

The established Fern MDX patterns from Phases 32-33 (frontmatter, `<Warning>` callouts, inline backend links on first mention) carry forward unchanged. No new libraries, tools, or infrastructure are needed -- this is purely a documentation authoring task.

**Primary recommendation:** Author `usage.md` as a single page with four sections (ETCD Coordination, Communication Patterns, CLI Options, Troubleshooting), drawing examples directly from the README source material and linking every backend name to its User Guide page on first mention.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Everything goes in a single `usage.md` page -- usage guide at the top, troubleshooting section at the bottom. No separate `troubleshooting.md`. This keeps the NIXLBench nav at 3 pages (index.md, build.md, usage.md).
- **D-02:** Page sections (in order): ETCD Coordination (brief), Communication Patterns (with examples), CLI Options (essential flags), Troubleshooting.
- **D-03:** Use a `<Warning>` callout for the 60-second join window barrier. Link to the existing "Metadata Exchange with ETCD" User Guide page (`docs/user-guide/etcd-metadata-exchange.md`). Brief Docker one-liner for starting ETCD. Do NOT re-explain ETCD setup in detail.
- **D-04:** Mention when ETCD is required vs optional (network backends require it; storage backends can run without it for single instances). Keep this concise -- 2-3 sentences.
- **D-05:** Focus examples on the 4 communication patterns (pairwise, many-to-one, one-to-many, TP) using UCX as the default backend. Show the `--scheme` flag variations with brief explanation of each pattern.
- **D-06:** Add 1-2 storage backend examples (GDS for local storage, OBJ for S3) to demonstrate how storage benchmarks differ from network benchmarks. Link to backend User Guide pages for backend-specific flags.
- **D-07:** Show multi-node examples demonstrating initiator/target worker launching (two `nixlbench` commands on separate hosts pointing to the same ETCD server).
- **D-08:** Claude's discretion on how many CLI flags to document. Should include enough for a developer to run all 4 communication patterns and basic storage benchmarks.
- **D-09:** Cover the 4 ROADMAP-required failure modes: ETCD connection failures, CUDA/GPU not found, backend library missing, build failures.
- **D-10:** Claude's discretion on whether to add runtime essentials (library-not-found errors, ETCD cleanup after failed runs).
- **D-11:** Link every backend name (UCX, GDS, GDS_MT, POSIX, GPUNETIO, Mooncake, HF3FS, OBJ, GUSLI, Azure Blob) to its corresponding User Guide backend page on first mention per page.
- **D-12:** Link to "Metadata Exchange with ETCD" when ETCD coordination is discussed.

### Claude's Discretion
- Exact CLI flag table scope (core only vs core + memory/transfer)
- Whether to include ETCD cleanup and library-not-found in troubleshooting
- Ordering of troubleshooting items
- Whether to include a config file example (TOML format)
- Whether to mention NVSHMEM worker type or keep focus on the default NIXL worker

### Deferred Ideas (OUT OF SCOPE)
- Full CLI reference (70+ flags) -- out of scope for v1.1
- Per-backend example pages -- deferred
- Performance tuning guide (CPU affinity, network tuning) -- not in ROADMAP scope
- Config file format documentation -- Claude's discretion whether to include a brief mention
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NB-03 | Usage guide covers launching workers (initiator/target), ETCD coordination setup, communication patterns (pairwise/many-to-one/one-to-many/TP), and reading benchmark output | README lines 390-603 provide complete source material for all four patterns, worker launching, and ETCD coordination |
| NB-04 | Troubleshooting covers ETCD connection failures, CUDA/GPU not found, backend library missing, build failures | README lines 817-915 provide troubleshooting content for all four failure modes |
| NB-05 | `<Warning>` callout for ETCD 60-second join window barrier; link to "Metadata Exchange with ETCD" page | ETCD metadata exchange page confirmed at `docs/user-guide/etcd-metadata-exchange.md`; `<Warning>` component verified in use across docs |
| NB-06 | Every backend name links to its corresponding User Guide backend page on first mention | All 12 backend pages confirmed at `docs/user-guide/backends/` (ucx, gds, gds-mt, posix, gpunetio, mooncake, hf3fs, obj, gusli, azure-blob, libfabric, uccl) |
</phase_requirements>

## Architecture Patterns

### Page Structure (per D-01 and D-02)

The single `usage.md` page follows this structure:

```
---
title: NIXLBench Usage and Troubleshooting
description: How to run NIXLBench benchmarks and troubleshoot common issues.
---

[Intro paragraph -- what this page covers]

## etcd Coordination
  - When ETCD is required vs optional (D-04)
  - Docker one-liner (D-03)
  - <Warning> for 60-second barrier (D-03)
  - Link to etcd-metadata-exchange page (D-12)

## Communication Patterns
  - Pairwise (D-05)
  - Many-to-one (D-05)
  - One-to-many (D-05)
  - TP (tensor parallel) (D-05)
  - Multi-node example (D-07)

## Storage Backend Examples
  - GDS example (D-06)
  - OBJ (S3) example (D-06)

## CLI Options
  - Core Configuration table
  - Memory and Transfer table
  - (Claude's discretion on scope)

## Troubleshooting
  - etcd connection failures (D-09)
  - CUDA/GPU not found (D-09)
  - Backend library missing (D-09)
  - Build failures (D-09)
  - [Optional: ETCD cleanup, library-not-found] (D-10)
```

### Fern MDX Patterns (established in Phase 33)

- **Frontmatter:** `title:` + `description:` on every page [VERIFIED: existing docs]
- **Callouts:** `<Warning>` for critical warnings, `<Note>` for supplementary info, `<Tip>` for helpful hints [VERIFIED: docs/api-reference/*.md, docs/user-guide/etcd-metadata-exchange.md]
- **Inline links:** Fern path format `/docs/user-guide/backends/ucx` (not relative paths) [VERIFIED: docs/development/benchmarking/nixlbench/index.md]
- **Code blocks:** Standard triple-backtick with language identifier [VERIFIED: existing docs]
- **Backend linking convention (Phase 33 D-09):** First mention of each backend name per page gets an inline link to its User Guide page [VERIFIED: index.md]

### Backend Link Map

All links verified against filesystem at `docs/user-guide/backends/`:

| Backend Name in Prose | Link Target |
|----------------------|-------------|
| UCX | `/docs/user-guide/backends/ucx` |
| GPUDirect Storage (GDS) | `/docs/user-guide/backends/gds` |
| GPUDirect Storage MT (GDS_MT) | `/docs/user-guide/backends/gds-mt` |
| POSIX | `/docs/user-guide/backends/posix` |
| DOCA GPUNetIO | `/docs/user-guide/backends/gpunetio` |
| Mooncake | `/docs/user-guide/backends/mooncake` |
| Libfabric | `/docs/user-guide/backends/libfabric` |
| HF3FS | `/docs/user-guide/backends/hf3fs` |
| OBJ | `/docs/user-guide/backends/obj` |
| Azure Blob | `/docs/user-guide/backends/azure-blob` |
| GUSLI | `/docs/user-guide/backends/gusli` |

[VERIFIED: `ls docs/user-guide/backends/` confirms all 12 files exist]

### Terminology (QS-01 compliance)

- Use `plug-in` (not `plugin`) in prose [VERIFIED: REQUIREMENTS.md QS-01]
- Use `etcd` (lowercase) in prose, `ETCD` only in CLI flag names like `--etcd_endpoints` [VERIFIED: existing docs pattern in etcd-metadata-exchange.md]
- Backend names follow their canonical capitalization: UCX, GDS, GDS_MT, POSIX, GPUNETIO, OBJ, GUSLI, HF3FS [VERIFIED: README source]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ETCD setup instructions | Duplicate the etcd-metadata-exchange page content | Link to `/docs/user-guide/etcd-metadata-exchange` | Per D-03 and QS-02 (no duplicated content) |
| Backend-specific flag tables | Exhaustive per-backend CLI reference | Brief example + link to backend User Guide page | Per D-06 and deferred scope (full CLI reference out of scope) |
| Build troubleshooting | Repeat build.md content | Link to build page for build-specific setup | QS-02 compliance |

## Source Material Mapping

This maps README source sections to page sections, so the implementer knows exactly where to extract content from.

| Page Section | README Source | Lines | Adaptation Notes |
|-------------|-------------|-------|------------------|
| etcd Coordination | "ETCD Coordination Setup" + "Using ETCD for Coordination" | 392-603 | Condense to Docker one-liner + required/optional rules + 60s barrier warning |
| Communication Patterns | "Basic Usage Examples" + scheme flag from CLI Options | 415-429, 447 | Restructure around 4 patterns with `--scheme` flag; add multi-node example from lines 596-603 |
| Storage Backend Examples | Backend-Specific Examples (GDS, OBJ) | 628-753 | Pick 1 GDS + 1 OBJ example; link to backend pages for rest |
| CLI Options | "Command Line Options" (Core + Memory/Transfer sections) | 432-470 | Tables for core config + memory/transfer; skip per-backend flag tables |
| Troubleshooting: ETCD | "ETCD Cleanup" + connection context | 911-915 | Frame as "etcd connection failures" with cleanup command |
| Troubleshooting: CUDA | "CUDA Not Found" + "GPU Access Issues" | 822-893 | Combine into single CUDA/GPU troubleshooting entry |
| Troubleshooting: Backend lib | "Library Not Found Errors" | 874-880 | `ldconfig` + `ldd` commands |
| Troubleshooting: Build | "UCX Build Failures" + "etcd-cpp-api Build Issues" + "Docker Build Failures" | 836-869 | Combine build failures into single section |

## Common Pitfalls

### Pitfall 1: Duplicating ETCD Setup Content
**What goes wrong:** Writing a full ETCD setup guide in usage.md that duplicates etcd-metadata-exchange.md
**Why it happens:** The README has extensive ETCD content that feels natural to include
**How to avoid:** Per D-03, only include Docker one-liner + 60s barrier `<Warning>` + link. No configuration details.
**Warning signs:** Usage.md ETCD section exceeds 15-20 lines

### Pitfall 2: Exhaustive CLI Reference
**What goes wrong:** Documenting all 70+ CLI flags when only core + memory/transfer flags are needed
**Why it happens:** README has comprehensive flag list that's easy to copy wholesale
**How to avoid:** Per D-08, include only flags needed for the 4 communication patterns + basic storage benchmarks. Per-backend flags are out of scope.
**Warning signs:** CLI Options section has more than 2-3 tables

### Pitfall 3: Using "ETCD" Instead of "etcd" in Prose
**What goes wrong:** Inconsistent capitalization fails QS-01
**Why it happens:** The README uses "ETCD" throughout
**How to avoid:** Use `etcd` in prose text, `ETCD` only when referring to CLI flag values (e.g., `--runtime_type ETCD`)
**Warning signs:** Uppercase "ETCD" appearing in descriptive text rather than code contexts

### Pitfall 4: Missing Backend Links on First Mention
**What goes wrong:** A backend name appears in prose without a link, failing NB-06
**Why it happens:** Backends mentioned in code blocks or passing references get overlooked
**How to avoid:** Audit every backend name in prose (not code blocks) and link the first occurrence per page
**Warning signs:** Backend name in prose text without `[Name](/docs/...)` wrapper

### Pitfall 5: Relative vs Fern Path Links
**What goes wrong:** Using relative markdown links (`../user-guide/backends/ucx.md`) instead of Fern paths (`/docs/user-guide/backends/ucx`)
**Why it happens:** Habit from standard markdown
**How to avoid:** All cross-page links use Fern absolute paths starting with `/docs/`
**Warning signs:** Links containing `../` or `.md` extensions

## Code Examples

### Warning Callout for 60-Second Barrier (verified Fern pattern)

```markdown
<Warning>
All NIXLBench workers in a benchmark group must connect to etcd within 60 seconds of the first worker joining. Workers that miss this window cause the barrier to fail and the benchmark to abort. For etcd setup and configuration details, see [Metadata Exchange with etcd](/docs/user-guide/etcd-metadata-exchange).
</Warning>
```

Source: Fern callout syntax verified from `docs/api-reference/python-api.md` and `docs/user-guide/etcd-metadata-exchange.md` [VERIFIED: codebase grep]

### Backend First-Mention Link Pattern (from Phase 33 index.md)

```markdown
NIXLBench supports network backends such as [UCX](/docs/user-guide/backends/ucx),
[Libfabric](/docs/user-guide/backends/libfabric), [Mooncake](/docs/user-guide/backends/mooncake),
and [DOCA GPUNetIO](/docs/user-guide/backends/gpunetio), as well as storage backends including
[GPUDirect Storage](/docs/user-guide/backends/gds) and [OBJ](/docs/user-guide/backends/obj).
```

Source: Pattern established in `docs/development/benchmarking/nixlbench/index.md` [VERIFIED: codebase]

### Communication Pattern Example Structure

```markdown
### Pairwise

Pairwise transfers data between matched pairs of initiators and targets. This is the default scheme.

On host 1 (initiator):

```bash
nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX \
  --initiator_seg_type VRAM --target_seg_type VRAM --scheme pairwise
```

On host 2 (target):

```bash
nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX \
  --initiator_seg_type VRAM --target_seg_type VRAM --scheme pairwise
```
```

Source: Derived from README lines 596-603 and CLI flag `--scheme` at line 447 [VERIFIED: README]

### Troubleshooting Entry Structure

```markdown
### etcd Connection Failures

**Symptoms:** Workers fail to join the benchmark group, or the barrier times out after 60 seconds.

**Resolution:**

1. Verify the etcd server is running and reachable:

   ```bash
   ETCDCTL_API=3 etcdctl endpoint health --endpoints=http://etcd-server:2379
   ```

2. If a previous run failed, clean up stale keys before retrying:

   ```bash
   ETCDCTL_API=3 etcdctl del "xferbench" --prefix=true
   ```
```

Source: README lines 911-915 for cleanup command [VERIFIED: README]

## Discretion Recommendations

For items marked as Claude's discretion in the CONTEXT.md:

### CLI Flag Scope: Core + Memory/Transfer (recommended)
Include two flag tables: Core Configuration (6 flags) and Memory and Transfer Configuration (12 flags). Skip Performance/Threading and Device/Network tables -- they are advanced tuning. This covers everything needed for the 4 communication patterns and basic storage use. [ASSUMED]

### Include ETCD Cleanup and Library-Not-Found (recommended)
These are the two most common runtime issues per D-10. The ETCD cleanup command (`etcdctl del "xferbench" --prefix=true`) is critical for developers hitting stale state after crashes. Library-not-found (`ldconfig` + `ldd`) covers the "backend library missing" requirement naturally. Both fit in 10-15 lines total. [ASSUMED]

### Troubleshooting Order (recommended)
Order by likelihood of encounter: (1) etcd connection failures -- most common first-run issue, (2) Build failures -- second most common for native builds, (3) CUDA/GPU not found, (4) Backend library missing. Then optional: ETCD cleanup, library-not-found. [ASSUMED]

### Config File: Brief Mention Only (recommended)
Include a 3-4 sentence note that `--config_file` accepts TOML files, with a minimal example. This is useful without being a full reference. [ASSUMED]

### NVSHMEM Worker: Brief Mention (recommended)
Include a 2-sentence note about `--worker_type nvshmem` for GPU-only VRAM transfers. Do not expand into a full section. [ASSUMED]

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Fern docs build check |
| Config file | `fern/fern.config.json` |
| Quick run command | `cd fern && fern check` |
| Full suite command | `cd fern && fern check` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NB-03 | Usage guide covers workers, ETCD, patterns, output | manual review + fern check | `cd fern && fern check` | stub exists at `docs/development/benchmarking/nixlbench/usage.md` |
| NB-04 | Troubleshooting covers 4 failure modes | manual review | grep for section headers in usage.md | stub exists |
| NB-05 | `<Warning>` callout for 60s barrier + ETCD link | manual review + fern check | grep for `<Warning>` and `/docs/user-guide/etcd-metadata-exchange` in usage.md | stub exists |
| NB-06 | Backend names link to User Guide pages | manual review | grep for backend link patterns in usage.md | stub exists |

### Sampling Rate
- **Per task commit:** `cd fern && fern check`
- **Per wave merge:** `cd fern && fern check` + manual link verification
- **Phase gate:** `fern check` green + all 4 requirement checks pass

### Wave 0 Gaps
None -- existing test infrastructure (fern check) covers all phase requirements.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Core + Memory/Transfer tables (18 flags) is the right scope for CLI Options | Discretion Recommendations | Page either too sparse or too bloated -- low risk, easy to adjust |
| A2 | ETCD cleanup and library-not-found should be included | Discretion Recommendations | Page slightly longer than needed -- very low risk |
| A3 | Troubleshooting order by likelihood is correct | Discretion Recommendations | Negligible -- just reordering |
| A4 | Brief config file mention is sufficient | Discretion Recommendations | Low risk -- can expand later |
| A5 | Brief NVSHMEM mention is sufficient | Discretion Recommendations | Low risk -- NVSHMEM is niche |

## Open Questions

None -- all source material is available in the README, all target link pages exist, and Fern MDX patterns are well established from Phases 32-33.

## Sources

### Primary (HIGH confidence)
- `benchmark/nixlbench/README.md` -- Complete source material for usage, CLI flags, ETCD coordination, troubleshooting
- `docs/development/benchmarking/nixlbench/index.md` -- Established backend linking pattern from Phase 33
- `docs/development/benchmarking/nixlbench/build.md` -- Established Fern MDX patterns from Phase 33
- `docs/user-guide/etcd-metadata-exchange.md` -- Confirmed link target for ETCD cross-reference
- `docs/user-guide/backends/` -- All 12 backend pages confirmed present
- `docs/api-reference/python-api.md` -- `<Warning>` callout syntax verified

### Secondary (MEDIUM confidence)
- None needed -- all claims verified against codebase

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- purely documentation, no libraries needed
- Architecture: HIGH -- page structure locked by user decisions, Fern patterns verified
- Pitfalls: HIGH -- patterns and anti-patterns observed directly from codebase and requirements

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (stable -- documentation patterns unlikely to change)
