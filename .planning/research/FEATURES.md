# Feature Research

**Domain:** Benchmark tool documentation pages (NIXLBench + KVBench) under Developer Guide
**Researched:** 2026-04-07
**Confidence:** HIGH — both READMEs are comprehensive, kvbench/docs/ contains three additional guides

## Context

"Features" here means documentation page categories and sections for two benchmark tools. The
existing site already covers ETCD, backends, building from source, and API references. The new
pages must not duplicate those — they link to them. The audience is developers benchmarking NIXL
transfer performance in distributed AI inference settings.

---

## NIXLBench Documentation Pages

### Table Stakes (Must Have)

Pages developers expect when looking up a benchmark tool. Missing these = "go read the source code".

| Page / Section | Why Expected | Complexity | Notes |
|----------------|--------------|------------|-------|
| Overview | What it is, why it exists, key capabilities | LOW | Covers: multi-backend, ETCD coordination, memory types, communication patterns, worker types |
| System requirements | Devs need to know if they can run it before spending time building | LOW | HW (x86/aarch64, GPU, NIC), SW (Ubuntu 22/24, Docker 20.10+, CUDA 12.8+), min RAM/disk |
| Building: Docker path | Recommended method; devs expect it to be prominent | MEDIUM | build.sh options table, ETCD container launch, first benchmark run |
| Building: Native path | Needed for CI / restricted environments | HIGH | Ordered deps (NIXL first, then nixlbench), meson options, PATH/LD_LIBRARY_PATH setup |
| Quick start | Concrete "copy-paste and it works" in under 5 minutes | LOW | Docker path only, UCX backend, single pair of hosts |
| CLI reference — core flags | Developers need all flags with types and defaults in one place | MEDIUM | Group by: core config, memory/transfer, performance/threading, device/network |
| CLI reference — storage flags | Storage backends have distinct flag sets | MEDIUM | Per-backend tables: GDS, GDS_MT, POSIX, HF3FS, OBJ, AZURE_BLOB, GUSLI |
| CLI reference — config file | TOML config file as alternative to CLI flags | LOW | Precedence rules, example TOML |
| ETCD coordination | ETCD is mandatory for network backends and multi-node — needs its own callout | LOW | Required vs optional matrix per backend type; cleanup command for stuck instances |
| Backend-specific examples | Devs target one backend; they need runnable copy-paste examples | MEDIUM | One subsection per backend: UCX, GPUNETIO, GDS, GDS_MT, POSIX, GUSLI, OBJ, AZURE_BLOB, NVSHMEM |
| Troubleshooting | Build issues, runtime errors, and perf tuning all appear in the README | MEDIUM | Sections: build failures (CUDA, UCX, etcd-cpp-api, Docker), runtime (library not found, GPU, network), ETCD cleanup, perf tuning (CPU affinity, network buffers) |

### Differentiators (Worth Having)

Sections not universally present in benchmark tool docs but genuinely useful here.

| Page / Section | Value Proposition | Complexity | Notes |
|----------------|-------------------|------------|-------|
| Communication patterns explainer | "Pairwise / many-to-one / one-to-many / TP" are non-obvious; a short explainer prevents misconfiguration | LOW | Embedded in overview or usage page; diagram optional |
| Worker types comparison (nixl vs nvshmem) | NVSHMEM worker is VRAM-only with different constraints; knowing when to use each saves debugging | LOW | Short table or callout box |
| Multi-node launch guide | The 60s ETCD window and rank coordination are footguns; explicit guidance prevents silent failures | LOW | Sequence diagram of rank registration; ETCD cleanup step |
| Performance tuning guide | CPU affinity, NUMA binding, network buffer sizing — non-obvious but high impact | LOW | Can live in Troubleshooting under "Performance Tuning" subheading |

### Anti-Features (Explicitly Avoid)

| Page / Section | Why Requested | Why Problematic | Alternative |
|----------------|---------------|-----------------|-------------|
| Full dependency build instructions duplicated from Building from Source | Devs find that page via nav | Creates drift when NIXL build steps change | Link to the existing Developer Guide > Building NIXL from Source pages |
| ETCD administration guide | Devs want to understand ETCD deeply | Out of scope for a benchmark tool; ETCD is a prerequisite | Link to the existing User Guide > Metadata Exchange with ETCD page |
| API reference for nixlbench internals | Might seem logical for a developer guide | nixlbench is a standalone binary, not a library — it has no public API | CLI reference is the right artifact |
| Separate "Installation" page | Mirrors what "Building" already covers | Two pages for the same concept creates duplication and nav confusion | Merge into a single "Building NIXLBench" page with Docker and native subsections |

---

## KVBench Documentation Pages

### Table Stakes (Must Have)

| Page / Section | Why Expected | Complexity | Notes |
|----------------|--------------|------------|-------|
| Overview | What KVBench does vs what NIXLBench does — the relationship is non-obvious | LOW | Two function groups: KVBench commands (LLM KV cache testing) and CTP commands (asymmetric traffic matrices); Python tool that calls nixlbench |
| Building: Docker path | Reuses nixlbench Docker container — must say so explicitly | LOW | Points to nixlbench/contrib/build.sh; no separate container |
| Building: Python venv path | Lightweight local install for plan/kvcache without running nixlbench | LOW | python3 -m venv, pip install uv, uv sync |
| Command reference: plan | Most-used command for pre-benchmark configuration | MEDIUM | --model, --model_config, --format, all shared benchmark args; show sample output |
| Command reference: profile | Runs the actual benchmark | MEDIUM | Same args as plan; explain it calls nixlbench under the hood |
| Command reference: kvcache | Inspect KV cache sizing without running anything | LOW | Tabular output with model, ISL, IO Size, TP, PP, page size |
| Command reference: ct-perftest | Custom traffic pattern benchmarking | MEDIUM | Config YAML structure, matrix file format, CUDA_VISIBLE_DEVICES note |
| Command reference: sequential-ct-perftest | Multi-pattern sequential testing | MEDIUM | YAML config with traffic_patterns list, sleep_after_launch_sec, JSON output |
| Model configuration guide | YAML schema is not self-evident; devs need to know what each field means | MEDIUM | strategy/runtime/system sections with field descriptions; block vs layer access patterns; already exists in kvbench/docs/creating-a-model-config.md |
| CLI override arguments | --pp, --tp, --isl, --osl etc let devs test without editing YAML files | LOW | Table of all override args with defaults |
| LLM architecture examples | Concrete examples for DeepSeek R1 and Llama 3.1 with sample outputs | MEDIUM | Block vs layer access, TP/PP combinations, sample nixlbench command output |
| CTP examples | Matrix file format and YAML config are both non-obvious | MEDIUM | Matrix file syntax, YAML config, matgen script usage, srun/Slurm usage |

### Differentiators (Worth Having)

| Page / Section | Value Proposition | Complexity | Notes |
|----------------|-------------------|------------|-------|
| Extending KVBench: new model architecture | Developer audience will want to add their own models | MEDIUM | Already exists in kvbench/docs/adding-a-new-model-architecture.md; inherit BaseModelArch, implement 3 methods, register in factory |
| KVBench to NIXLBench relationship explainer | It is not obvious that profile calls nixlbench or that plan just generates commands | LOW | Short diagram or callout: "KVBench is a command generator / runner wrapper around NIXLBench" |
| GDS profiling tutorial | End-to-end tutorial for a realistic use case (GPU-to-storage KV offload) | LOW | Already exists in kvbench/docs/tutorial-gds.md; write→read sequence with sample output |
| Traffic matrix generation | inference_workload_matgen.py is a separate helper not covered in README beyond a snippet | LOW | Brief callout with the matgen CLI flags table |

### Anti-Features (Explicitly Avoid)

| Page / Section | Why Requested | Why Problematic | Alternative |
|----------------|---------------|-----------------|-------------|
| Full NIXLBench CLI reference duplicated | Devs want all flags in one place | KVBench passes flags through to nixlbench; duplicating creates drift | Link to the NIXLBench CLI reference page with a note about which args are passed through |
| Separate "Installation" page | Mirrors Building | Same duplication risk as NIXLBench | Single "Building KVBench" page covering Docker and Python venv |
| Model architecture deep-dive (transformer math) | Might seem helpful for understanding KV cache sizes | Out of scope; KVBench abstracts the math | One-line explanation of KV cache sizing, link to model YAML format |

---

## Feature Dependencies

```
NIXLBench Overview
    └──must precede──> NIXLBench CLI Reference
                           └──must precede──> NIXLBench Backend Examples
                           └──must precede──> NIXLBench Troubleshooting

KVBench Overview
    └──must precede──> KVBench Command Reference (plan, profile, kvcache)
    └──must precede──> KVBench CTP Command Reference (ct-perftest, sequential-ct-perftest)

KVBench Model Config Guide
    └──must precede──> KVBench LLM Architecture Examples
    └──must precede──> KVBench GDS Tutorial

NIXLBench page (built)
    └──required by──> KVBench profile command docs
                          (profile invokes nixlbench; must link to nixlbench CLI ref)

Existing: Developer Guide > Building NIXL from Source
    └──referenced by──> NIXLBench Native Build section (links, does not duplicate)

Existing: User Guide > Metadata Exchange with ETCD
    └──referenced by──> NIXLBench ETCD Coordination section (links, does not duplicate)

Existing: User Guide > NIXL Backends (UCX, GDS, etc.)
    └──referenced by──> Both NIXLBench and KVBench backend examples (links, does not duplicate)
```

### Dependency Notes

- **KVBench profile requires NIXLBench to exist:** The profile command calls nixlbench as a subprocess. KVBench docs must be sequenced after NIXLBench docs or written with a cross-link placeholder.
- **Model config guide required by LLM examples:** Examples use YAML files; readers must understand the schema before examples make sense.
- **ETCD coordination is shared state:** Both tools rely on ETCD. NIXLBench docs explain ETCD setup. KVBench docs must link to NIXLBench's ETCD section and to the existing ETCD User Guide page rather than re-explaining it.

---

## MVP Definition

This is a documentation milestone, so "MVP" = minimum docs to make the tools usable without reading source code.

### Launch With (v1.1 — this milestone)

NIXLBench:
- [x] Overview page
- [x] Building page (Docker + native, combined)
- [x] Quick start section (Docker only)
- [x] CLI reference (core + storage + config file)
- [x] ETCD coordination section
- [x] Backend-specific examples
- [x] Troubleshooting page

KVBench:
- [x] Overview page (with relationship to NIXLBench)
- [x] Building page (Docker + Python venv)
- [x] Command reference (plan, profile, kvcache)
- [x] Command reference (ct-perftest, sequential-ct-perftest)
- [x] Model configuration guide (from kvbench/docs/creating-a-model-config.md)
- [x] LLM architecture examples (DeepSeek R1, Llama 3.1)
- [x] CTP examples (matrix format, YAML config, matgen script)

### Add After Validation (v1.x)

- [ ] Extending KVBench: adding a new model architecture — add when developer contributors increase
- [ ] KVBench GDS profiling tutorial — add when GDS is a high-traffic backend
- [ ] NIXLBench performance tuning standalone guide — add when perf troubleshooting is top support topic
- [ ] Traffic matrix generation deep-dive — add if matgen becomes widely used

### Future Consideration (v2+)

- [ ] Interactive benchmark configuration builder — nice UX but requires significant tooling investment
- [ ] Benchmark results comparison tables across backends — requires standardized output format

---

## Feature Prioritization Matrix

| Documentation Feature | User Value | Authoring Cost | Priority |
|-----------------------|------------|----------------|----------|
| NIXLBench CLI reference | HIGH | MEDIUM | P1 |
| NIXLBench backend-specific examples | HIGH | MEDIUM | P1 |
| NIXLBench building (Docker) | HIGH | LOW | P1 |
| NIXLBench ETCD coordination section | HIGH | LOW | P1 |
| NIXLBench troubleshooting | HIGH | LOW | P1 |
| KVBench command reference (plan/profile/kvcache) | HIGH | MEDIUM | P1 |
| KVBench model configuration guide | HIGH | LOW | P1 |
| KVBench LLM architecture examples | HIGH | MEDIUM | P1 |
| KVBench CTP command reference | MEDIUM | MEDIUM | P1 |
| KVBench CTP examples | MEDIUM | MEDIUM | P1 |
| NIXLBench building (native) | MEDIUM | HIGH | P2 |
| KVBench GDS profiling tutorial | MEDIUM | LOW | P2 |
| KVBench: adding new model architecture | MEDIUM | LOW | P2 |
| NIXLBench performance tuning section | MEDIUM | LOW | P2 |
| Traffic matrix generation guide | LOW | LOW | P3 |

**Priority key:**
- P1: Must have for this milestone
- P2: Include if content is straightforward (most are); defer only if timeline is tight
- P3: Nice to have, future consideration

---

## Sources

- `/home/omrik/Projects/nixl.gitlab/benchmark/nixlbench/README.md` — direct source for NIXLBench features, CLI flags, examples, troubleshooting
- `/home/omrik/Projects/nixl.gitlab/benchmark/kvbench/README.md` — direct source for KVBench commands, arguments, examples
- `/home/omrik/Projects/nixl.gitlab/benchmark/kvbench/docs/tutorial-gds.md` — KVBench GDS tutorial content
- `/home/omrik/Projects/nixl.gitlab/benchmark/kvbench/docs/creating-a-model-config.md` — KVBench model config guide content
- `/home/omrik/Projects/nixl.gitlab/benchmark/kvbench/docs/adding-a-new-model-architecture.md` — KVBench extension guide content
- `/home/omrik/Projects/nixl.gitlab/benchmark/kvbench/docs/ct-perftest.md` — CTPerftest implementation and usage details
- `/home/omrik/Projects/nixl.gitlab/docs/index.yml` — existing site navigation (to identify cross-link targets and avoid duplication)
- `/home/omrik/Projects/nixl.gitlab/.planning/PROJECT.md` — milestone scope and constraints

---
*Feature research for: NIXLBench and KVBench documentation pages (NIXL v1.1 milestone)*
*Researched: 2026-04-07*
