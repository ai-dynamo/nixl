# Phase 36: KVBench Commands, Model Config, and LLM Examples - Research

**Researched:** 2026-04-07
**Domain:** Technical documentation authoring (Fern MDX, CLI reference, YAML schema docs)
**Confidence:** HIGH

## Summary

This phase writes the body of `docs/development/benchmarking/kvbench/commands.md`, replacing the stub created in Phase 32. All content -- command reference for 5 subcommands, model configuration guide for both YAML schemas, and end-to-end LLM examples for DeepSeek R1 and Llama 3.1 -- goes on this single page per user decision D-01.

The source material is well-defined: the README contains all CLI argument tables and example outputs, `commands/args.py` is the authoritative Click source for argument definitions, `docs/creating-a-model-config.md` documents the model config schema, and the `examples/` directory provides 27 real YAML files. The NIXLBench usage page (Phase 34 output) establishes the Fern doc style pattern.

**Primary recommendation:** Transcribe CLI tables and YAML schemas directly from source code (`args.py`, `model_config.py`, example YAMLs), rewrite README prose in Fern doc style, and construct end-to-end examples from README examples section. Validate all CLI flags match `args.py` definitions.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** All content in a single `commands.md` page. Sections in order: (1) Command Reference (all 5 subcommands), (2) Model Configuration Guide (both YAML schemas), (3) LLM Examples (DeepSeek R1 and Llama 3.1). No additional pages -- KVBench nav stays at 3 pages (index.md, build.md, commands.md).
- **D-02:** Document all 5 subcommands: `plan`, `profile`, `kvcache`, `ct-perftest`, `sequential-ct-perftest`. Each subcommand gets a brief description and its CLI argument table.
- **D-03:** CLI argument tables use two-column format: Argument | Description. Default values included in the description text (not a separate column). Matches the README's existing format.
- **D-04:** Arguments grouped by: Common Arguments, CLI Override Arguments, Plan Command Arguments, Shared Benchmark Arguments, CTP Command Arguments. This satisfies REQUIREMENTS KB-03 grouping.
- **D-05:** CLI tables must be validated against `python main.py [cmd] --help` output. Note that KVBench uses `--etcd-endpoints` (hyphens) while NIXLBench uses `--etcd_endpoints` (underscores) -- per REQUIREMENTS QS-03.
- **D-06:** Document both YAML schemas with field description tables + annotated examples: Model architecture YAML and Model config YAML.
- **D-07:** Each schema gets a field table followed by a complete annotated YAML example showing real values from the `examples/` directory.
- **D-08:** Full end-to-end examples for DeepSeek R1 and Llama 3.1. Each example shows: (1) the model architecture YAML, (2) the model config YAML, (3) the `plan` command with its output, (4) the `profile` command. Developers can copy-paste and run.
- **D-09:** Include both block and layer access pattern variants for at least one model to demonstrate the difference.

### Claude's Discretion
- Whether to include a CTP commands section beyond the CLI table (basic description vs detailed usage)
- Ordering of subcommands within the reference section
- Whether to show `kvcache` command output example
- How many model config variants to include (just tp1-pp8, or also tp8-pp16)

### Deferred Ideas (OUT OF SCOPE)
- CTP examples with traffic matrices -- deferred per REQUIREMENTS.md
- KVBench GDS tutorial -- P2, deferred
- Adding new model architecture guide -- P2, deferred
- Additional model examples beyond DeepSeek R1 and Llama 3.1
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| KB-03 | KVBench command reference covers all 5 subcommands with CLI argument tables validated against `--help` output; arguments grouped by Common / CLI Override / Per-command | README CLI tables + `commands/args.py` Click decorators provide authoritative source; grouping matches README exactly |
| KB-04 | KVBench model configuration guide documents model architecture YAML schema and model config YAML schema with field descriptions and examples | `models/model_config.py` dataclasses + `docs/creating-a-model-config.md` + example YAMLs provide complete schema coverage |
| KB-05 | KVBench LLM examples page covers DeepSeek R1 and Llama 3.1 example configurations with end-to-end `plan` and `profile` command examples | README examples section has DeepSeek R1 block/layer examples with full command output; Llama 3.1 YAML available in examples/ |
</phase_requirements>

## Architecture Patterns

### Target File Structure
```
docs/development/benchmarking/kvbench/commands.md   # Replace stub (7 lines) with full content
```

No other files created or modified. The `docs/index.yml` entry already exists (line 86: `path: development/benchmarking/kvbench/commands.md`). [VERIFIED: codebase grep]

### Page Structure Pattern (from D-01)
```markdown
---
title: KVBench Commands and Examples
description: ...
---

[Intro paragraph]

## Command Reference
### plan
### profile  
### kvcache
### ct-perftest
### sequential-ct-perftest

## Command Line Arguments
### Common Arguments [table]
### CLI Override Arguments [table]
### Plan Command Arguments [table]
### Shared Benchmark Arguments [table]
### CTP Command Arguments [table]

## Model Configuration Guide
### Model Architecture YAML [schema table + example]
### Model Config YAML [schema table + example]

## LLM Examples
### DeepSeek R1 [end-to-end]
### Llama 3.1 [end-to-end]
```

### Fern Doc Style Pattern
From `docs/development/benchmarking/nixlbench/usage.md` (Phase 34): [VERIFIED: codebase read]
- Fern frontmatter: `title:` + `description:`
- First paragraph references related pages with relative links
- Two-column or three-column tables for CLI flags
- Code blocks with `bash` language tag
- First-mention inline links for backend names (Phase 33 D-09 pattern)
- `<Warning>`, `<Note>` Fern components where needed
- No HTML comments, no bare anchor links (QS-04)

### Anti-Patterns to Avoid
- **Duplicating build instructions:** Link to `./kvbench/build` instead
- **Re-explaining etcd setup:** Link to `/docs/user-guide/etcd-metadata-exchange`
- **Copying README verbatim:** Rewrite prose for Fern doc style (D-07 from Phase 33)
- **Ignoring the etcd flag discrepancy:** KVBench uses `--etcd-endpoints` (hyphens), NIXLBench uses `--etcd_endpoints` (underscores) -- must document per QS-03

## Source Material Inventory

### CLI Argument Tables (for KB-03)

Five argument groups from `benchmark/kvbench/README.md` lines 79-158 and `commands/args.py`: [VERIFIED: codebase read]

| Group | Defined In | Arg Count | Used By Commands |
|-------|-----------|-----------|-----------------|
| Common Arguments | `common_args()` decorator | 3 | plan, profile, kvcache |
| CLI Override Arguments | `cli_args()` decorator | 9 (includes `--source`, `--destination`) | plan, profile, kvcache |
| Plan Command Arguments | `plan_args()` decorator | 1 | plan only |
| Shared Benchmark Arguments | `nixl_bench_args()` decorator | 40+ | plan, profile |
| CTP Command Arguments | inline in main.py | 3-4 | ct-perftest, sequential-ct-perftest |

**Critical finding -- args.py has MORE arguments than the README documents.** The README's "Shared Benchmark Arguments" table lists ~30 arguments, but `args.py:nixl_bench_args()` defines additional arguments not in the README: [VERIFIED: codebase read]
- `--posix_api_type` (POSIX backend API type)
- `--benchmark_group` (parallel benchmark group name)
- `--num_files` (file count for storage)
- `--large_blk_iter_ftr` (large block iteration factor)
- `--gds_batch_pool_size`, `--gds_batch_limit` (GDS backend)
- `--gds_mt_num_threads` (GDS MT backend)
- `--gpunetio_device_list`, `--gpunetio_oob_list` (GPUNETIO backend)
- `--hf3fs_iopool_size` (HF3FS backend)
- `--obj_access_key`, `--obj_secret_key`, `--obj_session_token`, `--obj_bucket_name`, `--obj_scheme`, `--obj_region`, `--obj_use_virtual_addressing`, `--obj_endpoint_override`, `--obj_req_checksum`, `--obj_ca_bundle` (OBJ/S3 backend)

**Recommendation:** Document the arguments from the README tables (which match the user-facing documentation scope). Backend-specific arguments (GDS, GPUNETIO, HF3FS, OBJ) are passed through to nixlbench and are documented on respective backend pages. The planner should note this scope boundary.

### etcd Flag Discrepancy (QS-03)
- `args.py` line 162: `--etcd_endpoints` (underscores) -- Click accepts both forms [VERIFIED: codebase read]
- README examples: `--etcd-endpoints` (hyphens) [VERIFIED: codebase read]
- NIXLBench: `--etcd_endpoints` (underscores) [VERIFIED: Phase 34 output]
- Click's behavior: underscored option names accept both `--etcd_endpoints` and `--etcd-endpoints` on the command line [ASSUMED]
- **Action:** Document as `--etcd-endpoints` in KVBench docs (matching README convention) and note the difference from NIXLBench's `--etcd_endpoints`

### Model Architecture YAML Schema (for KB-04)

Two distinct schema patterns based on attention mechanism: [VERIFIED: codebase read]

**DeepSeek R1 (MLA -- Multi-Latent Attention):**
```yaml
model_name: 'DEEPSEEK_R1'
num_layers: 61
num_query_heads: 128
query_head_dimension: 128
embedding_dimension: 7168
rope_mla_dimension: 64
mla_latent_vector_dimension: 512
num_model_params: 671000000000
```

**Llama 3.1 70B (MHA/GQA -- Multi-Head / Grouped-Query Attention):**
```yaml
model_name: 'LLAMA3.1_70B'
num_layers: 80
num_query_heads_with_mha: 64
query_head_dimension: 128
gqa_num_queries_in_group: 8
num_model_params: 70000000000
```

Key schema differences:
- DeepSeek R1 uses `num_query_heads` + MLA-specific fields (`rope_mla_dimension`, `mla_latent_vector_dimension`, `embedding_dimension`)
- Llama 3.1 uses `num_query_heads_with_mha` + GQA field (`gqa_num_queries_in_group`)
- Both share: `model_name`, `num_layers`, `query_head_dimension`, `num_model_params`

### Model Config YAML Schema (for KB-04)

Three sections with dataclass definitions in `models/model_config.py`: [VERIFIED: codebase read]

| Section | Field | Type | Default | Description |
|---------|-------|------|---------|-------------|
| strategy | tp_size | int | 1 | Tensor parallelism size |
| strategy | pp_size | int | 1 | Pipeline parallelism size |
| strategy | model_quant_mode | str | "fp8" | Model weight quantization mode |
| strategy | kvcache_quant_mode | str | "fp8" | KV cache quantization mode |
| runtime | num_requests | int | 1 | Number of inference requests |
| runtime | isl | int | 1 | Input sequence length |
| runtime | osl | int | 1 | Output sequence length |
| system | backend | str | "SGLANG" | Inference backend engine |
| system | hardware | str | None | Hardware platform (e.g., "H100") |
| system | page_size | int | 1 | Page size for access pattern |
| system | access_pattern | str | None | KV cache access pattern ("block" or "layer") |
| system | source | str | None | Source descriptor type |
| system | destination | str | None | Destination descriptor type |

Note: The YAML key is `strategy` but `ModelConfig` stores it as `model` attribute (line 81: `model: StrategyConfig`). The YAML-to-Python mapping is: `strategy` -> `config.model`, `runtime` -> `config.runtime`, `system` -> `config.system`. [VERIFIED: model_config.py lines 162-173]

### LLM Examples (for KB-05)

The README provides three DeepSeek R1 examples with full command output: [VERIFIED: codebase read]
1. DeepSeek R1 with Block Access (TP=1, PP=16) -- lines 222-244
2. DeepSeek R1 with Layer Access (TP=1, PP=16) -- lines 246-269
3. CLI Override example (TP=1, PP=8, overriding to PP=32) -- lines 271-296

**Missing from README:** Llama 3.1 examples. The YAML exists (`model_llama_3_1_70b.yaml`) but no example command output in README.

**Recommendation for D-08/D-09:**
- DeepSeek R1: Use block-tp1-pp8 example with plan+profile, then show layer-tp1-pp8 variant (satisfies D-09: both access patterns)
- Llama 3.1: Construct a plan example using `model_llama_3_1_70b.yaml` + `block-tp1-pp8.yaml`. The command output cannot be verified without running the tool, so adapt the output format pattern from the DeepSeek examples.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI argument descriptions | Writing descriptions from scratch | Copy from `args.py` help strings and README tables | Authoritative source, avoids drift |
| Model config field descriptions | Inventing descriptions | Copy from `model_config.py` docstrings and `creating-a-model-config.md` | Already documented accurately |
| Example command output | Guessing nixlbench output format | Copy from README examples section (lines 218-296) | Exact output format matters for copy-paste |
| YAML schema structure | Describing from memory | Read actual YAML files from `examples/` directory | Real files are the schema specification |

## Common Pitfalls

### Pitfall 1: README CLI Table Drift from Source Code
**What goes wrong:** README documents `--etcd-endpoints` but `args.py` defines `--etcd_endpoints`. Some arguments in `args.py` are missing from README entirely.
**Why it happens:** README and code evolve independently.
**How to avoid:** Use `args.py` as the authoritative source for argument names and help text. Cross-reference with README for grouping and defaults. Flag any discrepancies.
**Warning signs:** Argument not found in `--help` output during QS-03 validation.

### Pitfall 2: Model Architecture Schema is Model-Specific
**What goes wrong:** Documenting a single "model architecture YAML schema" when different models have different fields.
**Why it happens:** DeepSeek R1 uses MLA fields, Llama uses MHA/GQA fields. They share some common fields but diverge.
**How to avoid:** Document common fields first, then model-specific fields in separate subsections or examples. Be explicit about which fields apply to which architecture.
**Warning signs:** A field table that claims to be universal but only matches one model.

### Pitfall 3: Llama 3.1 Example Output Cannot Be Verified
**What goes wrong:** Writing example `plan` command output for Llama 3.1 that doesn't match actual tool output.
**Why it happens:** README only has DeepSeek R1 examples. Cannot run the tool in this environment.
**How to avoid:** Either (a) construct the example with a clear note about the format being based on DeepSeek R1 patterns, or (b) show only the command invocation without output. Phase 37 QS-03 validation will catch output mismatches.
**Warning signs:** Batch size / block size numbers that don't make sense for Llama 3.1 70B architecture.

### Pitfall 4: YAML Key vs Python Attribute Name Mismatch
**What goes wrong:** Documenting `model:` as a YAML section when the YAML key is `strategy:`.
**Why it happens:** `ModelConfig.model` attribute stores strategy data, but YAML uses `strategy:` key.
**How to avoid:** Document the YAML keys (`strategy`, `runtime`, `system`), not the Python attributes.
**Warning signs:** Example YAML that uses `model:` instead of `strategy:`.

## Code Examples

### CLI Argument Table Format (from README, two-column per D-03)
```markdown
| Argument | Description |
| -------- | ----------- |
| `--model` | Path to a model architecture config YAML file |
| `--model_config` | Path to a model config YAML file |
```
Source: `benchmark/kvbench/README.md` lines 85-89 [VERIFIED: codebase read]

### Model Config Annotated YAML (for D-07)
```yaml
# Model config: block access, TP=1, PP=8
strategy:
  tp_size: 1                    # Tensor parallelism -- GPUs for tensor-parallel execution
  pp_size: 8                    # Pipeline parallelism -- GPUs for pipeline-parallel execution
  model_quant_mode: "fp8"       # Model weight quantization (fp8, fp16, int8)
  kvcache_quant_mode: "fp8"     # KV cache quantization (fp8, fp16, int8)

runtime:
  isl: 1000                     # Input sequence length (tokens)
  osl: 100                      # Output sequence length (tokens)
  num_requests: 10              # Number of inference requests

system:
  hardware: "H100"              # Hardware platform
  backend: "SGLANG"             # Inference backend engine
  access_pattern: "block"       # KV cache access pattern (block or layer)
  page_size: 16                 # Page size for block access pattern
```
Source: `benchmark/kvbench/examples/block-tp1-pp8.yaml` + `docs/creating-a-model-config.md` [VERIFIED: codebase read]

### End-to-End Example Pattern (for D-08)
```markdown
#### DeepSeek R1 -- Block Access (TP=1, PP=8)

**Model architecture** (`model_deepseek_r1.yaml`):
[show YAML]

**Model config** (`block-tp1-pp8.yaml`):
[show YAML]

**Plan command:**
```bash
python main.py plan \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp8.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
```

**Output:**
[show command output from README]

**Profile command:**
```bash
python main.py profile \
  --model ./examples/model_deepseek_r1.yaml \
  --model_config ./examples/block-tp1-pp8.yaml \
  --backend GDS \
  --source gpu \
  --etcd-endpoints "http://localhost:2379"
```
```
Source: `benchmark/kvbench/README.md` lines 218-296 [VERIFIED: codebase read]

## Discretion Recommendations

Based on the discretion areas from CONTEXT.md:

1. **CTP commands section depth:** Keep it minimal -- brief description for each CTP subcommand plus the CLI table. CTP examples are explicitly deferred. Include the Sequential CT Perftest and CT Perftest report format descriptions from README lines 195-215 as they help users understand what the commands do.

2. **Subcommand ordering:** Follow the README order: plan, profile, kvcache (KVBench commands first), then ct-perftest, sequential-ct-perftest (CTP commands). This matches the two-category structure from the index page.

3. **kvcache command output example:** Yes, include it. The README has a complete example with tabulated output (line 186-189). It's 4 lines and helps developers understand what kvcache shows.

4. **Model config variants:** Include tp1-pp8 as the primary example (simpler, matches build page examples). The LLM examples section uses tp1-pp16 (from README examples) and shows block vs layer variants, which is sufficient variety. No need for tp8-pp16 -- it would add length without new concepts.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Fern CLI |
| Config file | `fern/fern.config.json` |
| Quick run command | `cd fern && fern check` |
| Full suite command | `cd fern && fern check` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| KB-03 | CLI argument tables present for all 5 subcommands | manual review + fern check | `cd fern && fern check` | N/A (content review) |
| KB-04 | Both YAML schemas documented with field tables and examples | manual review | visual inspection of commands.md | N/A (content review) |
| KB-05 | DeepSeek R1 and Llama 3.1 end-to-end examples present | manual review | visual inspection of commands.md | N/A (content review) |

### Sampling Rate
- **Per task commit:** `cd fern && fern check`
- **Per wave merge:** Full fern check
- **Phase gate:** Fern check green + manual review of content completeness

### Wave 0 Gaps
None -- existing Fern infrastructure covers build validation. Content accuracy requires manual review against source material (Phase 37 QS-03 handles formal CLI validation).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Click accepts both `--etcd_endpoints` and `--etcd-endpoints` interchangeably when option defined with underscores | Source Material Inventory | Low -- if wrong, examples using hyphens would fail; easily tested |
| A2 | Llama 3.1 plan command output follows the same format as DeepSeek R1 (header block + nixlbench command) | Common Pitfalls / Pitfall 3 | Medium -- if output format differs, example would be misleading; Phase 37 QS-03 catches this |

## Open Questions

1. **Llama 3.1 example output values**
   - What we know: The command invocation is straightforward (`python main.py plan --model ./examples/model_llama_3_1_70b.yaml --model_config ./examples/block-tp1-pp8.yaml`)
   - What's unclear: The exact batch_size and block_size values in the generated nixlbench command
   - Recommendation: Use the DeepSeek R1 example format but note that specific output values depend on the model architecture. If possible, annotate the example as "representative output" or omit the output block for Llama 3.1.

2. **Backend-specific arguments scope**
   - What we know: `args.py` has ~15 backend-specific arguments (GDS, GPUNETIO, HF3FS, OBJ) not in README tables
   - What's unclear: Whether users expect these in KVBench docs or on backend pages
   - Recommendation: Document only the arguments from the README tables (matching D-04 grouping). Backend-specific args are passthrough to nixlbench and documented elsewhere.

## Sources

### Primary (HIGH confidence)
- `benchmark/kvbench/README.md` -- CLI tables, command descriptions, examples
- `benchmark/kvbench/commands/args.py` -- Authoritative Click CLI definitions
- `benchmark/kvbench/models/model_config.py` -- Model config dataclass schema
- `benchmark/kvbench/docs/creating-a-model-config.md` -- Model config guide
- `benchmark/kvbench/examples/*.yaml` -- All 27 example YAML files
- `benchmark/kvbench/main.py` -- Entry point, command registration, kvcache output
- `docs/development/benchmarking/nixlbench/usage.md` -- Phase 34 Fern doc style pattern
- `docs/development/benchmarking/kvbench/commands.md` -- Existing stub (7 lines)
- `docs/index.yml` line 86 -- Nav entry already registered

### Secondary (MEDIUM confidence)
- None needed -- all source material is in the codebase

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- this is documentation authoring, no libraries needed
- Architecture: HIGH -- page structure locked by user decisions, source material fully inventoried
- Pitfalls: HIGH -- identified from direct source code analysis (args.py vs README drift, schema differences)

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (stable -- source code changes would require re-research)
