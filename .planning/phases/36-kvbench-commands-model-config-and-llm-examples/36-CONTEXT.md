# Phase 36: KVBench Commands, Model Config, and LLM Examples - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Author the KVBench command reference (all 5 subcommands with CLI argument tables), model configuration guide (both YAML schemas), and LLM architecture examples (DeepSeek R1 and Llama 3.1 end-to-end). All content goes in a single `commands.md` page. No CTP-specific examples — those are deferred.

</domain>

<decisions>
## Implementation Decisions

### Page Structure
- **D-01:** All content in a single `commands.md` page. Sections in order: (1) Command Reference (all 5 subcommands), (2) Model Configuration Guide (both YAML schemas), (3) LLM Examples (DeepSeek R1 and Llama 3.1). No additional pages — KVBench nav stays at 3 pages (index.md, build.md, commands.md).

### Command Reference
- **D-02:** Document all 5 subcommands: `plan`, `profile`, `kvcache`, `ct-perftest`, `sequential-ct-perftest`. Each subcommand gets a brief description and its CLI argument table.
- **D-03:** CLI argument tables use two-column format: Argument | Description. Default values included in the description text (not a separate column). Matches the README's existing format.
- **D-04:** Arguments grouped by: Common Arguments, CLI Override Arguments, Plan Command Arguments, Shared Benchmark Arguments, CTP Command Arguments. This satisfies REQUIREMENTS KB-03 grouping.
- **D-05:** CLI tables must be validated against `python main.py [cmd] --help` output. Note that KVBench uses `--etcd-endpoints` (hyphens) while NIXLBench uses `--etcd_endpoints` (underscores) — per REQUIREMENTS QS-03.

### Model Configuration Guide
- **D-06:** Document both YAML schemas with field description tables + annotated examples:
  - **Model architecture YAML** (e.g., `model_deepseek_r1.yaml`): fields like `model_name`, `num_layers`, `num_query_heads`, `query_head_dimension`, etc.
  - **Model config YAML** (e.g., `block-tp1-pp8.yaml`): three sections — `strategy` (tp/pp/quant), `runtime` (isl/osl/requests), `system` (hardware/backend/access_pattern/page_size).
- **D-07:** Each schema gets a field table followed by a complete annotated YAML example showing real values from the `examples/` directory.

### LLM Examples
- **D-08:** Full end-to-end examples for DeepSeek R1 and Llama 3.1. Each example shows: (1) the model architecture YAML, (2) the model config YAML, (3) the `plan` command with its output (generated nixlbench command), (4) the `profile` command. Developers can copy-paste and run.
- **D-09:** Include both block and layer access pattern variants for at least one model to demonstrate the difference.

### Claude's Discretion
- Whether to include a CTP commands section beyond the CLI table (basic description vs detailed usage)
- Ordering of subcommands within the reference section
- Whether to show `kvcache` command output example
- How many model config variants to include (just tp1-pp8, or also tp8-pp16)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source material — CLI reference
- `benchmark/kvbench/README.md` §Command Line Arguments (line ~79) — All CLI argument tables grouped by category
- `benchmark/kvbench/README.md` §Command Descriptions (line ~162) — Subcommand descriptions and usage
- `benchmark/kvbench/main.py` — Entry point; validate `--help` output against documented arguments

### Source material — Model config
- `benchmark/kvbench/docs/creating-a-model-config.md` — Existing developer guide for model config YAML schema
- `benchmark/kvbench/examples/model_deepseek_r1.yaml` — DeepSeek R1 model architecture config
- `benchmark/kvbench/examples/model_llama_3_1_70b.yaml` — Llama 3.1 70B model architecture config
- `benchmark/kvbench/examples/block-tp1-pp8.yaml` — Block access pattern model config
- `benchmark/kvbench/examples/layer-tp1-pp8.yaml` — Layer access pattern model config

### Source material — Examples
- `benchmark/kvbench/README.md` §Examples (line ~218) — KVBench and CTP example commands with output

### Doc patterns
- `docs/development/benchmarking/nixlbench/usage.md` — NIXLBench usage page pattern (Phase 34 output)

### Requirements
- `.planning/REQUIREMENTS.md` — KB-03 (command reference), KB-04 (model config guide), KB-05 (LLM examples)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `benchmark/kvbench/docs/creating-a-model-config.md` — Existing model config guide; can be adapted (rewrite for Fern style, don't copy verbatim)
- `benchmark/kvbench/examples/` — 27 example YAML files covering various tp/pp/access pattern combinations
- Fern frontmatter pattern: `title:` + `description:` on all pages

### Established Patterns
- Two-column CLI argument tables (Argument | Description) used in the README
- Three-section model config structure (strategy / runtime / system) is well-established
- First-mention inline links for backend names (carried from Phase 33 D-09)
- Rewrite README prose for Fern doc style (carried from Phase 33 D-07)

### Integration Points
- `commands.md` replaces the stub file created in Phase 32 at `docs/development/benchmarking/kvbench/commands.md`
- No changes to `docs/index.yml` needed (commands.md entry already exists from Phase 32)

</code_context>

<specifics>
## Specific Ideas

- User wants full end-to-end examples with command output — developers should be able to copy-paste
- Field tables + annotated examples for both YAML schemas — structured documentation
- Two-column CLI tables matching README format

</specifics>

<deferred>
## Deferred Ideas

- CTP examples with traffic matrices — deferred per REQUIREMENTS.md
- KVBench GDS tutorial (`benchmark/kvbench/docs/tutorial-gds.md`) — P2, deferred
- Adding new model architecture guide (`benchmark/kvbench/docs/adding-a-new-model-architecture.md`) — P2, deferred
- Additional model examples beyond DeepSeek R1 and Llama 3.1

</deferred>

---

*Phase: 36-kvbench-commands-model-config-and-llm-examples*
*Context gathered: 2026-04-07*
