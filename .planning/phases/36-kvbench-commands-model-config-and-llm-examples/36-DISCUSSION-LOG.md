# Phase 36: KVBench Commands, Model Config, and LLM Examples - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-07
**Phase:** 36-KVBench Commands, Model Config, and LLM Examples
**Areas discussed:** Page structure, CLI table format, Model config documentation, LLM examples depth

---

## Page Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Single commands.md | All content in one page: commands + model config + examples | ✓ |
| Two pages | commands.md + examples.md | |
| Three pages | commands.md + model-config.md + llm-examples.md | |

**User's choice:** Single commands.md
**Notes:** KVBench content is less voluminous than NIXLBench. Keeps nav at 3 pages.

---

## CLI Table Format

| Option | Description | Selected |
|--------|-------------|----------|
| Argument + description | Two-column table, defaults in description text | ✓ |
| Argument + description + default | Three-column table with explicit defaults column | |
| You decide | Claude picks matching existing patterns | |

**User's choice:** Argument + description (two-column)
**Notes:** Matches README format.

---

## Model Config Documentation

| Option | Description | Selected |
|--------|-------------|----------|
| Field tables + annotated example | Table per schema + complete annotated YAML | ✓ |
| Annotated examples only | YAML with inline comments | |
| You decide | Claude picks | |

**User's choice:** Field tables + annotated example
**Notes:** Two schemas: model architecture and model config.

---

## LLM Examples Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Full end-to-end with output | Plan command + output + profile command, YAML configs inline | ✓ |
| Commands only | Plan and profile commands with file paths, no output | |
| You decide | Claude determines completeness | |

**User's choice:** Full end-to-end with output
**Notes:** Developers can copy-paste and run.

---

## Claude's Discretion

- CTP commands section depth
- Subcommand ordering
- kvcache command output example
- Number of model config variants

## Deferred Ideas

- CTP examples — deferred per REQUIREMENTS.md
- GDS tutorial — P2, deferred
- Adding new model architecture guide — P2, deferred
