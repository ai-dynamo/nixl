# MILESTONES

## v1.1 NIXLBench and KVBench Documentation (Shipped: 2026-04-07)

**Phases completed:** 6 phases, 9 plans, 10 tasks

**Key accomplishments:**

- Complete usage guide with etcd coordination, four communication patterns (pairwise, many-to-one, one-to-many, TP), GDS and OBJ storage examples, 18-flag CLI reference, and four troubleshooting sections with cross-links to all backend User Guide pages
- KVBench overview page with nixlbench subprocess relationship and command categories, plus build page with Docker cross-link and Python venv tabs
- Complete KVBench commands.md page (494 lines) with all 5 subcommand references, CLI argument tables, dual YAML schema documentation, and DeepSeek R1 / Llama 3.1 end-to-end examples
- Zero terminology drift found across all 6 benchmarking pages; added 2 missing first-mention cross-links (etcd on kvbench/commands, UCX on nixlbench/build)
- All 6 benchmarking pages pass fern check with zero errors and zero terminology violations on final re-sweep

---

## v1.0 — Initial Documentation (Completed)

**Goal:** Launch the full NIXL documentation site on Fern with complete coverage of all existing NIXL capabilities.

**Phases:** 21–31 (documentation polish, terminology, backends, getting started, user guide, examples, API ref, resources, cross-page coherency, overview, stack diagrams, architecture diagrams, examples reorder/expand, backend support matrix)

**Shipped:**

- Getting Started section (Overview, Architecture, Quick Start, Contributing)
- User Guide: all 13 backends, ETCD metadata exchange, telemetry
- Developer Guide: all build paths (Docker, C++, Python, Rust), backend plugin development
- Examples: 6 examples covering key NIXL usage patterns
- API Reference: C++, Python, Rust, Device, Plugin APIs
- Resources: Environment Variables, Troubleshooting
- Full NVIDIA branding, custom typography, dark/light theme

**Last phase:** 31 (examples reorder and expand)

---
