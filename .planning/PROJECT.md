# NIXL Documentation

## What This Is

The official documentation site for NVIDIA Inference Xfer Library (NIXL), built on the Fern platform. NIXL accelerates point-to-point communications in AI inference frameworks (e.g., NVIDIA Dynamo) and abstracts over memory types (CPU/GPU) and storage (file, block, object store) through a modular plugin architecture. This documentation site serves developers integrating or extending NIXL, including benchmarking tools for performance evaluation.

## Core Value

Developers can find accurate, complete documentation for every NIXL capability — from quick start through advanced backend development and performance benchmarking — in one place.

## Requirements

### Validated

- ✓ Getting Started section (Overview, Architecture, Quick Start, Contribution Guide) — v1.0
- ✓ User Guide: NIXL Backends (UCX, Libfabric, Mooncake, UCCL-P2P, DOCA GPUNetIO, GDS, GDS-MT, POSIX, HF3FS, OBJ, Azure Blob, GUSLI) — v1.0
- ✓ User Guide: Metadata Exchange with ETCD, Telemetry Guide — v1.0
- ✓ Developer Guide: Building from Source (Docker, C++/Meson, Python, Rust bindings), Building a Backend Plugin — v1.0
- ✓ Examples: Basic Transfer, GDS Direct Storage, Remote Storage, NIXL-EP, ETCD Metadata Exchange, Telemetry Reader — v1.0
- ✓ API Reference: C++, Python, Rust, Device, Plugin (Southbound) APIs — v1.0
- ✓ Resources: Environment Variables, Troubleshooting — v1.0
- ✓ Fern platform setup: custom NVIDIA branding, typography, colors, navbar, footer — v1.0
- ✓ Terminology standards and cross-page coherence — v1.0
- ✓ NIXLBench documentation: overview, build (Docker + native), usage guide, CLI reference, troubleshooting — v1.1
- ✓ KVBench documentation: overview, build (Docker + venv), command reference (5 subcommands), model config schemas, LLM examples — v1.1

### Active

(None — awaiting next milestone definition)

### Out of Scope

- Doxygen-generated C++ API detail pages — rendered separately via Doxygen, not Fern
- edit-this-page GitHub integration — ai-dynamo/nixl GitHub repo does not yet contain the fern/ directory
- AI-powered search — requires Fern Pro plan / Dashboard enablement
- NIXLBench full 70+ flag CLI reference — deferred; coverage via usage guide sufficient
- NIXLBench backend-specific deep-dive examples — backend User Guide pages already exist
- KVBench CTP examples — deferred
- KVBench GDS tutorial — deferred
- KVBench adding new model architecture guide — deferred

## Context

- Built on Fern documentation platform, hosted at nixl.docs.buildwithfern.com
- Source lives in fern/ (config) and docs/ (markdown content)
- NVIDIA branding: NVIDIA green (#76B900), NVIDIA Sans font, dark/light theme
- v1.0 shipped Phases 21–31: all sections polished and cross-referenced
- v1.1 shipped Phases 32–37: 6 benchmarking pages (968 lines total), 18 requirements satisfied
- NIXLBench: C++ benchmark tool using etcd coordination, supports multiple network/storage backends and 4 communication patterns
- KVBench: Python tool that generates NIXLBench commands for KV cache transfer testing across LLM architectures (DeepSeek R1, Llama 3.1)

## Constraints

- **Platform**: Fern — all pages must be valid MDX/Markdown compatible with Fern's parser
- **Branding**: NVIDIA brand guidelines — NVIDIA green accent, official fonts, no unauthorized logos
- **Content**: Must reflect actual NIXLBench/KVBench capabilities — no invented features

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fern platform | Structured, versioned docs with good DX and NVIDIA-compatible theming | ✓ Good |
| Separate products structure (versions at v1.0.0) | Enables future versioned docs as NIXL evolves | ✓ Good |
| docs/ for content, fern/ for config | Clean separation; aligns with Fern's recommended structure | ✓ Good |
| edit-this-page disabled | fern/ not yet in public ai-dynamo/nixl repo | — Pending |
| Benchmarking under User Guide (not Developer Guide) | User decision: benchmarking is user-facing, not dev-only | ✓ Good |
| Combined usage + troubleshooting page | User decision D-01: fewer pages, better discoverability | ✓ Good |
| NIXLBench CLI essentials only (not full 70+ flags) | Usage guide covers essentials; full reference deferred | ✓ Good |
| KVBench README CLI args only | Backend-specific args are passthrough, documented on backend pages | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-07 after v1.1 milestone*
