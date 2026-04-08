# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.1 — NIXLBench and KVBench Documentation

**Shipped:** 2026-04-07
**Phases:** 6 | **Plans:** 9 | **Sessions:** ~4

### What Was Built
- 6 benchmarking documentation pages (968 lines total) under User Guide
- NIXLBench: overview, build (Docker + native), usage guide with 4 communication patterns, CLI reference, and troubleshooting
- KVBench: overview, build (Docker + Python venv), command reference for 5 subcommands, model config YAML schemas, DeepSeek R1 and Llama 3.1 end-to-end examples
- Full terminology normalization and cross-link audit across all new pages

### What Worked
- Phase-per-page structure worked well for documentation — each phase produced one or two focused pages
- Research phase before each content phase caught edge cases (e.g., etcd underscore vs hyphen convention difference)
- Terminology normalization as a final phase caught 2 missing cross-links that individual phases missed
- fern check validation in Phase 37 confirmed zero errors across all pages

### What Was Inefficient
- SUMMARY frontmatter `requirements_completed` not populated for early phases (32, 33) — caused bookkeeping gaps at audit time
- REQUIREMENTS.md traceability table not updated after each phase — 13/16 entries still showed "Pending" at milestone audit
- Directory path changed mid-milestone (Developer Guide → User Guide) requiring plan path updates in Phase 37

### Patterns Established
- Combined usage + troubleshooting into single page (user preference D-01) — reduces page count, improves discoverability
- Cross-link pattern: link backend names to User Guide pages on first mention per page
- CLI flag convention documentation: explicit note when two tools use different flag styles (underscores vs hyphens)

### Key Lessons
1. Update REQUIREMENTS.md traceability table after each phase completion, not just at audit time — prevents bookkeeping debt accumulation
2. Confirm directory structure decisions before Phase 1 content authoring — the Developer Guide → User Guide move created unnecessary plan updates
3. Documentation-only projects benefit from fern check as a continuous validation step, not just a final audit

### Cost Observations
- Model mix: ~70% opus, ~30% sonnet (research phases used sonnet)
- Sessions: ~4
- Notable: All 6 phases completed in a single day — documentation projects with clear source material (READMEs, --help output) move fast

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | ~10 | 11 | Initial site buildout, established Fern patterns |
| v1.1 | ~4 | 6 | Added benchmarking docs, established cross-link and terminology patterns |

### Top Lessons (Verified Across Milestones)

1. Terminology normalization should be a dedicated final phase — catches drift that individual phases miss
2. Cross-links to existing pages are better than duplicated content — reduces maintenance burden
