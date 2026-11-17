# Skills

NIXL Agent Skills live here as direct child directories. Each skill directory
must match the `name` field in its `SKILL.md` frontmatter so it remains
compatible with the Agent Skills specification.

## Available Skills

- `nixl-install`: install, import, plugin, CUDA/wheel, native-library, and
  framework connector readiness diagnosis.
- `nixl-backend-selector`: source-backed backend selection for Dynamo, vLLM,
  SGLang, RDMA, EFA, GDS, POSIX, and object storage.
- `nixl-python-api`: source-matched NIXL Python API help for agents,
  descriptors, metadata, transfers, polling, and cleanup.
- `nixl-cpp-api`: source-matched NIXL C++ API help for agents, descriptors,
  metadata, transfers, notifications, and cleanup.
- `nixl-code-review`: review NIXL or NIXL-facing PRs, diffs, and patches for
  correctness, concurrency, lifecycle, observability, tests, and scope.
- `nixl-debug-session`: structure developer debugging notes across
  reproduction, evidence, root cause, fix, and verification.

## Layout

Installers should consume this directory without category wrapper folders. For
example, `nixl-install` is located at `.agents/skills/nixl-install`.
