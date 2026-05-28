# Agent And Backend Setup

Use this reference for agent construction, plugin inspection, backend
initialization, and backend memory-type readiness.

## Readiness Before Recipes

Confirm these from the user's installed runtime/source when possible:

- `import nixl` succeeds in the same environment that runs the workload.
- `nixl_agent` can be constructed.
- The requested backend appears in `agent.get_plugin_list()`.
- `agent.get_plugin_params(<backend>)` is available if backend parameters are
  needed.
- Backend creation succeeds.
- `agent.get_backend_mem_types(<backend>)` includes the requested memory type:
  `DRAM` for CPU tensors, `VRAM` for CUDA tensors, or the required storage type
  for file/object paths.

If any item is missing, return `Environment not ready` or
`Backend evidence missing` rather than implementing readiness-dependent API code.

## Minimal Backend Probe

Run only in the trusted environment where the workload already runs:

```python
from nixl import nixl_agent, nixl_agent_config

backend = "UCX"  # replace only with a source-backed/user-selected backend
agent = nixl_agent("probe", nixl_agent_config(backends=[]))

plugins = agent.get_plugin_list()
if backend not in plugins:
    raise RuntimeError(f"{backend} plugin unavailable; plugins={plugins}")

params = agent.get_plugin_params(backend)
agent.create_backend(backend, params)
mem_types = agent.get_backend_mem_types(backend)
if not mem_types:
    raise RuntimeError(f"{backend} created but reported no memory types")
```

Do not invent backend parameter values. If the user wants tuning, compare
`get_plugin_params()` with installed backend source/docs. If values remain
unknown, state the exact backend parameter evidence still needed.

## Agent Construction Pattern

```python
from nixl import nixl_agent, nixl_agent_config

cfg = nixl_agent_config(
    enable_prog_thread=True,
    enable_listen_thread=True,
    listen_port=5555,  # private/control-plane reachable address only
    backends=["UCX"],
)
agent = nixl_agent("target", cfg)
```

Use explicit listener ports for ready-to-run multi-process examples unless the
user's installed source proves that an ephemeral listener port is advertised and
usable by peers for the intended metadata/notification path.

## Memory Type Gate

Before a tensor recipe:

- CPU tensor path requires backend `DRAM` support.
- CUDA tensor path requires backend `VRAM` support plus CUDA tensor/device
  readiness in the same runtime.
- Storage/file/object paths require backend-specific metadata and memory-type
  evidence; use `references/storage-and-partial-metadata.md`.

If the backend reports only `DRAM`, do not provide a CUDA tensor transfer recipe
without additional source/runtime evidence.
