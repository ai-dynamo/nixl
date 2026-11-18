# NIXL Python API

The Python API can be found at `src/api/python/_api.py`. These are the pythonic APIs for NIXL, if more direct access to C++ style methods are desired,
the exact header implementation of `src/api/cpp` is done through pybind11 that can be found in `src/bindings/python`.

## Python API Features

The Python bindings provide access to the full NIXL API including:

- **Agent Management**: Create and configure NIXL agents
- **Memory Registration**: Register and deregister memory/storage
- **Transfer Operations**: Create and manage data transfers
- **QueryMem API**: Query memory/storage information and accessibility
- **Backend Management**: Create and configure different backends (UCX, GDS, etc.)

## Installation

### From PyPI

The nixl python API and libraries, including UCX, are available directly through PyPI:

```bash
pip install nixl
```

For AMD ROCm (x86_64 only), install `rocm-nixl` instead:

```bash
pip install rocm-nixl
```

With PyTorch:

```bash
pip install "rocm-nixl[rocm10.1]" --extra-index-url https://stable.repo.amd.com/rocm/whl-next/
```

One wheel carries the backends for the supported ROCm versions (7.14 and 10.1), and `import nixl` selects the one matching the installed ROCm at import time. The wheel itself does not depend on PyTorch. The `rocm<X>.<Y>` extra adds AMD's ROCm build of PyTorch for that ROCm version; it needs AMD's package index on the command line because that build is not on PyPI (for ROCm 7.14: `pip install "rocm-nixl[rocm7.14]" --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/`). Never install the default `torch` from PyPI on a ROCm system: it is the CUDA build. The extra needs pip 23.3 or newer (`python -m pip install -U pip`); older pip ignores it with a warning and installs no PyTorch. `rocm-nixl` and `nixl` both provide the `nixl` import package and must not be installed in the same environment.

A ROCm install (7.14 or 10.1) is required on the host, with its libraries resolvable by the dynamic loader before Python starts (e.g. `export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH`; use `/opt/rocm/core-<X.Y>/lib` if `/opt/rocm/lib` does not exist). The ROCm version is read from an installed ROCm build of PyTorch, or from `$ROCM_PATH` (default `/opt/rocm`): `.info/version` or `core/.info/version`. If ROCm is not installed under `/opt/rocm`, e.g. only `/opt/rocm/core-<X.Y>` exists, also set `ROCM_PATH` to that prefix (e.g. `export ROCM_PATH=/opt/rocm/core-7.14`); if no version can be detected, the backend for the newest ROCm version is used.

### From Source

To build from source, follow the main build instructions in the README.md, then install the Python bindings:

```bash
# From the root nixl directory
pip install .
```

## Backend initialization parameters

Backends expose their initialization parameters through `get_plugin_params(backend)`, which
returns the defaults. Override the values you need before passing the map to `create_backend`:

```python
from nixl import nixl_agent, nixl_agent_config

# backends=[] leaves backend creation to the caller. The default config
# initializes UCX, and a backend can only be created once per agent.
agent = nixl_agent("example_agent", nixl_agent_config(backends=[]))

params = agent.get_plugin_params("UCX")
params["ucx_error_handling_mode"] = "peer"   # or "none"
params["ucx_vram_memtype_hint"] = "auto"     # recommended default
agent.create_backend("UCX", params)
```

See [UCX backend initialization options](BackendGuide.md#ucx-backend-initialization-options)
for the supported UCX keys and their semantics. Note that `ucx_error_handling_mode` influences
UCP transport lane selection in addition to error reporting.

For `ucx_vram_memtype_hint`:

- `auto` is the recommended default. It hints `ze-device` when the UCX context supports `ze-device` and none of `cuda`, `cuda-managed` or `rocm`, and otherwise leaves detection to UCX.
- `none` never hints and always leaves detection to UCX. It differs from `auto` only on those contexts, where it turns the `ze-device` hint off.
- Explicit accelerator hints are also supported for advanced tuning: `cuda`, `cuda-managed`, `rocm`, `ze-device`.
- Values are case-sensitive, so `CUDA` is rejected.
- An explicit hint fails backend creation when the UCX context does not support the requested memory type.

## Examples

See the [Python examples](../examples/python/) directory for complete working examples including:

- [query_mem_example.py](../examples/python/query_mem_example.py) - QueryMem API demonstration
- [nixl_gds_example.py](../examples/python/nixl_gds_example.py) - GDS backend usage
- [nixl_api_example.py](../examples/python/nixl_api_example.py) - General API usage
- [basic_two_peers.py](../examples/python/basic_two_peers.py) - Basic transfer operations
- [partial_md_example.py](../examples/python/partial_md_example.py) - Partial metadata handling
