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

- `auto` is the recommended default.
- `none` disables NIXL memory-type hinting and leaves detection to UCX.
- Explicit accelerator hints are also supported for advanced tuning: `cuda`, `cuda-managed`, `rocm`, `ze-device`.
- Values are case-sensitive, so `CUDA` is rejected.
- An explicit hint fails backend creation when the UCX context memory types cannot be queried, or when the queried UCX context does not advertise the requested memtype. `auto` and `none` skip hinting instead of failing.

## Examples

See the [Python examples](../examples/python/) directory for complete working examples including:

- [query_mem_example.py](../examples/python/query_mem_example.py) - QueryMem API demonstration
- [nixl_gds_example.py](../examples/python/nixl_gds_example.py) - GDS backend usage
- [nixl_api_example.py](../examples/python/nixl_api_example.py) - General API usage
- [basic_two_peers.py](../examples/python/basic_two_peers.py) - Basic transfer operations
- [partial_md_example.py](../examples/python/partial_md_example.py) - Partial metadata handling
