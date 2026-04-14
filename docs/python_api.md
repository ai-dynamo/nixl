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

## Examples

See the [Python examples](../examples/python/) directory for complete working examples including:

- [query_mem_example.py](../examples/python/query_mem_example.py) - QueryMem API demonstration
- [nixl_gds_example.py](../examples/python/nixl_gds_example.py) - GDS backend usage
- [nixl_api_example.py](../examples/python/nixl_api_example.py) - General API usage
- [basic_two_peers.py](../examples/python/basic_two_peers.py) - Basic transfer operations
- [partial_md_example.py](../examples/python/partial_md_example.py) - Partial metadata handling

## Backend initialization parameters

Backends can expose initialization parameters through `get_plugin_params(backend)`.
You can override values before calling `create_backend`.

```python
params = agent.get_plugin_params("UCX")

# Common UCX options.
params["ucx_error_handling_mode"] = "peer"   # or "none"
params["ucx_vram_memtype_hint"] = "auto"     # recommended default

agent.create_backend("UCX", params)
```

For `ucx_vram_memtype_hint`:

- `auto` is the recommended default.
- `none` disables NIXL memory-type hinting and leaves detection to UCX.
- Explicit accelerator hints are also supported for advanced tuning.
- Values are case-sensitive.
