# NIXL Python API

The pybind11 bindings for the public facing NIXL API are available in `src/bindings/python`. These bindings implement the headers in the `src/api/cpp` directory.

The preferred way is to build it through meson-python, which will just let it be installed with pip. This can be done from the root nixl directory:

```bash
pip install .
```

## Python API Features

The Python bindings provide access to the full NIXL API including:

- **Agent Management**: Create and configure NIXL agents
- **Memory Registration**: Register and deregister memory/storage
- **Transfer Operations**: Create and manage data transfers
- **QueryMem API**: Query memory/storage information and accessibility
- **Backend Management**: Create and configure different backends (UCX, GDS, etc.)

## QueryMem Example for files

```python
import nixl._bindings as nixl_bindings

# Create an agent
config = nixl_bindings.nixlAgentConfig(enable_prog_thread=False)
agent = nixl_bindings.nixlAgent("my_agent", config)

# Create descriptor list with file paths
descs = nixl_bindings.nixlRegDList(nixl_bindings.FILE_SEG, False)
descs.addDesc((0, 0, 0, "/path/to/file.txt"))  # File path in metaInfo

# Query memory/storage information
resp = agent.queryMem(descs)

# Check results
for i, result in enumerate(resp):
    print(f"File {i}: accessible={result.accessible}")
    if result.accessible:
        print(f"  Size: {result.info.get('size')} bytes")
        print(f"  Mode: {result.info.get('mode')}")
```

For more examples, see [examples/python/](../examples/python/).

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
- [blocking_send_recv_example.py](../examples/python/blocking_send_recv_example.py) - Basic transfer operations
- [partial_md_example.py](../examples/python/partial_md_example.py) - Partial metadata handling