# NIXL Python API

The Python API can be found at `src/api/python/_api.py`. These are the pythonic APIs for NIXL, if more direct access to C++ style methods are desired,
the exact header implementation of `src/api/cpp` is done through pybind11 that can be found in `src/bindings/python`.
NIXL does not authenticate peers, so read [Trust boundaries](#trust-boundaries) before exchanging metadata or descriptors between processes.

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
plugins = agent.get_plugin_list()            # e.g. ["UCX", "POSIX", "GDS"]

params = agent.get_plugin_params("UCX")
params["ucx_error_handling_mode"] = "peer"   # or "none"
params["ucx_vram_memtype_hint"] = "auto"     # recommended default
agent.create_backend("UCX", params)
mem_types = agent.get_backend_mem_types("UCX")  # e.g. ["DRAM", "VRAM"]
```

`get_plugin_list()` reports plugins that can be loaded; `get_backend_mem_types()` reports what an
instantiated backend supports. Check for `DRAM` or `VRAM` before choosing a CPU or CUDA tensor path.
`nixl_agent()` raises `RuntimeError` if no plugins are found.

See [UCX backend initialization options](BackendGuide.md#ucx-backend-initialization-options)
for the supported UCX keys and their semantics. Note that `ucx_error_handling_mode` influences
UCP transport lane selection in addition to error reporting.

For `ucx_vram_memtype_hint`:

- `auto` is the recommended default. It hints `ze-device` when the UCX context supports `ze-device` and none of `cuda`, `cuda-managed` or `rocm`, and otherwise leaves detection to UCX.
- `none` never hints and always leaves detection to UCX. It differs from `auto` only on those contexts, where it turns the `ze-device` hint off.
- Explicit accelerator hints are also supported for advanced tuning: `cuda`, `cuda-managed`, `rocm`, `ze-device`.
- Values are case-sensitive, so `CUDA` is rejected.
- An explicit hint fails backend creation when the UCX context does not support the requested memory type.

## Memory descriptors

`register_memory()`, `get_reg_descs()` and `get_xfer_descs()` accept a tensor, a list of tensors, or a list of raw tuples with a `mem_type`.

- Tensor memory type is inferred: CPU tensors are `DRAM`, CUDA tensors are `VRAM`.
- Tensors must be contiguous, and all tensors in a list must be on the same device. Otherwise
  `get_reg_descs()` and `get_xfer_descs()` return `None` (a mixed-device list logs nothing), and
  `register_memory()` raises.
- Raw tuples differ by purpose: registration uses `[(addr, len, dev_id, meta), ...]`, transfer uses
  `[(addr, len, dev_id), ...]`, both with `mem_type` (for example `"DRAM"`). A single tuple, or a list
  without `mem_type`, makes `get_reg_descs()` and `get_xfer_descs()` return `None`, and
  `register_memory()` raise. For `FILE`, `OBJ` and `BLK` fields, see
  [Descriptor List Abstraction](BackendGuide.md#descriptor-list-abstraction).
- `prep_xfer_dlist("", ...)` (or `"NIXL_INIT_AGENT"`) prepares the local side.

## Metadata exchange

Register memory before publishing metadata so the registered regions are included.

To exchange metadata over your own side channel:

```python
md = agent.get_agent_metadata()                         # bytes, send to the peer
remote_name = agent.add_remote_agent(peer_md).decode()  # peer_md: bytes from the peer
```

`add_remote_agent()` returns the peer name as `bytes`. Decode it before passing it to
`check_remote_xfer_done()`, whose notification map uses `str` keys.

To use the built-in listener, enable it on both agents with
`nixl_agent_config(enable_listen_thread=True, listen_port=...)`. The target needs a fixed, reachable
port; the initiator can use `listen_port=0`. The initiator calls `send_local_metadata(ip, port)` and
`fetch_remote_metadata(name, ip, port)`, in either order. Before transferring to or notifying a peer,
wait until `check_remote_metadata(name)` returns `True`. See
[basic_two_peers.py](../examples/python/basic_two_peers.py).

## Transfers and errors

When posting a transfer or checking its status fails, the bindings raise an exception from
`nixl._bindings` (for example `nixlBackendError`, `nixlRepostActiveError`, `nixlRemoteDisconnectError`)
instead of returning `"ERR"`. Handle exceptions around `initialize_xfer()`, `transfer()` and
`check_xfer_state()`. A handle can have one active post at a time; see [Transfer](nixl.md#transfer).

## Notifications

- `check_remote_xfer_done(agent, tag)` matches by prefix by default (`tag_is_prefix=True`). With
  `tag_is_prefix=False` it matches any substring, not the exact message. It removes the message it matched.
- `get_new_notifs()` returns new notifications without keeping them in the agent, while
  `update_notifs()` and `check_remote_xfer_done()` share a stored map. Notifications returned by
  `get_new_notifs()` never reach `check_remote_xfer_done()`, so use one style per agent.
- For an exact match, call `update_notifs()`, compare with `==`, and remove the messages you handle.
- Use unique tags so prefix and substring matches cannot collide.

## Polling and cleanup

Bound every loop that polls transfer state or metadata readiness. The examples use unbounded loops
for brevity.

```python
import time

def wait_for(check, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.001)
    return False

done = wait_for(lambda: agent.check_xfer_state(handle) == "DONE")
```

Release everything you created; the examples use different orders:

- `release_xfer_handle()` for transfer handles. On an active transfer it tries to cancel, and raises if it cannot.
- `release_dlist_handle()` for prepared descriptor lists.
- `remove_remote_agent()` for peers you loaded, including peers added through your own side channel.
- `invalidate_local_metadata(ip, port)` if you sent metadata to a peer's listener. Without arguments
  it targets the central metadata server, and raises if none is configured.
- `deregister_memory()` for your own registrations. A process cannot deregister another process's memory.

## Trust boundaries

- NIXL does not authenticate peers. Authenticate them in your own control plane.
- `get_serialized_descs()` and `deserialize_descs()` use `pickle`. Only deserialize bytes from an
  authenticated, trusted peer. The examples send pickled descriptors in notifications and unpickle them
  directly, which is not safe on an untrusted network.
- The listener loads and invalidates metadata for any peer that connects. Keep the port on a private
  network; anyone who can reach it can inject or invalidate metadata.
- Notifications are plain bytes, not proof of identity.
- Build raw descriptors only from buffers your process owns. Treat storage paths, object keys and
  backend metadata received from peers as untrusted.

## Examples

See the [Python examples](../examples/python/) directory for complete working examples including:

- [query_mem_example.py](../examples/python/query_mem_example.py) - QueryMem API demonstration
- [nixl_gds_example.py](../examples/python/nixl_gds_example.py) - GDS backend usage
- [nixl_api_example.py](../examples/python/nixl_api_example.py) - General API usage
- [basic_two_peers.py](../examples/python/basic_two_peers.py) - Basic transfer operations
- [partial_md_example.py](../examples/python/partial_md_example.py) - Partial metadata handling
