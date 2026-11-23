# NIXL Python API

The Python API can be found at `src/api/python/_api.py`. These are the pythonic APIs for NIXL, if more direct access to C++ style methods are desired,
the exact header implementation of `src/api/cpp` is done through pybind11 that can be found in `src/bindings/python`.

## Python API Features

The Python bindings provide access to the full NIXL API including:

- **Agent Management**: Create and configure NIXL agents
- **Memory Registration**: Register and deregister memory/storage
- **Transfer Operations**: Create and manage data transfers
- **Notification Monitors**: Wait for incoming notifications
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

## Waiting for notifications

On Linux, notification monitors expose a pollable file descriptor. They need a
backend that delivers notifications from its own threads, which
`agent.get_notif_callback_support()` reports as
`nixl_notif_callback_support_t.NIXL_NOTIF_CALLBACK_BACKGROUND`, also available as
`nixl.NIXL_NOTIF_CALLBACK_BACKGROUND`. Without it, `add_notif_monitor()` raises
`nixlNotSupportedError`. With UCX, `enable_prog_thread=True` on the receiving
agent provides background progress and is the `nixl_agent_config` default. When
using `device_proxy=true`, the proxy provides background progress and the
receiving agent must use `enable_prog_thread=False`.

```python
from nixl import nixl_notification_monitor

# After creating the agents and exchanging metadata:
with nixl_notification_monitor() as incoming:
    target.add_notif_monitor(incoming, b"kv:")
    source.send_notif(target.name, b"kv:ready")

    for local_agent, remote_agent, message in incoming.wait_any(timeout_ms=5000):
        print(local_agent, remote_agent, message)
```

- `wait_any()` returns all queued notifications once at least one arrives.
  `timeout_ms=0` drains without blocking and `-1` waits indefinitely. A timed wait
  returns an empty list if no notification arrives. Use one consumer per monitor.
- A notification monitor can subscribe to multiple agents and byte prefixes.
  Nonempty prefixes for an agent must be disjoint. An empty prefix receives
  messages with no matching nonempty prefix, and an agent has at most one.
  Adding a prefix that overlaps an existing subscription of that agent, or a
  second default, raises `nixlInvalidParamError`. Call
  `agent.remove_notif_monitor(monitor, prefix)` first to replace a subscription.
  It returns whether the monitor had that subscription. Already queued messages
  remain available.
- Unconsumed notifications remain available through `get_new_notifs()`.
  A monitor queues up to 4,096 notifications, or `max_queued` when created as
  `nixl_notification_monitor(max_queued=...)`. Further matching messages go to
  the normal notification queue instead. The next `wait_any()` after the first
  overflow logs one warning through the `nixl` Python logger. Later overflows of
  the same monitor are not logged again, and `overflow_count()` reports the total
  number of redirected messages. Backends without callback support keep their
  normal polling behavior.
- Use `fd()` or `fileno()` with poll or an event loop. The descriptor is borrowed.
  When it becomes readable, call `wait_any(timeout_ms=0)` to drain the monitor.
  Remove any external reader before closing the monitor. `close()`, including at
  the end of a `with` block, wakes a blocking wait with `RuntimeError`, unregisters
  all subscriptions, and releases the descriptor once no wait still uses it.
  Destroying an agent removes its subscriptions but leaves the monitor open, so
  it can still receive from other agents or be reused. Agent destruction does not
  wake a wait, even when the last subscribed agent is destroyed. Use a finite
  timeout or close the monitor from another thread to end such a wait.

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
