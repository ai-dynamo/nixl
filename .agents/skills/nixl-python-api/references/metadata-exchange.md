# Metadata Exchange

Use this reference for direct metadata exchange, listener-based metadata
exchange, remote metadata readiness, and metadata timeout diagnosis.

## Direct Control Plane

For same-process tests or an existing trusted out-of-band control plane, exchange
metadata bytes directly:

```python
target.add_remote_agent(initiator.get_agent_metadata())
initiator.add_remote_agent(target.get_agent_metadata())
```

Metadata bytes are not descriptor pickle, but they are still untrusted remote
input. Exchange them only through a trusted control plane or authenticated peer.

## Listener Path

Use explicit private/control-plane-reachable ports in examples. Avoid relying on
ephemeral listener behavior unless the user's installed source proves it for the
intended topology.

```python
# Target process
target_port = 5555
target = nixl_agent(
    "target",
    nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=target_port,
        backends=["UCX"],
    ),
)
target_reg = target.register_memory(target_tensor)
```

```python
# Initiator process
initiator_port = 5556
target_ip = "127.0.0.1"  # use a private/LAN IP for two hosts
initiator = nixl_agent(
    "initiator",
    nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=initiator_port,
        backends=["UCX"],
    ),
)
init_reg = initiator.register_memory(init_tensor)

initiator.send_local_metadata(ip_addr=target_ip, port=target_port)
initiator.fetch_remote_metadata("target", ip_addr=target_ip, port=target_port)
```

After metadata exchange, bound waiting is required:

```python
import time

def wait_until(predicate, label, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {label}")
        time.sleep(0.01)

wait_until(lambda: initiator.check_remote_metadata("target"), "target metadata")
```

If metadata never becomes available, do not change transfer logic first. Collect
listener IP/port, both agent names, backend creation evidence, plugin list, and
whether a framework is expected to own metadata exchange.

## Descriptor Exchange Over Notifications

The Python examples send serialized descriptors through notifications after
metadata is ready. This is a convenience pattern, not an authentication model:

```python
target_descs = target.get_xfer_descs(target_rows)
target.send_notif("initiator", target.get_serialized_descs(target_descs))
```

The receiver must bound waits and deserialize only trusted descriptor bytes.

## Partial Metadata

Partial metadata APIs exist in the fallback snapshot, but topology-specific
ordering and invalidation guidance must be matched to the user's installed
source and example. Use `references/storage-and-partial-metadata.md` and
state any missing topology/source evidence for partial metadata requests.
