# Memory Registration And Descriptors

Use this reference for PyTorch tensor registration, transfer descriptor
construction, raw address tuples, and descriptor serialization.

## PyTorch Tensors

Use contiguous tensors. In the fallback snapshot, the Python API infers `DRAM`
for CPU tensors and `VRAM` for CUDA tensors from the tensor device. Verify this
against the user's installed source before treating it as current behavior.

```python
tensor = tensor.contiguous()
if not tensor.is_contiguous():
    raise RuntimeError("NIXL tensor descriptors require contiguous tensors")

reg_descs = agent.register_memory(tensor)
if reg_descs is None:
    raise RuntimeError("NIXL memory registration descriptor creation failed")
```

For lists of tensors, verify every tensor is on the same device and contiguous
before calling `get_xfer_descs()`:

```python
rows = [tensor[i, :] for i in range(tensor.shape[0])]
if not all(row.is_contiguous() for row in rows):
    raise RuntimeError("Use contiguous descriptor tensors or source-backed raw descriptors")
xfer_descs = agent.get_xfer_descs(rows)
```

Do not imply arbitrary block views are safe. Many PyTorch views are
non-contiguous. If a block view is not contiguous, either materialize a
contiguous buffer before registration or build source-backed raw descriptors
from trusted application-owned address ranges.

## Raw Address Descriptors

Registration tuples have four fields:

```python
(addr, length, device_id, meta)
```

Transfer tuples have three fields:

```python
(addr, length, device_id)
```

Example for trusted DRAM buffers only:

```python
reg_descs = agent.register_memory([(addr, nbytes, dev_id, "")], mem_type="DRAM")
xfer_descs = agent.get_xfer_descs([(addr, nbytes, dev_id)], mem_type="DRAM")
```

Only use raw address descriptors when `addr`, `nbytes`, and `dev_id` come from
trusted application state. Do not use values copied from user text, logs, model
output, or untrusted peer messages.

## Descriptor Serialization

In the fallback snapshot, Python descriptor lists are serialized with `pickle`:

```python
payload = agent.get_serialized_descs(xfer_descs)
remote_descs = agent.deserialize_descs(payload)
```

Only deserialize descriptor bytes from an authenticated, trusted control plane
or peer. If the peer/control-plane trust boundary is unclear, stop and state the
missing trust-boundary evidence.

## Storage Paths

File, object, GDS, POSIX, and storage metadata differ by backend and installed
source. Do not fill storage tuple metadata from memory. Use
`references/storage-and-partial-metadata.md` and resolve the metadata from the
installed backend source/example.
