<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scality AI Connector

An accelerated engine of the NIXL OBJ backend for Scality AI Connector
endpoints. Each transfer is split into two planes:

- **Data plane (RDMA):** the object bytes move directly between the local
  buffer (DRAM or GPU memory) and the storage server over RDMA DC (Dynamically
  Connected), zero-copy, so GPU memory needs no staging copy through the host.
  The storage server performs the one-sided RDMA operations; this side only
  registers memory.
- **Control plane (HTTP):** a header-only request tells the server *what* to do
  (`PUT`/`GET` of an object). The RDMA descriptor, which describes the registered
  buffer range, travels in an `x-scal-rdma` header. The body is empty. The
  RDMA descriptor is fully compatible with the cuObject one: the connector
  builds it itself with libibverbs, in the same DC format the cuObject library
  generates, so a server that accepts cuObject descriptors accepts it unchanged.

```text
  local buffer (DRAM/VRAM) <----- RDMA (server-driven) -----> Scality endpoint
                                                                   ^
  control: HTTP PUT/GET {endpoint}/{object_id}                     |
           header  x-scal-rdma: <descriptor> ----------------------+
           body    (empty)
```

The connector speaks Scality's own REST dialect. It is **not** an S3 client and
uses neither the AWS SDK nor AWS authentication.

## Prerequisites

- An RDMA NIC with DC support (NVIDIA ConnectX, `mlx5`), reachable by the
  Scality endpoint.
- For GPU memory: the `nvidia_peermem` kernel module (GPUDirect RDMA peer
  memory).
- A reachable Scality AI Connector endpoint.
- Build dependencies: `libcurl`, `libibverbs` and `libmlx5`.

## Building

The engine is part of NIXL's standard Meson build and is compiled into the OBJ
plugin when its dependencies are found at configure time:

```sh
meson setup build
ninja -C build
```

Otherwise it is skipped, and selecting it at run time reports
`Accelerated engine type 'scality_ai_connector' is not available (not compiled)`.

## Configuration

The engine is selected and configured through the backend's `customParams`.

**Required**

| Parameter | Meaning |
|---|---|
| `accelerated` | Must be `"true"` to request an accelerated engine |
| `type` | Must be `"scality_ai_connector"` to select this engine |
| `endpoint_override` | Base URL of the connector, e.g. `http://10.0.0.1:10000` |

**HTTP requests (optional)**

| Parameter | Default | Meaning |
|---|---|---|
| `max_inflight` | `min(512, RLIMIT_NOFILE / 4)` | Cap on requests running at once. Each holds a connection and therefore a file descriptor, so the default is clamped to a share of `RLIMIT_NOFILE`; excess requests queue and start as slots free. `0` disables the cap. |
| `num_threads` | `max(2, cpu_threads / 4)` | Size of the callback worker pool (see [Concurrency model](#concurrency-model)). |
| `request_timeout_ms` | `2000` | Limit on each request, connection included (connecting itself is limited to 1 s). A stalled request fails after this instead of holding its transfer; for a cache, falling back is faster than waiting. |

**RDMA (optional)**

| Parameter | Default | Meaning |
|---|---|---|
| `rdma_dc_key` | from `cufile.json`, else `0xffeeddcc` | DC access key the server's DCI side must present, in hex (e.g. `0xffeeddcc`). |
| `rdma_nics` | from `cufile.json` | NICs to register memory on: comma-separated IPv4 addresses or device names, e.g. `10.10.40.208,10.10.48.208` or `mlx5_1,mlx5_2`. |
| `rdma_sl` | `3` | RoCE service level (0-15). Selects the egress priority under PCP trust. See [RoCE priority](#roce-priority). |
| `rdma_traffic_class` | `106` | RoCE traffic class, the IP ToS byte (0-255, DSCP = value >> 2). Selects the priority under DSCP trust. See [RoCE priority](#roce-priority). |

Requests are issued to `{endpoint_override}/{object_id}`.

Example (parameters as passed to the OBJ backend):

```ini
accelerated       = true
type              = scality_ai_connector
endpoint_override = http://10.0.0.1:10000
rdma_nics         = mlx5_1,mlx5_2
```

When `rdma_nics` or `rdma_dc_key` is unset, the value is read from the
`rdma_dev_addr_list` and `rdma_dc_key` keys of `cufile.json`
(`$CUFILE_ENV_PATH_JSON`, else `/etc/cufile.json`), so a host already configured
for GPUDirect Storage RDMA needs no extra settings. The NIC list is only needed
once DRAM or VRAM is registered.

## Example

This Python example writes a GPU buffer to an object, then reads the object back
into the buffer.

```python
import torch
from nixl import nixl_agent, nixl_agent_config

agent = nixl_agent("client", nixl_agent_config(backends=[]))
agent.create_backend(
    "OBJ",
    {
        "accelerated": "true",
        "type": "scality_ai_connector",
        "endpoint_override": "http://10.0.0.1:10000",
        "rdma_nics": "mlx5_1,mlx5_2",
    },
)

size = 8 * 1024 * 1024
buf = torch.ones(size, dtype=torch.uint8, device="cuda:0")
local = agent.register_memory(buf)
# One object with the key "block-0". Transfers refer to it by its device ID, 0.
obj = agent.register_memory([(0, size, 0, "block-0")], "OBJ")


def run(op):
    # The NIC reads and writes the buffer outside the CUDA stream, so the GPU work
    # on the buffer must be finished first.
    torch.cuda.synchronize(buf.device)
    # The remote agent is the agent itself: the object store is local to it.
    handle = agent.initialize_xfer(op, local.trim(), obj.trim(), "client")
    state = agent.transfer(handle)
    while state == "PROC":
        state = agent.check_xfer_state(handle)
    agent.release_xfer_handle(handle)
    assert state == "DONE", f"{op} failed"


run("WRITE")  # buffer -> object
buf.zero_()
run("READ")  # object -> buffer
assert bool((buf == 1).all())

agent.deregister_memory(obj)
agent.deregister_memory(local)
```

## Concurrency model

A single poller thread drives all HTTP requests through libcurl's multi
interface, so the in-flight request count does not depend on `num_threads`
(the bulk data moves over RDMA; HTTP only carries the header-only command). Each
completed request's callback runs in a worker pool sized by `num_threads`, so a
slow callback cannot stall the poller. The default is sufficient for
lightweight callbacks.

At most `max_inflight` requests run at once; the rest wait in a queue and start
as running ones complete. At teardown the client logs the peak number of
running and waiting requests: a non-zero `peak_pending` means the cap was the
binding constraint.

The cap is per backend instance, and every running request holds its own
connection. A host running N workers (for example one vLLM process per GPU,
each with its own NIXL agent) therefore opens up to N × `max_inflight`
connections to the endpoint: 16 workers at the default of 512 is 8192. If the
endpoint limits connections per client, lower `max_inflight` accordingly.

## Multi-NIC

The connector opens one DC target per NIC. It registers each buffer on every
NIC, with one memory region per NIC that covers the whole buffer. The NIC a
request travels on is chosen when its RDMA descriptor is built, so consecutive
requests against one buffer spread across the rails: each request goes to the
NIC that has carried the fewest so far.

Bonded (LAG) NICs are detected from sysfs; one DCT QP is created per physical
port and RDMA descriptors cycle through them, so the distinct DCT numbers feed the
hardware bond hash (DCT QPs cannot be pinned to a physical port directly).

## RoCE priority

The defaults follow NVIDIA's standard lossless RoCE configuration: RoCE on
priority 3 with PFC enabled, marked DSCP 26. `rdma_sl=3` selects priority 3
when switches and NICs trust PCP (VLAN-tagged traffic); `rdma_traffic_class=106`
(DSCP 26 plus the ECN-capable bit) selects it when they trust DSCP. On a fabric
configured differently, set both to match it.

These settings only mark what this side sends: the data of a WRITE, which the
server reads from the local buffer, and RDMA acknowledgements. The data of a
READ is written into the local buffer by the server, so its priority follows
the server's own RoCE settings.

## How a transfer works

1. **Register memory.** Local DRAM or VRAM buffers are registered for RDMA on
   every NIC; remote object segments are mapped to their object ids.
2. **Prepare.** For each piece of the transfer the connector builds an RDMA
   descriptor for the buffer range (no data moves yet).
3. **Post.** It issues the HTTP `PUT`/`GET` for the object id, carrying the
   RDMA descriptor in the `x-scal-rdma` header, and returns immediately. The server
   performs the RDMA read or write on the buffer.
4. **Poll.** The transfer is checked for completion until all parts succeed or
   one reports an error.

## Limitations

- **DC transport only**; there is no RC fallback.
- The cuObject RDMA descriptor's SIZE field is 32 bits, so a single request is
  at most 4 GiB - 1 byte (0xFFFFFFFF == 4294967295).
- A buffer can be registered more than once only with the same address and
  length. A registration that starts at the same address as an existing one but
  has a different length is refused. Register the buffer once, or register
  sub-ranges that start at different addresses.
- After a failed or timed-out request the server may still complete its RDMA on
  the buffer, so do not reuse the buffer for other data.

## Troubleshooting

| Symptom (in logs) | Potential cause / fix |
|---|---|
| `'endpoint_override' parameter is required` | Set `endpoint_override` in `customParams`. |
| `RDMA requires a NIC list` | Set `rdma_nics`, or `rdma_dev_addr_list` in `cufile.json`. |
| `ibverbs_dc: no RDMA device found for NIC` | The IPv4 address or device name matches no local RDMA device. |
| `restClient: first connect failure. fds open=...` | Open descriptors close to the soft limit: lower `max_inflight` or raise `ulimit -n`. Far below it: the endpoint could not be reached; see the `curl says` text in the log. |
| `<op>: failed url=<url> curl_code=<n> http_code=<n>` | The HTTP call failed. A non-2xx `http_code` comes from the server; check the object id and endpoint URL. |

## For contributors

| File | Role |
|---|---|
| `engine_impl.{h,cpp}` | `ScalityObjEngineImpl`: the engine (register / prepare / post / check). |
| `client.{h,cpp}` | `restClient`: the libcurl control-plane client, behind the `iRestClient` interface (also the test-injection seam). |
| `rdma_descriptor_provider.h` | `iRdmaDescriptorProvider`: the data-plane interface. |
| `dc_descriptor_provider.{h,cpp}` | `dcDescriptorProvider`: in-process libibverbs DC implementation. |
| `dc_descriptor.h` | DC RDMA descriptor wire format, as generated by cuObject, free of libibverbs. |
| `cufile_nics.h` | `cufile.json` readers for `rdma_dev_addr_list` and `rdma_dc_key`. |
