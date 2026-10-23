<!--
Copyright 2026 Everpure, Inc.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NIXL Everpure accelerated S3 engine

`everpure` is the accelerated OBJ engine for S3-over-RDMA using the CUDA
Toolkit's cuObject client library (`cuobjclient`, `<cuobjclient.h>`), driven
through the same AWS S3 SDK plumbing as the rest of the OBJ backend. It
targets any S3-compatible endpoint that speaks the cuObject RDMA descriptor
protocol described below - `type: everpure` selects this engine regardless
of which such endpoint you're talking to. Its defaults are tuned for
FlashBlade, the reference implementation it was built against, but every
protocol-specific detail (header names, the confirmation check, the
checksum value) is overridable via [Backend
parameters](#backend-parameters)/[Environment
variables](#environment-variables) so another cuObject-compatible endpoint
can reuse this client without a fork.

If a FlashBlade data VIP is reachable but its RDMA feature flag isn't turned
on, use the plain `s3` engine (`accelerated` unset or `false`) instead - a
non-RDMA client works against the same endpoint.

## Wire protocol

This engine's default headers and general protocol follow the [S3-over-RDMA
protocol
spec](https://github.com/KiranModukuri/aws-c-s3/blob/nvidia_rdma/RDMA_PROTOCOL_SPEC.md).
It expects its RDMA endpoint to speak the following contract on top of
ordinary S3:

- The RDMA descriptor travels in a request header (`rdma_token_header`,
  default `x-amz-rdma-token`) as the **bare** cuObject descriptor string
  produced by `cuMemObjGetRDMAToken`, with nothing appended to it.
- The request body is empty (`Content-Length: 0`) since the payload moves
  over RDMA rather than the HTTP connection. RDMA PUT always writes the
  whole object from byte zero, so a non-zero write offset is rejected
  before a request is ever built.
- The request-body-checksum header (`x-amz-content-sha256`) carries a fixed
  value (`content_sha256_value`, default `UNSIGNED-PAYLOAD`) rather than a
  real checksum: the AWS SDK computes request checksums from the literal
  body bytes it's about to send, so a checksum computed over the (empty)
  HTTP body would never match the real payload moved over RDMA.
- A successful response carries a confirmation header (`rdma_reply_header`,
  default `x-amz-rdma-reply`). Its absence despite an HTTP-level success
  means the endpoint fell back to ordinary S3 semantics instead of actually
  moving data over RDMA, and this client treats that as a failure.

This engine's client (`client.cpp`) enforces the client-side half of this
contract before ever building a request: it will not send a token derived
from an empty descriptor, will not accept a nonzero PUT offset, and applies
the header/value defaults above unless overridden.

### FlashBlade specifics

FlashBlade implements the spec's default protocol described above.
Beyond that baseline, its implementation adds the following specifics:

- FlashBlade splits the RDMA token header value on `:` and requires
  **exactly 7 fields**; anything extra (for example a `:start_addr:size`
  suffix tacked on by a caller) is rejected with `InvalidRDMAToken`. Fields
  0 and 1 are the hex-encoded RDMA buffer start address and size, parsed
  directly by the array; the remaining fields are opaque to the client.
- A successful GET carries `x-amz-rdma-reply: 200` (or `206` for a ranged
  read) plus `x-amz-rdma-bytes-transferred`; a `501` reply means the array
  declined RDMA for that request.
- RDMA requests must be plain HTTP, not HTTPS, and the client must connect
  over IPv4 - FlashBlade rejects RDMA over HTTPS or IPv6.
- RDMA is gated behind a feature flag on the array; if it isn't enabled,
  every RDMA request is rejected up front.

## Dependencies

This engine requires aws-sdk-cpp. This dependency and steps to install it
are provided in the OBJ backend documentation.

This engine requires the CUDA Toolkit (which provides `cuobjclient`) version
13.1.1 or later. See the [CUDA GDS Install and Setup
guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## Backend parameters

In addition to the parameters documented for the OBJ backend generally, set
these to select the Everpure engine:

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `accelerated` | Enable an accelerated engine | `false` | Yes (`true`) |
| `type` | Vendor engine to use | - | Yes (`everpure`) |
| `resp_checksum` | AWS SDK response checksum validation (`required`/`supported`) | `required` | No |
| `req_checksum` | AWS SDK request checksum policy (`required`/`supported`) | `required` | No |
| `content_sha256_value` | Value sent for the `x-amz-content-sha256` header; set to an empty string to omit the header | `UNSIGNED-PAYLOAD` | No |

See [Environment variables](#environment-variables) below for the RDMA
token/confirmation header names and connection pool/timeout tuning.

`resp_checksum` defaults to `required` (`ResponseChecksumValidation::WHEN_REQUIRED`)
because RDMA GET responses arrive with an empty body and no checksum
headers to validate against - matching FlashBlade's own restriction that RDMA
responses never carry checksum headers. Pass it explicitly to override.

`req_checksum` defaults to `required` (`RequestChecksumCalculation::WHEN_REQUIRED`)
because RDMA PUT requests also carry an empty body - the real payload moves
over RDMA - so a checksum computed over that body would never match the
object FlashBlade actually receives. `WHEN_REQUIRED` is the more
conservative of the two supported settings, since `WHEN_SUPPORTED` would
have the SDK attach one proactively. Pass it explicitly to override.

```cpp
nixl_b_params_t params = {
    {"access_key", "..."},
    {"secret_key", "..."},
    {"bucket", "my-bucket"},
    {"endpoint_override", "http://<rdma-capable-s3-endpoint>"},
    {"scheme", "http"},
    {"use_virtual_addressing", "false"},
    {"accelerated", "true"},
    {"type", "everpure"},
};
agent.createBackend("OBJ", params);
```

The endpoint's data VIP must be reachable over the RDMA fabric (RoCEv2)
from the host running NIXL, and RDMA support must be enabled on that
endpoint (see [FlashBlade specifics](#flashblade-specifics) above for the
exact requirement there).

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NIXL_EVERPURE_RDMA_TOKEN_HEADER` | Request header carrying the RDMA descriptor | `x-amz-rdma-token` |
| `NIXL_EVERPURE_RDMA_REPLY_HEADER` | Response header confirming the request was actually served over RDMA | `x-amz-rdma-reply` |
| `NIXL_EVERPURE_MAX_CONNECTIONS` | Size of this engine's HTTP connection pool | AWS SDK default (25) |
| `NIXL_EVERPURE_CONNECT_TIMEOUT_MS` | TCP connect timeout, in milliseconds | AWS SDK default (1000) |
| `NIXL_EVERPURE_REQUEST_TIMEOUT_MS` | Stall timeout: aborts a transfer once it has moved under 1 byte/sec for this long, in milliseconds | AWS SDK default (3000) |

Leaving `NIXL_EVERPURE_RDMA_TOKEN_HEADER`/`NIXL_EVERPURE_RDMA_REPLY_HEADER`
unset picks the FlashBlade-protocol default above; setting either to an
empty string is rejected with an error rather than treated as meaningful,
since an empty header name doesn't map to any well-defined behavior. This
engine talks to any S3 endpoint that accepts a cuObject RDMA descriptor in
a request header, so a different endpoint speaking that same protocol with
different header names can reuse it by setting these instead of relying on
the defaults above.

`NIXL_EVERPURE_RDMA_REPLY_HEADER` guards against a request succeeding at the
HTTP level without actually moving data over RDMA - for example if the
endpoint didn't recognize the token header and fell back to ordinary S3
semantics. A response missing this header is treated as a failure even
though the underlying HTTP request succeeded.

`NIXL_EVERPURE_MAX_CONNECTIONS`, `NIXL_EVERPURE_CONNECT_TIMEOUT_MS`, and
`NIXL_EVERPURE_REQUEST_TIMEOUT_MS` apply only to this engine, not the
default S3 engine, and tune this engine's connection pool size and
timeouts independently of the AWS SDK's own defaults (25 connections, a 1s
connect timeout, a 3s stall timeout). Raise these for high-concurrency
workloads.

Each of these can also be set via a NIXL TOML config file instead of the
process environment - see the OBJ backend documentation's configuration
section for the file's resolution order (`NIXL_CONFIG_FILE`, then
`~/.nixl.cfg`, then `/etc/nixl.cfg`). A set environment variable always
takes priority over the same key in the TOML file. Example:

```toml
NIXL_EVERPURE_RDMA_TOKEN_HEADER = "x-amz-rdma-token"
NIXL_EVERPURE_RDMA_REPLY_HEADER = "x-amz-rdma-reply"
NIXL_EVERPURE_MAX_CONNECTIONS = 64
NIXL_EVERPURE_CONNECT_TIMEOUT_MS = 3000
NIXL_EVERPURE_REQUEST_TIMEOUT_MS = 10000
```

## cuObject client configuration

`cuobjclient` needs to know which local RDMA-capable NICs to use. Point it
at a JSON config listing the client IP addresses on the RDMA fabric:

```json
{
    "execution": {
        "parallel_io": false
    },
    "properties": {
        "allow_compat_mode": true,
        "use_pci_p2pdma": true,
        "rdma_peer_type": "dmabuf",
        "rdma_dev_addr_list": ["10.0.1.2", "10.0.2.2"]
    }
}
```

```bash
export CUFILE_ENV_PATH_JSON=/path/to/cufile.json
```

## Supported memory types

- `OBJ_SEG` - S3 object
- `DRAM_SEG` - host memory
- `VRAM_SEG` - GPU memory (data moves directly between the RDMA fabric and
  GPU memory; it never transits host RAM)

## Transfer semantics

- **Object key mapping**: registering an `OBJ_SEG` blob records a mapping
  from its `devId` to an object key - `metaInfo` if provided, otherwise the
  stringified `devId`. `prepXfer` looks this mapping up per descriptor, so an
  `OBJ_SEG` must be registered before it appears as a transfer target.
- **Reads**: the remote descriptor's `addr` is the byte offset into the
  object; `len` is how much to read. The engine issues a ranged GET
  (`Range: bytes=<offset>-<offset+len-1>`) with the RDMA token attached, and
  the endpoint writes the result directly into the registered local buffer.
- **Writes**: RDMA PUT always writes the object in full from byte zero - a
  non-zero write offset is rejected before a request is ever built.
- **Completion**: `postXfer` fires all transfer units for the request
  concurrently and returns `NIXL_IN_PROG` immediately; each unit's AWS SDK
  callback updates the request handle's completion state as it lands, so
  `checkXfer` is a cheap, lock-free poll rather than a blocking wait.
  `releaseReqH` releases each unit's RDMA token via `cuMemObjPutRDMAToken`
  and frees the request handle once `checkXfer` reports the transfer is
  done.
