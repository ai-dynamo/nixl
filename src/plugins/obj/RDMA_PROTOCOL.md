<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# S3-over-RDMA Wire Protocol

This document specifies the GPU-direct S3-over-RDMA protocol the object plugin
speaks, so that **any S3-compatible storage vendor can comply on the server side
without NIXL-specific code**. The control plane stays standard HTTP + SigV4,
while the object payload moves out-of-band over RDMA. A server that implements
the contract below works with this client when the caller opts in via
`accelerated=true` (the standard-S3 engine; no `type`, or `type=s3`).

**Provenance.** The `x-amz-rdma-*` header names and the token layout come
directly from NVIDIA's published [cuObject documentation][cuobj] (the
client-side API and the RDMA-enabled GET/PUT workflow). The wire protocol was
written up as a spec ([aws-c-s3 `RDMA_PROTOCOL_SPEC.md`][spec]) and proposed to
AWS. It is not yet GA'ed by AWS; however AWS refined it inline with what they
could eventually implement, so it is a stable, vendor-neutral base to
standardize on rather than an endpoint-specific convention.

[cuobj]: https://docs.nvidia.com/gpudirect-storage/cuobject/index.html
[spec]: https://github.com/KiranModukuri/aws-c-s3/blob/nvidia_rdma/RDMA_PROTOCOL_SPEC.md

The standard-S3 engine is the **preferred** GPU-direct path: it follows this
proposed protocol and needs no per-vendor code, so it works against any
conformant server. Vendor `type=...` engines exist only for servers that speak a
vendor's own RDMA headers.

> The client treats RDMA as an explicit opt-in and does **not** auto-fall-back to
> HTTP today (see the README's "No automatic fallback (yet)"). A compliant server
> signals an unsupported request with `x-amz-rdma-reply: 501`, but a
> _non_-compliant server may silently ignore the token and accept a body-less PUT
> as a 0-byte object — so auto-fallback becomes safe only once the `501` contract
> below is universal.

The reference server implementation is MinIO AIStor (`internal/rdma`, repo at
`../aistor`).

## Transport

- RDMA **Dynamically Connected (DC)** transport over **InfiniBand** or
  **RoCEv2** (cuObject protocol `CUOBJ_PROTO_RDMA_DC_V1`).
- **GET**: the server issues `RDMA_WRITE`, pushing object bytes directly into the
  client's registered buffer (host DRAM or GPU VRAM).
- **PUT** / **UploadPart**: the server issues `RDMA_READ`, pulling bytes directly
  out of the client's registered buffer.
- The HTTP request/response body is **empty**; payload never traverses it.

## Headers

| Header                                   | Direction        | Meaning                                                                                                                                                                                                                               |
| ---------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x-amz-rdma-token`                       | request          | RDMA negotiation token (see format below). Its presence is what marks a request as RDMA.                                                                                                                                              |
| `x-amz-content-sha256: UNSIGNED-PAYLOAD` | request          | Body is not signed (there is no body).                                                                                                                                                                                                |
| `Content-Length: 0`                      | request (PUT)    | No HTTP body; data is delivered via `RDMA_READ`.                                                                                                                                                                                      |
| `Range: bytes=a-b`                       | request (GET)    | Optional; selects a byte range (server replies `206`).                                                                                                                                                                                |
| `x-amz-checksum-crc64nvme`               | request/response | Optional CRC64NVME checksum (PUT/UploadPart).                                                                                                                                                                                         |
| `x-amz-rdma-reply`                       | response         | Outcome marker. **Required on success.** `200`/`204`/`206` = success; `501` (or absent) = server declined RDMA. Under `accelerated=true` the client treats a decline as a hard error (no automatic HTTP fallback today — see README). |
| `x-amz-rdma-bytes-transferred`           | response (GET)   | Actual bytes moved over RDMA (may be `< requested` for ranged GETs).                                                                                                                                                                  |
| `ETag`, `x-amz-checksum-crc64nvme`       | response (PUT)   | Standard S3 result metadata.                                                                                                                                                                                                          |

### Token format

```text
x-amz-rdma-token: <descriptor>:<start_addr_hex>:<size_hex>
```

- `<descriptor>` — the opaque, fixed-length RDMA descriptor minted by the
  client's RDMA provider (cuObject `cuMemObjGetRDMAToken`); it encodes the
  registered buffer's base address, max size, remote key and NIC routing
  (GID/LID/DCTN). The server splits it off (it is a known fixed length) and
  forwards exactly those bytes to its own RDMA provider (cuObjServer); it must
  not interpret them.
- `<start_addr_hex>` — client buffer start address for this operation, 16-digit
  lowercase hex.
- `<size_hex>` — transfer size in bytes, 16-digit lowercase hex.

Appending the per-operation start address and size to the descriptor mirrors
cuObject's own **IO descriptor** layout (its reference server reads
`<descriptor><rem_buf_start>…<rem_msg_size>…`); it is not an endpoint-specific
addition. Because the HTTP body is empty (`Content-Length: 0`), the token is the
only place these per-request fields can travel. The two trailing fields are
`:`-separated and base-16; the server splits them off the end and treats the
remaining prefix as the descriptor. A trailing `;` (some clients append one)
must be tolerated.

## Request flow (server obligations)

1. **Authenticate** the request with standard SigV4. `x-amz-rdma-token` is part
   of the signed headers; `x-amz-content-sha256` is `UNSIGNED-PAYLOAD`.
2. **Detect** an RDMA request by the presence of `x-amz-rdma-token`; parse it
   into `(descriptor, start_addr, size)`.
3. **Decide** whether RDMA can be honored (fabric reachable, buffer registrable,
   object/permissions valid).
   - **If not** → respond `x-amz-rdma-reply: 501`. You MAY include a normal HTTP
     error body (do not force `Content-Length: 0`). The `501` signal is what lets
     a client safely decide what to do; today this client treats it as an error
     under `accelerated=true` (it does not auto-retry over HTTP — see README), but the
     signal is required so future clients can fall back transparently.
4. **Transfer** using the cuObject server APIs (register a local buffer, then):
   - GET / RANGE_GET → `RDMA_WRITE` the requested bytes into the client buffer.
   - PUT / UploadPart → `RDMA_READ` the bytes from the client buffer and persist
     them as the object/part.
5. **Respond** on success:
   - `x-amz-rdma-reply: 200` (full GET/PUT), `204` (PUT, no content), or `206`
     (ranged GET partial).
   - `x-amz-rdma-bytes-transferred: <n>` for GET.
   - `Content-Length: 0` and **no HTTP body** (the data went over RDMA — without
     this the client blocks waiting for body bytes that never arrive).
   - `ETag` (and `x-amz-checksum-crc64nvme` if requested) for PUT.

## Completion semantics

`x-amz-rdma-reply` is **not** a strict requirement on the PUT success path — it
is an outcome marker the server is free to use:

- **PUT success:** the server completes the `RDMA_READ` and returns a standard
  HTTP `200` + `ETag` (the body is empty; data moved out-of-band). The client
  treats `200` + a non-empty `ETag` as success. AIStor does not set
  `x-amz-rdma-reply` on this path, and a client must not require it.
- **GET success:** the server returns `x-amz-rdma-reply: 200/206` and
  `x-amz-rdma-bytes-transferred` (the authoritative moved-byte count); the HTTP
  body is empty.
- **Decline:** `x-amz-rdma-reply: 501` (or its absence on a non-RDMA server)
  signals the request was not honored over RDMA.

The data path's integrity is covered by RoCEv2 iCRC and, when requested, the
`x-amz-checksum-crc64nvme` header — not by the (empty) HTTP-body content hash.

## Multipart uploads

UploadPart uses the same body-less, token-carrying PUT with the standard
`?uploadId=<id>&partNumber=<n>` query parameters (`1 ≤ n ≤ 10000`). The part's
`ETag`/checksum are returned as usual; CompleteMultipartUpload is an ordinary
HTTP call.

## Compliance checklist

- [ ] Parse `x-amz-rdma-token` as `<descriptor>:<addr16>:<size16>` (tolerate a
      trailing `;`).
- [ ] Authenticate with SigV4 treating the body as `UNSIGNED-PAYLOAD`.
- [ ] GET ⇒ `RDMA_WRITE` into the client buffer; PUT ⇒ `RDMA_READ` from it.
- [ ] Always set `x-amz-rdma-reply` (`200`/`204`/`206`) on success, `501` to decline.
- [ ] Set `x-amz-rdma-bytes-transferred` on GET; force `Content-Length: 0` with
      no body on every success.
- [ ] Return `ETag`/checksum for PUT; honor `Range` for GET (`206`).
- [ ] Fall through to normal HTTP semantics for non-RDMA requests.

A vendor that satisfies this checklist is supported by the NIXL object plugin's
default S3 client automatically — no NIXL-side engine or plugin is required.
