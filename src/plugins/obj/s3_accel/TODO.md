<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# s3_accel engines

The `s3_accel/` tree provides S3-over-RDMA engines selected through the
`objAccelEngineRegistry`, enabled per backend via `accelerated=true`.

The standard-S3 engine is the preferred choice because it complies with the
agreed, published S3-over-RDMA protocol. It lives in `s3_accel/generic/`, speaks
the `x-amz-rdma-*` headers, and is vendor-neutral, so it needs no per-vendor
code. It is selected via `accelerated=true` with no `type`, or `type=s3`. Its
protocol helpers live in `../s3/rdma_protocol.h` and its transport in
`../s3/rdma.cpp`. See [`../RDMA_PROTOCOL.md`](../RDMA_PROTOCOL.md) and the
[GPU-Direct section of the README](../README.md#gpu-direct-s3-over-rdma).

Vendor engines serve servers that use vendor-specific RDMA headers. For example,
the Dell engine (`s3_accel/dell`, `type=dell`) uses the `x-rdma-info` header and
selects its engine statically at backend creation. It is enabled via
`accelerated=true, type=dell`.
