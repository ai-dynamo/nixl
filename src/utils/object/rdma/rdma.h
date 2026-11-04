/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_UTILS_OBJECT_RDMA_RDMA_H
#define NIXL_SRC_UTILS_OBJECT_RDMA_RDMA_H

// Generic S3-over-RDMA data path for the object backend.
//
// RDMA is NOT a separate engine or a vendor plugin. It is an optimization of
// the normal S3 GET/PUT path on the standard client, enabled per backend via
// `accelerated=true` (generic S3-over-RDMA): the client issues an out-of-band
// RDMA transfer over the published `x-amz-rdma-*` protocol. Under
// `accelerated=true` an RDMA decline/failure is a hard
// error — there is no silent HTTP fallback today, because a server that ignores
// the token (instead of returning `x-amz-rdma-reply: 501`) would accept a
// body-less PUT as a 0-byte object (see s3/client.cpp). The protocol is an AWS
// S3 convention (not vendor-specific), so the same code works against any
// compliant S3 endpoint that adopts it (including AWS S3 itself).
//
// This is the public umbrella header for the S3-over-RDMA utilities. It composes
// units with a single responsibility each:
//   - rdma_protocol.h         pure wire-protocol helpers (no AWS/cuObject deps)
//   - cuobj_client.h          cuObject buffer registration + token minting
//   - s3_control_plane_http.h the signed, body-less control-plane GET/PUT
//   - rdmaRetry.h             token-lifecycle + one-transient-retry wrappers
// The cuObject-dependent units (everything except rdma_protocol.h) are added to
// the build only when the cuObjClient library is present, so this umbrella is
// meant to be included from code compiled with cuObject support.

#include "rdma_protocol.h"
#include "cuobj_client.h"
#include "s3_control_plane_http.h"
#include "rdmaRetry.h"

#endif // NIXL_SRC_UTILS_OBJECT_RDMA_RDMA_H
