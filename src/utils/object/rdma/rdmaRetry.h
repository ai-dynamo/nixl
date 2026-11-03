/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_UTILS_OBJECT_RDMA_RDMARETRY_H
#define NIXL_SRC_UTILS_OBJECT_RDMA_RDMARETRY_H

// Token-lifecycle + one-transient-retry wrappers that drive a full RDMA PUT/GET:
// mint a cuObject token (cuobj_client.h), issue the signed control-plane request
// (s3_control_plane_http.h), release the token, and retry once on a transient
// failure. Compiled only when the cuObject library is present.

#include <cstdint>

#include <sys/types.h> // ssize_t

#include "cuobj_client.h"
#include "s3_control_plane_http.h"

namespace nixl_obj_rdma {

/**
 * @brief Mint a token, run rdmaPut, release the token, with one transient retry.
 *
 * The retry covers token-mint and control-plane hiccups. The buffer must
 * already be registered via SharedCuObjClient::registerBuffer().
 * @param rdma Shared cuObject client used to mint/release the token.
 * @param cp Control plane that issues the signed request.
 * @param ctx Request context (bucket/object, multipart, etag out).
 * @param buf Source buffer.
 * @param size Number of bytes to transfer.
 * @return >0 bytes transferred (success), rdma_not_supported (server declined),
 *         or rdma_error (failure). The caller treats anything < 0 as an error —
 *         there is no HTTP fallback under accelerated=true.
 */
[[nodiscard]] ssize_t
rdmaPutWithRetry(SharedCuObjClient &rdma,
                 S3RdmaControlPlane &cp,
                 S3RdmaClientCtx &ctx,
                 void *buf,
                 uint64_t size);

/**
 * @brief Mint a token, run rdmaGet, release the token, with one transient retry.
 *
 * The transfer is byte-ranged via @p offset. The buffer must already be
 * registered via SharedCuObjClient::registerBuffer().
 * @param rdma Shared cuObject client used to mint/release the token.
 * @param cp Control plane that issues the signed request.
 * @param ctx Request context (bucket/object, etag out).
 * @param buf Destination buffer.
 * @param size Number of bytes to fetch.
 * @param offset Byte offset into the object.
 * @return >0 bytes transferred (success), rdma_not_supported (server declined),
 *         or rdma_error (failure). The caller treats anything < 0 as an error —
 *         there is no HTTP fallback under accelerated=true.
 */
[[nodiscard]] ssize_t
rdmaGetWithRetry(SharedCuObjClient &rdma,
                 S3RdmaControlPlane &cp,
                 S3RdmaClientCtx &ctx,
                 void *buf,
                 uint64_t size,
                 uint64_t offset);

// cuObject mints a token for at most this many bytes, even from a larger
// registration, so a single S3 request moves at most this much.
inline constexpr uint64_t rdma_max_request_size = CUOBJ_MAX_MEMORY_REG_SIZE;

/**
 * @brief Write buf to ctx.bucket/ctx.object over RDMA. An object larger than
 *        rdma_max_request_size is uploaded as a multipart upload, one RDMA part
 *        per rdma_max_request_size bytes.
 * @return size on success, or a negative rdma_* code.
 */
[[nodiscard]] ssize_t
rdmaPutObject(SharedCuObjClient &rdma,
              S3RdmaControlPlane &cp,
              S3RdmaClientCtx &ctx,
              void *buf,
              uint64_t size);

/**
 * @brief Read size bytes of ctx.bucket/ctx.object from offset into buf over
 *        RDMA, as ranged GETs of at most rdma_max_request_size bytes each.
 *        Every range after the first must match the first range's ETag, so a
 *        read never mixes two versions of the object.
 * @return Bytes read, or a negative rdma_* code.
 */
[[nodiscard]] ssize_t
rdmaGetObject(SharedCuObjClient &rdma,
              S3RdmaControlPlane &cp,
              S3RdmaClientCtx &ctx,
              void *buf,
              uint64_t size,
              uint64_t offset);

} // namespace nixl_obj_rdma

#endif // NIXL_SRC_UTILS_OBJECT_RDMA_RDMARETRY_H
