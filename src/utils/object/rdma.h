/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_UTILS_OBJECT_RDMA_H
#define NIXL_SRC_UTILS_OBJECT_RDMA_H

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
// S3 convention (not vendor-specific), so the same code works against MinIO
// AIStor today and against any future endpoint (including AWS S3) that adopts
// it.
//
// This entire translation unit is compiled only when the cuObjClient library is
// present (HAVE_CUOBJ_CLIENT). The pure wire-protocol helpers live in
// rdma_protocol.h and have no such dependency.

#include "rdma_protocol.h"

#ifdef HAVE_CUOBJ_CLIENT

#include <memory>
#include <mutex>
#include <string>

#include <cuobjclient.h>

#include "nixl_types.h"

namespace nixl_obj_rdma {

/**
 * Process-wide cuObjClient singleton.
 *
 * libcuobjclient is expensive to construct and its callbacks may fire on
 * threads other than the caller's; constructing one per backend (or per call)
 * was observed to corrupt allocator state under concurrency in the reference
 * SDKs. A single instance per process is the supported pattern. Buffer
 * registration (cuMemObjGetDescriptor) and token minting are serialized through
 * an internal mutex.
 */
class SharedCuObjClient {
public:
    /// Returns the process-wide instance, or nullptr if the fabric is unavailable.
    static SharedCuObjClient *
    instance();

    bool
    isConnected() const {
        return connected_;
    }

    /// Pin a buffer for RDMA. Required before minting a token for it.
    bool
    registerBuffer(void *ptr, size_t size);

    /// Release a buffer registration acquired via registerBuffer().
    void
    deregisterBuffer(void *ptr);

    /// True if the pointer is CUDA device (VRAM) memory (no HTTP fallback possible).
    bool
    isDeviceMemory(const void *ptr) const;

    /// Mint an RDMA token for a registered buffer (caller releases via putToken()).
    char *
    getToken(void *ptr, size_t size, size_t offset, cuObjOpType_t op);
    void
    putToken(char *token);

private:
    SharedCuObjClient();
    CUObjIOOps ops_{};
    std::unique_ptr<cuObjClient> client_;
    bool connected_ = false;
    std::mutex mutex_;
};

/// Per-call context for an RDMA PUT/GET control-plane request.
/// (region/credentials live in the control plane's signer, not here.)
struct S3RdmaClientCtx {
    std::string bucket;
    std::string object;
    std::string uploadId; // empty for single-shot (non-multipart)
    uint32_t partNumber = 0; // 1..=10000 when uploadId is set
    std::string checksumCrc64nvme; // optional, in/out
    std::string etag; // populated on success
};

/**
 * S3 RDMA control plane.
 *
 * Owns the AWS SDK primitives (SigV4 signer + HTTP client + resolved endpoint)
 * used to issue the body-less, RDMA-token-carrying GET/PUT that negotiates the
 * out-of-band transfer. This is the only component that touches the AWS SDK's
 * low-level HTTP layer; it is deliberately narrow so the protocol logic around
 * it stays SDK-agnostic and testable.
 */
class S3RdmaControlPlane {
public:
    /// Build the control plane from backend params (endpoint, region, credentials).
    /// If initialization fails, valid() returns false and the instance is unusable.
    explicit S3RdmaControlPlane(nixl_b_params_t *custom_params);
    ~S3RdmaControlPlane();

    /// True iff the HTTP client and (access + secret) credentials were resolved.
    bool
    valid() const {
        return valid_;
    }

    /**
     * Issue the signed control-plane PUT carrying the RDMA token.
     * @return bytes transferred (>0) on RDMA success, rdma_not_supported if the
     *         server declined, or rdma_error on transport failure.
     */
    ssize_t
    rdmaPut(S3RdmaClientCtx &ctx, const char *token, uint64_t buf_addr, uint64_t size);

    /**
     * Issue the signed control-plane GET carrying the RDMA token. When
     * @p offset is non-zero a byte-range request is made (server replies 206).
     * @return bytes transferred (>0), rdma_not_supported if declined, or
     *         rdma_error on failure.
     */
    ssize_t
    rdmaGet(S3RdmaClientCtx &ctx,
            const char *token,
            uint64_t buf_addr,
            uint64_t size,
            uint64_t offset);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool valid_ = false;
};

/**
 * Mint a token, run rdmaPut, release the token, with one transient retry
 * (covering token-mint and control-plane hiccups). The buffer must already be
 * registered via SharedCuObjClient::registerBuffer().
 *
 * @return >0 bytes transferred (success), rdma_not_supported (server declined),
 *         or rdma_error (failure). The caller treats anything < 0 as an error —
 *         there is no HTTP fallback under accelerated=true.
 */
ssize_t
rdmaPutWithRetry(SharedCuObjClient &rdma,
                 S3RdmaControlPlane &cp,
                 S3RdmaClientCtx &ctx,
                 void *buf,
                 size_t size);

/**
 * Mint a token, run rdmaGet (optionally byte-ranged via @p offset), release the
 * token, with one transient retry. The buffer must already be registered via
 * SharedCuObjClient::registerBuffer().
 *
 * @return >0 bytes transferred (success), rdma_not_supported (server declined),
 *         or rdma_error (failure). The caller treats anything < 0 as an error —
 *         there is no HTTP fallback under accelerated=true.
 */
ssize_t
rdmaGetWithRetry(SharedCuObjClient &rdma,
                 S3RdmaControlPlane &cp,
                 S3RdmaClientCtx &ctx,
                 void *buf,
                 size_t size,
                 size_t offset);

} // namespace nixl_obj_rdma

#endif // HAVE_CUOBJ_CLIENT

#endif // NIXL_SRC_UTILS_OBJECT_RDMA_H
