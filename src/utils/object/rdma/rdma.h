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
 * @brief Process-wide cuObjClient singleton.
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
    /**
     * @brief Get the process-wide instance.
     * @return The singleton, or nullptr if the RDMA fabric is unavailable.
     */
    [[nodiscard]] static SharedCuObjClient *
    instance();

    /**
     * @brief Whether the RDMA fabric is connected.
     * @return true if the underlying cuObjClient connected successfully.
     */
    [[nodiscard]] bool
    isConnected() const {
        return connected_;
    }

    /**
     * @brief Pin a buffer for RDMA. Required before minting a token for it.
     * @param ptr Start of the buffer to register.
     * @param size Buffer length in bytes.
     * @return true on success, false if registration failed.
     */
    [[nodiscard]] bool
    registerBuffer(void *ptr, size_t size);

    /**
     * @brief Release a buffer registration acquired via registerBuffer().
     * @param ptr Buffer previously passed to registerBuffer().
     */
    void
    deregisterBuffer(void *ptr);

    /**
     * @brief Test whether a pointer is CUDA device (VRAM) memory.
     * @param ptr Pointer to classify.
     * @return true for device memory (no HTTP fallback possible), false otherwise.
     */
    [[nodiscard]] bool
    isDeviceMemory(const void *ptr) const;

    /**
     * @brief Mint an RDMA token for a registered buffer.
     * @param ptr Registered buffer.
     * @param size Length in bytes covered by the token.
     * @param offset Byte offset into the buffer.
     * @param op Operation the token authorizes (CUOBJ_GET / CUOBJ_PUT).
     * @return An opaque token string (release via putToken()), or nullptr on failure.
     */
    [[nodiscard]] char *
    getToken(void *ptr, size_t size, size_t offset, cuObjOpType_t op);

    /**
     * @brief Release a token acquired via getToken().
     * @param token Token to release; a nullptr is ignored.
     */
    void
    putToken(char *token);

private:
    SharedCuObjClient();
    CUObjIOOps ops_{};
    std::unique_ptr<cuObjClient> client_;
    bool connected_ = false;
    std::mutex mutex_;
};

/**
 * @brief Per-call context for an RDMA PUT/GET control-plane request.
 *
 * Region and credentials live in the control plane's signer, not here.
 */
struct S3RdmaClientCtx {
    std::string bucket; ///< Target bucket.
    std::string object; ///< Object key.
    std::string uploadId; ///< Multipart upload id; empty for single-shot.
    uint32_t partNumber = 0; ///< Part number 1..10000 when uploadId is set.
    std::string checksumCrc64nvme; ///< Optional CRC64NVME checksum, in/out.
    std::string etag; ///< ETag returned by the server; populated on success.
};

/**
 * @brief S3 RDMA control plane.
 *
 * Owns the AWS SDK primitives (SigV4 signer + HTTP client + resolved endpoint)
 * used to issue the body-less, RDMA-token-carrying GET/PUT that negotiates the
 * out-of-band transfer. This is the only component that touches the AWS SDK's
 * low-level HTTP layer; it is deliberately narrow so the protocol logic around
 * it stays SDK-agnostic and testable.
 */
class S3RdmaControlPlane {
public:
    /**
     * @brief Build the control plane from backend params.
     *
     * Resolves the endpoint, region, and credentials. On failure, valid()
     * returns false and the instance is unusable.
     * @param custom_params Backend key-value params; may be nullptr.
     */
    explicit S3RdmaControlPlane(const nixl_b_params_t *custom_params);
    ~S3RdmaControlPlane();

    /**
     * @brief Whether the control plane initialized successfully.
     * @return true iff the HTTP client and (access + secret) credentials resolved.
     */
    [[nodiscard]] bool
    valid() const {
        return valid_;
    }

    /**
     * @brief Issue the signed control-plane PUT carrying the RDMA token.
     * @param ctx Request context (bucket/object, multipart, checksum, etag out).
     * @param token RDMA token minted for the buffer.
     * @param buf_addr Start address of the source buffer.
     * @param size Number of bytes to transfer.
     * @return Bytes transferred (>0) on RDMA success, rdma_not_supported if the
     *         server declined, or rdma_error on transport failure.
     */
    [[nodiscard]] ssize_t
    rdmaPut(S3RdmaClientCtx &ctx, const char *token, uint64_t buf_addr, uint64_t size);

    /**
     * @brief Issue the signed control-plane GET carrying the RDMA token.
     * @param ctx Request context (bucket/object, checksum, etag out).
     * @param token RDMA token minted for the buffer.
     * @param buf_addr Start address of the destination buffer.
     * @param size Number of bytes to fetch.
     * @param offset Byte offset into the object; a byte-range request is made
     *        (server replies 206) when it is non-zero.
     * @return Bytes transferred (>0), rdma_not_supported if declined, or
     *         rdma_error on failure.
     */
    [[nodiscard]] ssize_t
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
 * @brief Mint a token, run rdmaPut, release the token, with one transient retry.
 *
 * The retry covers token-mint and control-plane hiccups. The buffer must
 * already be registered via SharedCuObjClient::registerBuffer().
 * @param rdma Shared cuObject client used to mint/release the token.
 * @param cp Control plane that issues the signed request.
 * @param ctx Request context (bucket/object, multipart, checksum, etag out).
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
 * @param ctx Request context (bucket/object, checksum, etag out).
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

} // namespace nixl_obj_rdma

#endif // HAVE_CUOBJ_CLIENT

#endif // NIXL_SRC_UTILS_OBJECT_RDMA_RDMA_H
