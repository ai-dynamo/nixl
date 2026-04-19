/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_DELL_CUOBJ_TOKEN_MANAGER_H
#define NIXL_OBJ_PLUGIN_S3_DELL_CUOBJ_TOKEN_MANAGER_H

#include <cuobjclient.h>
#include <memory>
#include <string>

/**
 * RAII wrapper around cuObjClient for Pattern B (manual RDMA token management).
 *
 * This class does NOT use cuObjPut/cuObjGet callbacks (Pattern A).  Instead it
 * calls cuMemObjGetRDMAToken() to generate tokens with explicit, caller-managed
 * lifetimes (cuObject spec section 1.12.4).
 *
 * Memory registration is 1:1: each registerMemory() call maps to exactly one
 * cuMemObjGetDescriptor(), and each deregisterMemory() maps to exactly one
 * cuMemObjPutDescriptor().  Registration happens once at init time (not on the
 * hot path), so per-page registration cost is acceptable.
 *
 * Token generation calls cuMemObjGetRDMAToken() with buffer_offset=0 because
 * each page is registered individually at its exact address and size.
 *
 * Thread safety:
 *   - generatePutToken / generateGetToken call cuMemObjGetRDMAToken which is
 *     thread-safe per the cuObject spec (section 1.14.1: "I/O operations are
 *     thread safe for different buffers").  Each call gets its own token
 *     allocation.
 */
class CuObjTokenManager {
public:
    /**
     * Construct the token manager.
     * Creates a cuObjClient with empty CUObjIOOps (Pattern B — no callbacks).
     *
     * @param proto  RDMA protocol version.  Defaults to CUOBJ_PROTO_RDMA_DC_V1.
     */
    explicit CuObjTokenManager(cuObjProto_t proto = CUOBJ_PROTO_RDMA_DC_V1);

    /** Destructor.  Destroys the cuObjClient. */
    ~CuObjTokenManager();

    /* Non-copyable, non-movable — owns the cuObjClient instance. */
    CuObjTokenManager(const CuObjTokenManager &) = delete;
    CuObjTokenManager &operator=(const CuObjTokenManager &) = delete;

    /**
     * @return true if the cuObjClient is connected and ready for operations.
     */
    bool isConnected() const;

    /**
     * Register a memory region for RDMA.
     * Wraps cuMemObjGetDescriptor(ptr, size).
     *
     * @param ptr   Start address of the memory to register.
     * @param size  Size in bytes.  Must be < 4 GiB (CUOBJ_MAX_MEMORY_REG_SIZE).
     * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure.
     */
    cuObjErr_t registerMemory(void *ptr, size_t size);

    /**
     * Deregister a previously registered memory region.
     * Wraps cuMemObjPutDescriptor(ptr).
     *
     * @param ptr  Start address passed to registerMemory().
     * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure.
     */
    cuObjErr_t deregisterMemory(void *ptr);

    /**
     * Generate an RDMA token for a PUT (upload) operation.
     *
     * Calls cuMemObjGetRDMAToken(ptr, size, 0, CUOBJ_PUT).  The returned
     * string is a copy — the library allocation is freed immediately via
     * cuMemObjPutRDMAToken().
     *
     * @param data_ptr  Address of the data to transfer (must be a registered region).
     * @param size      Number of bytes to transfer.
     * @return RDMA token string for use as an HTTP header value.
     * @throws std::runtime_error on failure.
     */
    std::string generatePutToken(void *data_ptr, size_t size);

    /**
     * Generate an RDMA token for a GET (download) operation.
     * Same semantics as generatePutToken but for CUOBJ_GET.
     */
    std::string generateGetToken(void *data_ptr, size_t size);

private:
    /** Common token generation logic. */
    std::string generateToken(void *data_ptr, size_t size, cuObjOpType_t op);

    std::unique_ptr<cuObjClient> client_;
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_CUOBJ_TOKEN_MANAGER_H
