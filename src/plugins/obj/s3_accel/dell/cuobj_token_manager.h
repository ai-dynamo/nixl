/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_DELL_CUOBJ_TOKEN_MANAGER_H
#define NIXL_OBJ_PLUGIN_S3_DELL_CUOBJ_TOKEN_MANAGER_H

#include <cuobjclient.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

/**
 * RAII wrapper around cuObjClient for Pattern B (manual RDMA token management).
 *
 * This class does NOT use cuObjPut/cuObjGet callbacks (Pattern A).  Instead it
 * calls cuMemObjGetRDMAToken() to generate tokens with explicit, caller-managed
 * lifetimes (cuObject spec section 1.12.4).
 *
 * Memory registration is pool-aware: when many page-sized chunks from a single
 * contiguous buffer are registered, only the first chunk triggers the heavyweight
 * cuMemObjGetDescriptor() NIC registration.  Subsequent chunks that fall within
 * an already-registered region are instant no-ops (refcount increment).
 *
 * Token generation computes pool-relative buffer_offset automatically, so the
 * caller can pass the per-page data pointer directly.
 *
 * Thread safety:
 *   - registerMemory / deregisterMemory are serialised by regions_mutex_.
 *   - generatePutToken / generateGetToken are serialised by regions_mutex_
 *     for the region lookup, then call cuMemObjGetRDMAToken which is thread-safe
 *     per the cuObject spec (section 1.14.1: "I/O operations are thread safe
 *     for different buffers").  Each call gets its own token allocation.
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

    /**
     * Destructor.  Deregisters any remaining regions and destroys the client.
     */
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
     *
     * If [ptr, ptr+size) is already within a previously registered region the
     * call is a no-op (refcount increment) and CU_OBJ_SUCCESS is returned.
     * Otherwise cuMemObjGetDescriptor(ptr, reg_size) is called, where reg_size
     * may be expanded to @p pool_hint if the hint is non-zero and within the
     * cuObject 4 GiB limit.
     *
     * @param ptr        Start address of the memory to register.
     * @param size       Size of this particular chunk.
     * @param pool_hint  Optional hint for the total pool size.  When non-zero
     *                   the first registration from this region is expanded to
     *                   cover [ptr, ptr+pool_hint) so that subsequent chunks
     *                   within the same pool are de-duplicated.
     * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure.
     */
    cuObjErr_t registerMemory(void *ptr, size_t size, size_t pool_hint = 0);

    /**
     * Deregister a previously registered memory chunk.
     *
     * Decrements the refcount for the containing region.  The actual
     * cuMemObjPutDescriptor() call happens only when the refcount reaches zero.
     *
     * @param ptr   Start address passed to registerMemory().
     * @param size  Size passed to registerMemory().
     * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure.
     */
    cuObjErr_t deregisterMemory(void *ptr, size_t size);

    /**
     * Generate an RDMA token for a PUT (upload) operation.
     *
     * Finds the registered region containing @p data_ptr, computes the
     * pool-relative offset, and calls cuMemObjGetRDMAToken().  The returned
     * string is a copy — the library allocation is freed immediately via
     * cuMemObjPutRDMAToken().
     *
     * @param data_ptr  Address of the data to transfer (within a registered region).
     * @param size      Number of bytes to transfer.
     * @param offset    Additional offset for the S3 object (passed through).
     * @return RDMA token string for use as an HTTP header value.
     * @throws std::runtime_error on failure.
     */
    std::string generatePutToken(void *data_ptr, size_t size, size_t offset);

    /**
     * Generate an RDMA token for a GET (download) operation.
     * Same semantics as generatePutToken but for CUOBJ_GET.
     */
    std::string generateGetToken(void *data_ptr, size_t size, size_t offset);

private:
    /** Common token generation logic. */
    std::string generateToken(void *data_ptr, size_t size,
                              size_t offset, cuObjOpType_t op);

    /** A contiguous memory region registered with cuMemObjGetDescriptor. */
    struct RegisteredRegion {
        uintptr_t base;      ///< Start address passed to cuMemObjGetDescriptor.
        size_t    size;       ///< Size passed to cuMemObjGetDescriptor.
        size_t    refcount;   ///< Number of registerMemory() calls within this region.
    };

    /**
     * Find the region that fully contains [addr, addr+size).
     * @return Pointer to the region, or nullptr if not found.
     */
    RegisteredRegion *findContainingRegion(uintptr_t addr, size_t size);

    std::unique_ptr<cuObjClient> client_;
    std::vector<RegisteredRegion> regions_;
    mutable std::mutex regions_mutex_;  ///< Protects regions_ for thread safety.
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_CUOBJ_TOKEN_MANAGER_H
