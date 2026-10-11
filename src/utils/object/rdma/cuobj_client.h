/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_UTILS_OBJECT_RDMA_CUOBJ_CLIENT_H
#define NIXL_SRC_UTILS_OBJECT_RDMA_CUOBJ_CLIENT_H

// Process-wide cuObject client for the S3-over-RDMA data path: buffer
// registration (cuMemObjGetDescriptor) and RDMA-token minting. This is the only
// component that links libcuobjclient, so it is compiled only when that library
// is present.

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include <cuobjclient.h>

namespace nixl_obj_rdma {

// Largest buffer cuObject registers in one cuMemObjGetDescriptor call.
inline constexpr size_t max_region_size = CUOBJ_MAX_MEMORY_REG_SIZE;
// A large registration is covered by overlapping max_region_size windows, one
// starting every window_stride bytes, so any range of at most
// max_windowed_xfer_size bytes lies inside a single window.
inline constexpr size_t window_stride = size_t(2) << 30;
inline constexpr size_t max_windowed_xfer_size = max_region_size - window_stride;

/**
 * @brief The cuObjClient calls SharedCuObjClient makes, so that tests can
 * substitute them. The methods mirror cuObjClient's.
 */
class cuObjApi {
public:
    virtual ~cuObjApi() = default;

    virtual cuObjErr_t
    cuMemObjGetDescriptor(void *ptr, size_t size) = 0;

    virtual cuObjErr_t
    cuMemObjPutDescriptor(void *ptr) = 0;

    virtual cuObjErr_t
    cuMemObjGetRDMAToken(void *ptr,
                         size_t size,
                         size_t buffer_offset,
                         cuObjOpType_t operation,
                         char **desc_str_out) = 0;

    virtual cuObjErr_t
    cuMemObjPutRDMAToken(char *desc_str) = 0;
};

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
     * @brief Construct a client over the given cuObject calls, for tests.
     * Production code uses instance().
     * @param api The cuObject calls to make.
     */
    explicit SharedCuObjClient(std::unique_ptr<cuObjApi> api);

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
     *
     * Registering an already registered buffer again with the same size adds a
     * reference instead of pinning it twice. A buffer larger than
     * max_region_size is rejected unless allow_large is set, in which case it is
     * registered as overlapping windows (see window_stride).
     * @param ptr Start of the buffer to register.
     * @param size Buffer length in bytes.
     * @param allow_large Register a buffer larger than max_region_size as windows.
     * @return true on success, false if registration failed.
     */
    [[nodiscard]] bool
    registerBuffer(void *ptr, size_t size, bool allow_large = false);

    /**
     * @brief Release a buffer registration acquired via registerBuffer().
     *
     * The buffer is unpinned once its last reference is released.
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
     * @brief Mint an RDMA token for a range inside a registered buffer.
     *
     * cuObject accepts only a registration's base address plus an offset, so the
     * registration containing [ptr, ptr + size) is looked up and the token is
     * minted as (base, size, ptr - base). That registration holds a reference
     * until the token is released, so it stays registered while the token is in
     * use even if its owner deregisters it.
     * @param ptr Start of the range; anywhere inside a registered buffer.
     * @param size Length in bytes covered by the token.
     * @param op Operation the token authorizes (CUOBJ_GET / CUOBJ_PUT).
     * @return An opaque token string (release via putToken()), or nullptr if no
     *         registration contains the range or minting failed.
     */
    [[nodiscard]] char *
    getToken(void *ptr, size_t size, cuObjOpType_t op);

    /**
     * @brief Release a token acquired via getToken().
     * @param token Token to release; a nullptr is ignored.
     */
    void
    putToken(char *token);

private:
    struct Registration {
        size_t size;
        size_t refs;
        std::vector<uintptr_t> regions;
    };

    struct Region {
        size_t size;
        uintptr_t owner;
    };

    SharedCuObjClient();
    void
    release(std::map<uintptr_t, Registration>::iterator it);

    std::unique_ptr<cuObjApi> client_;
    bool connected_ = false;
    std::mutex mutex_;
    std::map<uintptr_t, Registration> registrations_;
    // cuObject registrations backing registrations_, by base address.
    std::map<uintptr_t, Region> regions_;
    // Outstanding tokens and the registration each was minted from.
    std::map<char *, uintptr_t> tokens_;
};

} // namespace nixl_obj_rdma

#endif // NIXL_SRC_UTILS_OBJECT_RDMA_CUOBJ_CLIENT_H
