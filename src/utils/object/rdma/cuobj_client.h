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

#include <cuobjclient.h>

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
     * @brief Mint an RDMA token for a range inside a registered buffer.
     *
     * cuObject accepts only a registration's base address plus an offset, so the
     * registration containing [ptr, ptr + size) is looked up and the token is
     * minted as (base, size, ptr - base).
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
    SharedCuObjClient();
    CUObjIOOps ops_{};
    std::unique_ptr<cuObjClient> client_;
    bool connected_ = false;
    std::mutex mutex_;
    std::map<uintptr_t, size_t> registrations_;
};

} // namespace nixl_obj_rdma

#endif // NIXL_SRC_UTILS_OBJECT_RDMA_CUOBJ_CLIENT_H
