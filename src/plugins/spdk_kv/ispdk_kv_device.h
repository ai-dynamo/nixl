/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file ispdk_kv_device.h
 * @brief iSpdkKvDevice: an ABSTRACT INTERFACE over the kv_host_shim NVMe
 *        Key-Value protocol, used by the shared nixlSpdkKvEngine data plane.
 *
 * WHAT THIS IS
 * ------------
 * iSpdkKvDevice is a small, purely abstract C++ interface that captures the
 * generic NVMe Key-Value protocol exposed by the in-process SPDK "kv_host_shim"
 * C API (Store / Retrieve / Exist over an NVMe-KV controller). It is the seam
 * between the backend-agnostic data plane and the concrete transport: the shared
 * engine talks ONLY to this interface and knows nothing about SPDK allocation
 * APIs, NVMe status codes, or kv_host_shim itself. It only knows that values
 * use abstract transfer-capable buffers supplied by the device. All concrete
 * transport details live BEHIND the interface.
 *
 * DESIGN / CLASS HIERARCHY
 * ------------------------
 *   nixlBackendEngine                       (NIXL core)
 *     ├── nixlSpdkKvEngine (abstract; shared NVMe-KV data plane)
 *     │     ├── nixlRadosNkvEngine (device: RADOS target)
 *     │     └── nixlCsalNkvEngine  (device: CSAL target)
 *     └── nixlRedisKVEngine (standalone; implemented directly)
 *
 *   iSpdkKvDevice (interface; used by nixlSpdkKvEngine)
 *     ├── nixlKvHostShimDevice (real: wraps the kv_host_shim C API)
 *     └── nixlFakeKvDevice     (test double, no SPDK/hardware needed)
 *
 * WHY AN INTERFACE (rationale)
 * ----------------------------
 * nixlSpdkKvEngine implements getSupportedMems, registerMem, deregisterMem,
 * queryMem, prepXfer, postXfer, checkXfer, and releaseReqH ONCE, against
 * iSpdkKvDevice.
 *
 * RADOS_NKV and CSAL_NKV share that base class (nixlSpdkKvEngine) by INHERITANCE,
 * not composition. Both talk to a device through the same generic kv_host_shim
 * NVMe-KV protocol, and that protocol is already backend-agnostic. Since the two
 * concrete backends need the exact same data-plane algorithm
 * (validate -> derive key -> DMA-stage -> store/retrieve/exist), a plain abstract
 * base class is the simplest way to share it: nixlSpdkKvEngine implements that
 * algorithm once, and nixlRadosNkvEngine / nixlCsalNkvEngine only provide a
 * device (preferably by constructor injection) and override deriveKey(). The
 * legacy deferred path may instead override openDevice() and call initDevice().
 *
 * Splitting the DEVICE out as its own interface additionally lets the whole
 * engine data plane be unit-tested with nixlFakeKvDevice (an in-memory double)
 * with no SPDK or hardware, while production uses nixlKvHostShimDevice.
 *
 * OWNERSHIP OF THE CONCRETE CODE
 * ------------------------------
 * This header defines ONLY the abstract contract; it carries no implementation.
 * The concrete devices (nixlKvHostShimDevice, nixlFakeKvDevice) and the shared
 * engine body are provided by the backend plugin PRs, which inherit these types.
 *
 * SEMANTIC RETURNS
 * ----------------
 * The interface is deliberately semantic: operations return SpdkKvStatus, not a
 * raw device status code, so callers never branch on NVMe/SPDK-specific values.
 * Mapping any device-native error space (e.g. NVMe status codes) onto SpdkKvStatus
 * is the concrete implementation's job (see nixlKvHostShimDevice).
 *
 * CONCURRENCY
 * -----------
 * iSpdkKvDevice does not require implementations to be thread-safe. The shared
 * nixlSpdkKvEngine serializes each complete allocation/staging/device-operation/
 * free transaction with its device mutex, which makes a single-qpair adapter a
 * valid implementation.
 */

#ifndef NIXL_SRC_PLUGINS_SPDK_KV_ISPDK_KV_DEVICE_H
#define NIXL_SRC_PLUGINS_SPDK_KV_ISPDK_KV_DEVICE_H

#include <cstddef>
#include <cstdint>

/**
 * @brief Semantic outcome of a KV device operation.
 *
 * Concrete devices map their native error space onto these values, so callers
 * have a single, transport-independent contract to reason about.
 */
enum class SpdkKvStatus {
    /** Operation succeeded. For exist(): the key is present. */
    OK,
    /** The key does not exist (exist() miss, or retrieve() of an absent key). */
    NOT_FOUND,
    /**
     * retrieve() only: the provided buffer is smaller than the stored value. The
     * value_len_out parameter carries the device's TRUE value length so the
     * caller can resize its buffer and retry. No usable data is returned.
     */
    BUFFER_TOO_SMALL,
    /** A device- or transport-level error (never used to signal a plain miss). */
    ERROR,
};

/**
 * @class iSpdkKvDevice
 * @brief Synchronous Key-Value device contract for the SPDK-based backends.
 *
 * Implementations (provided by the backend PRs) own the underlying connection
 * and release it in their destructor. Value buffers passed to store()/retrieve()
 * must be transfer-capable buffers obtained from dmaAlloc().
 */
class iSpdkKvDevice {
public:
    virtual ~iSpdkKvDevice() = default;

    /** Maximum key length the device key space can hold (bytes), or 0 when the
     *  device advertises no limit. */
    virtual uint32_t
    maxKeyLen() const = 0;

    /** Maximum value length the device can store (bytes), or 0 when the device
     *  advertises no additional limit. An adapter whose native API uses a
     *  narrower length type must reject an unrepresentable value before
     *  narrowing it. */
    virtual size_t
    maxValueLen() const = 0;

    /** Allocate a transfer-capable buffer of @p len bytes (zeroed). NULL on
     *  failure. store()/retrieve() value buffers must come from here. */
    virtual void *
    dmaAlloc(size_t len) = 0;

    /** Free a buffer returned by dmaAlloc(). Safe with NULL. */
    virtual void
    dmaFree(void *buf) = 0;

    /**
     * Store @p value (@p value_len bytes) under @p key. @p value must be a
     * buffer from dmaAlloc().
     * Implementations must range-check @p value_len before converting it to a
     * narrower native length type.
     * @return SpdkKvStatus::OK on success, SpdkKvStatus::ERROR otherwise.
     */
    virtual SpdkKvStatus
    store(const void *key, uint8_t key_len, const void *value, size_t value_len) = 0;

    /**
     * Retrieve the value for @p key into @p value (@p buf_len bytes). @p value
     * must be a buffer from dmaAlloc().
     * @param value_len_out When non-NULL, receives the device's TRUE stored value
     *        length on both OK and BUFFER_TOO_SMALL, so a short buffer can be
     *        resized and retried.
     * Implementations must range-check @p buf_len before converting it to a
     * narrower native length type.
     * @return OK (whole value fit; value_len_out <= buf_len), BUFFER_TOO_SMALL
     *         (value_len_out > buf_len; contents unusable), NOT_FOUND, or ERROR.
     */
    virtual SpdkKvStatus
    retrieve(const void *key,
             uint8_t key_len,
             void *value,
             size_t buf_len,
             size_t *value_len_out) = 0;

    /**
     * Query whether @p key is present, transferring no value data.
     * @return OK if present, NOT_FOUND if absent, ERROR on a device/transport
     *         failure (a real failure is never reported as NOT_FOUND).
     */
    virtual SpdkKvStatus
    exist(const void *key, uint8_t key_len) = 0;
};

#endif // NIXL_SRC_PLUGINS_SPDK_KV_ISPDK_KV_DEVICE_H
