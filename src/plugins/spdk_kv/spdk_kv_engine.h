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
 * @file spdk_kv_engine.h
 * @brief nixlSpdkKvEngine: the abstract base INTERFACE for SPDK-based NVMe
 *        Key-Value NIXL backends (RADOS_NKV, CSAL_NKV, ...).
 *
 * This header defines ONLY the abstract contract; it contains NO data-plane
 * implementation. The shared data plane (the bodies of the South-Bound API
 * declared below) and the concrete subclasses are provided by the backend plugin
 * PRs, which inherit this class:
 *
 *   nixlBackendEngine                       (NIXL core)
 *     +-- nixlSpdkKvEngine                  (this header: abstract contract)
 *           +-- nixlRadosNkvEngine          (backend PR: overrides the hooks)
 *           +-- nixlCsalNkvEngine           (backend PR: overrides the hooks)
 *
 * WHY A SHARED ABSTRACT BASE (rationale)
 * --------------------------------------
 * nixlSpdkKvEngine implements getSupportedMems, registerMem, deregisterMem,
 * queryMem, prepXfer, postXfer, checkXfer, and releaseReqH ONCE, against
 * iSpdkKvDevice.
 *
 * RADOS_NKV and CSAL_NKV share this base class by INHERITANCE, not composition.
 * Both talk to a device through the same generic kv_host_shim NVMe-KV protocol,
 * and that protocol is already backend-agnostic. Since the two concrete backends
 * need the exact same data-plane algorithm
 * (validate -> derive key -> DMA-stage -> store/retrieve/exist), a plain abstract
 * base class is the simplest way to share it: nixlSpdkKvEngine implements that
 * algorithm once. Concrete engines inject an already-open device and override
 * deriveKey(). The existing deferred path is retained for compatibility: a
 * concrete engine may override openDevice() and call initDevice().
 *
 * (See ispdk_kv_device.h for the full design write-up and class hierarchy.)
 *
 * HEADER vs. IMPLEMENTATION SPLIT
 * ------------------------------
 * Because a C++ class must be declared in one place, this header declares the
 * full shared South-Bound API surface (as overrides, WITHOUT bodies) so the
 * backend PR can provide a single spdk_kv_engine.cpp that implements the data
 * plane once against iSpdkKvDevice. Concrete backends then provide a device and
 * override the key-derivation policy.
 *
 * DEVICE SERIALIZATION
 * --------------------
 * A single engine serializes each complete DMA staging and device operation
 * through deviceMutex_. iSpdkKvDevice implementations therefore need not make a
 * single SPDK qpair concurrently callable.
 */

#ifndef NIXL_SRC_PLUGINS_SPDK_KV_SPDK_KV_ENGINE_H
#define NIXL_SRC_PLUGINS_SPDK_KV_SPDK_KV_ENGINE_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend/backend_engine.h"
#include "ispdk_kv_device.h"

/**
 * @class nixlSpdkKvEngine
 * @brief Abstract NIXL backend for the generic NVMe Key-Value protocol.
 *
 * The class is abstract because deriveKey() is pure virtual. Every South-Bound
 * method declared here is only DECLARED; the common implementation PR supplies
 * the definitions.
 */
class nixlSpdkKvEngine : public nixlBackendEngine {
public:
    ~nixlSpdkKvEngine() override;

    // --- nixlBackendEngine South-Bound API (shared data plane) ---------------
    // Declarations only. The single shared implementation is provided by the
    // backend PR's spdk_kv_engine.cpp; it drives everything through device_.

    bool
    supportsRemote() const override;

    bool
    supportsLocal() const override;

    bool
    supportsNotif() const override;

    nixl_mem_list_t
    getSupportedMems() const override;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs,
             std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    connect(const std::string &remote_agent) override;

    nixl_status_t
    disconnect(const std::string &remote_agent) override;

    nixl_status_t
    unloadMD(nixlBackendMD *input) override;

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

protected:
    /**
     * Compatibility construction path. A derived constructor must call
     * initDevice() after its own state is initialized. Prefer the injecting
     * overload below for new implementations.
     */
    explicit nixlSpdkKvEngine(const nixlBackendInitParams *init_params);

    /**
     * Preferred construction path. Installs an already-open device and caches
     * its effective key length during base construction. A null device marks
     * the engine initialization as failed.
     */
    nixlSpdkKvEngine(const nixlBackendInitParams *init_params,
                     std::unique_ptr<iSpdkKvDevice> device);

    /**
     * Open the device once during initialization and cache the effective key
     * length. Concrete backends call this from their constructor (virtual
     * dispatch of openDevice() only works once the derived object exists). On
     * failure it sets initErr so getInitErr() reports the error.
     * Implemented by the backend PR.
     */
    void
    initDevice();

    // --- Backend customization hooks -----------------------------------------

    /**
     * Compatibility hook that creates the KV device for this backend's target.
     * Called once by initDevice(); unused by the injecting constructor.
     * @param err On failure, set to a human-readable error message.
     * @return the opened device, or nullptr on failure (with @p err set).
     */
    virtual std::unique_ptr<iSpdkKvDevice>
    openDevice(std::string &err) const {
        err = "no SPDK KV device factory was provided";
        return nullptr;
    }

    /**
     * Map a token sequence (the OBJ_SEG descriptor's metaInfo) to a fixed-length
     * KV key of at most @p key_len bytes.
     * @return true on success; false to reject the input (e.g. empty).
     */
    virtual bool
    deriveKey(const std::string &token_seq, uint8_t key_len, std::vector<uint8_t> &out) const = 0;

    // Ratified maximum NVMe-KV inline key length (bytes); derived keys are
    // clamped to this.
    static constexpr uint8_t kMaxKeyLen = 16;

    // The KV device abstraction the shared data plane drives.
    std::unique_ptr<iSpdkKvDevice> device_;

    // Serializes a complete DMA staging and device-operation transaction. It is
    // mutable because queryMem/postXfer are const South-Bound API methods.
    mutable std::mutex deviceMutex_;

    // Effective derived-key length: min(device max key length, kMaxKeyLen).
    uint8_t maxKeyLen_ = kMaxKeyLen;

private:
    // Install a device and cache its effective key length. Used by both the
    // injecting constructor and the deferred initDevice() compatibility path.
    void
    setDevice(std::unique_ptr<iSpdkKvDevice> device);
};

#endif // NIXL_SRC_PLUGINS_SPDK_KV_SPDK_KV_ENGINE_H
