/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_TRANSPORT_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_TRANSPORT_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include <nixl_types.h>
#include "backend_aux.h"
#include "proxy_config.h"
#include "proxy_protocol.h"

namespace nixl {

/** A ring command with its view tokens resolved into transport descriptors. */
struct proxyBackendSubmission {
    uint64_t op_idx = 0;
    nixl_proxy_opcode_t opcode = nixl_proxy_opcode_t::PUT;
    uint32_t channel_id = 0;
    uint32_t peer_index = 0;

    /** nixlMetaDesc() leaves addr, len and devId unset, so zero them explicitly. */
    nixlMetaDesc local{0, 0, 0, nullptr};
    nixlMetaDesc remote{0, 0, 0, nullptr};

    size_t size = 0;
    uint64_t value = 0;
};

/** One in-flight transfer, named by a single token. */
struct proxyBackendRequest {
    uint64_t token = 0;

    explicit
    operator bool() const noexcept {
        return token != 0;
    }
};

/**
 * The transport a proxyRuntime drives. Internal to libnixl_device_proxy and the plugins; not
 * installed, not an ABI. Ownership, threading and request rules: README.md in this directory.
 */
class proxyTransport {
public:
    virtual ~proxyTransport() = default;

    /**
     * Called once by proxyRuntime::create() before any ring exists. On failure nothing else is
     * called.
     */
    [[nodiscard]] virtual nixl_status_t
    init(const proxyConfig &config) = 0;

    /** NIXL_IN_PROG with a non-empty request, or a terminal status with an empty one; never a
     *  transient error (post from progress() instead). */
    [[nodiscard]] virtual nixl_status_t
    submit(const proxyBackendSubmission &submission, proxyBackendRequest &request) = 0;

    /** Polled in submission order per ring. `channel` and `peer` select that ring. A terminal
     *  status releases the request, which is not passed again. */
    [[nodiscard]] virtual nixl_status_t
    checkCompletion(uint32_t channel, uint32_t peer, const proxyBackendRequest &request) = 0;

    /** Once per ring per worker pass, busy or idle; an idle ring should return quickly.
     *  Transfer errors surface through checkCompletion(). */
    virtual void
    progress(uint32_t channel, uint32_t peer) noexcept = 0;

    /** On the owning thread, once the ring has no outstanding request. On success every operation
     *  submitted on the ring is remotely visible and no longer references local or remote memory.
     *  A failure is fatal to the process. */
    [[nodiscard]] virtual nixl_status_t
    quiesce(uint32_t channel, uint32_t peer) = 0;

    /** Once, after a successful init(), with workers joined and every ring drained. */
    [[nodiscard]] virtual nixl_status_t
    shutdown() = 0;

    /**
     * Optional: leave direct_ptrs empty (no direct access) or fill one entry per descriptor,
     * nullptr where unavailable. Any error is a real error.
     */
    [[nodiscard]] virtual nixl_status_t
    resolveDirectPtrs(const nixl_remote_meta_dlist_t &, std::vector<void *> &direct_ptrs) {
        direct_ptrs.clear();
        return NIXL_SUCCESS;
    }
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_TRANSPORT_H
