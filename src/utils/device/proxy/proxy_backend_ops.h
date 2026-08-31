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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_BACKEND_OPS_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_BACKEND_OPS_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <nixl_types.h>
#include "backend_aux.h"
#include "proxy_config.h"
#include "proxy_protocol.h"

namespace nixl {

/** One side of a transfer, resolved from the registry against a ring record. */
struct proxyBackendXferDesc {
    nixl_mem_t mem_type = VRAM_SEG;
    /** nixlMetaDesc() leaves addr, len and devId unset, so zero them explicitly. */
    nixlMetaDesc desc{0, 0, 0, nullptr};
};

/** A ring record with its view tokens resolved into transport descriptors. */
struct proxyBackendSubmission {
    uint64_t op_idx = 0;
    nixl_proxy_opcode_t opcode = nixl_proxy_opcode_t::PUT;
    uint32_t channel_id = 0;
    uint32_t peer_index = 0;
    uint64_t flags = 0;

    proxyBackendXferDesc local;
    proxyBackendXferDesc remote;

    size_t size = 0;
    uint64_t value = 0;
};

/** Opaque backend handle for one in-flight transfer. */
struct proxyBackendRequest {
    uint64_t token = 0;
    size_t context = 0;

    explicit
    operator bool() const noexcept {
        return token != 0;
    }
};

/** Backend callbacks used by the owning proxy runtime; not a public ABI. */
struct proxyBackendOps {
    /** Required; the runtime rejects a struct with any of these unset. */
    std::function<nixl_status_t(const proxyConfig &)> init;
    std::function<nixl_status_t(const proxyBackendSubmission &, proxyBackendRequest &)> submit;
    std::function<nixl_status_t(const proxyBackendRequest &)> check_completion;
    /** Establish transport quiescence before view retirement; runs on the owning worker. */
    std::function<nixl_status_t(uint32_t channel, uint32_t peer)> quiesce;
    std::function<nixl_status_t(uint32_t channel, uint32_t peer)> progress;
    std::function<nixl_status_t()> shutdown;

    /** Optional direct-access pointers, indexed by descriptor. */
    std::function<nixl_status_t(const nixl_remote_meta_dlist_t &, std::vector<void *> &)>
        resolve_direct_ptrs;

    [[nodiscard]] bool
    complete() const noexcept {
        return init && submit && check_completion && quiesce && progress && shutdown;
    }
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_BACKEND_OPS_H
