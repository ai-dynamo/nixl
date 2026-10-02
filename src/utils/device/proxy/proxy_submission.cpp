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
#include "proxy_submission.h"

#include "nixl_log.h"
#include "proxy_registry_entry.h"

namespace nixl {

namespace {

    bool
    rangeFits(const nixlMetaDesc &desc, size_t offset, size_t size) {
        return offset <= desc.len && size <= desc.len - offset;
    }

    nixl_status_t
    lookupDesc(uint64_t host_view,
               size_t index,
               size_t offset,
               size_t size,
               bool want_remote,
               const registryEntry *&entry_out,
               const registryEntry::storedDesc *&desc_out) {
        entry_out = nullptr;
        desc_out = nullptr;

        const char *const role = want_remote ? "dst" : "src";
        const auto *entry =
            reinterpret_cast<const registryEntry *>(static_cast<uintptr_t>(host_view));
        if (entry == nullptr) {
            NIXL_DEBUG << "resolveSubmission: " << role << " not ready, host_view=" << host_view;
            return NIXL_ERR_NOT_FOUND;
        }
        if (entry->remote != want_remote) {
            NIXL_DEBUG << "resolveSubmission: " << role
                       << " has the wrong role, host_view=" << host_view;
            return NIXL_ERR_INVALID_PARAM;
        }
        if (index >= entry->descs.size()) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const registryEntry::storedDesc &desc = entry->descs[index];
        if (!desc.usable || !rangeFits(desc.desc, offset, size)) {
            return NIXL_ERR_INVALID_PARAM;
        }

        entry_out = entry;
        desc_out = &desc;
        return NIXL_SUCCESS;
    }

} // namespace

nixl_status_t
resolveSubmission(const nixlProxyCommand &command,
                  uint32_t channel,
                  uint32_t peer,
                  proxyBackendSubmission &prepared_submission) noexcept {
    bool needs_source = false;
    size_t transfer_size = 0;
    switch (command.opcode) {
    case nixl_proxy_opcode_t::PUT:
        needs_source = true;
        transfer_size = command.size;
        break;
    case nixl_proxy_opcode_t::ATOMIC_ADD:
        transfer_size = sizeof(uint64_t);
        break;
    default:
        NIXL_ERROR << "resolveSubmission: unsupported opcode: "
                   << static_cast<uint32_t>(command.opcode);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const registryEntry *dst_entry = nullptr;
    const registryEntry::storedDesc *dst_desc = nullptr;
    nixl_status_t status = lookupDesc(command.dst_view,
                                      command.dst_index,
                                      command.dst_offset,
                                      transfer_size,
                                      /*want_remote=*/true,
                                      dst_entry,
                                      dst_desc);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    proxyBackendSubmission prepared{};
    prepared.op_idx = command.op_idx;
    prepared.opcode = command.opcode;
    prepared.channel_id = channel;
    prepared.peer_index = peer;
    prepared.size = transfer_size;
    prepared.value = needs_source ? 0 : command.atomicValue();
    prepared.remote = dst_desc->desc;
    prepared.remote.addr += command.dst_offset;
    prepared.remote.len = transfer_size;

    if (needs_source) {
        const registryEntry *src_entry = nullptr;
        const registryEntry::storedDesc *src_desc = nullptr;
        status = lookupDesc(command.src_view,
                            command.src_index,
                            command.sourceOffset(),
                            transfer_size,
                            /*want_remote=*/false,
                            src_entry,
                            src_desc);
        if (status != NIXL_SUCCESS) {
            return status;
        }

        prepared.local = src_desc->desc;
        prepared.local.addr += command.sourceOffset();
        prepared.local.len = transfer_size;
    }

    prepared_submission = prepared;
    return NIXL_SUCCESS;
}

} // namespace nixl
