/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_TCPXO_NIXL_MEMORY_METADATA_H
#define NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_TCPXO_NIXL_MEMORY_METADATA_H

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_format.h"
#include "backend/backend_aux.h"
#include "nixl_types.h"

#ifndef TCPXO_STUB_RXDM_DXS
#include "dxs/client/dxs-client-types.h"
#else
#include "rxdm_dxs_stub.h"
#endif

namespace tcpxo {

struct MemoryHandle {
    // Used for memory registered with DXS only, dxs::kInvalidRegistration
    // otherwise
    dxs::Reg reg_handle = dxs::kInvalidRegistration;
    void *start_addr = nullptr; // starting addr of memory region
    size_t size = 0; // Number of bytes registered in this region
    int dmabuf_fd = -1; // dmabuf fd used by the memory

    // clang-format off
    bool operator==(const MemoryHandle &other) const {
        return (reg_handle == other.reg_handle) &&
               (start_addr == other.start_addr) &&
               (size == other.size) &&
               (dmabuf_fd == other.dmabuf_fd);
    }

    // clang-format on

    bool
    operator!=(const MemoryHandle &other) const {
        return !(*this == other);
    }

    template<typename Sink>
    friend void
    AbslStringify(Sink &sink, const MemoryHandle &mh) {
        absl::Format(&sink,
                     "MemoryHandle<.reg_handle=%" PRIu64
                     ", .start_addr=%p, .size=%zu, .dmabuf_fd=%d>",
                     mh.reg_handle,
                     mh.start_addr,
                     mh.size,
                     mh.dmabuf_fd);
    }
};

// Metadata for locally registered memory
class nixlTcpxoLocalMemoryMetadata : public nixlBackendMD {
public:
    nixlTcpxoLocalMemoryMetadata(dxs::Reg reg_handle,
                                 void *start_addr,
                                 size_t size,
                                 int dmabuf_fd,
                                 uint8_t fastrak_idx)
        : nixlBackendMD(true),
          mem_handle_{.reg_handle = reg_handle,
                      .start_addr = start_addr,
                      .size = size,
                      .dmabuf_fd = dmabuf_fd},
          fastrak_idx_(fastrak_idx) {}

    // Getter functions for MemoryHandle
    const MemoryHandle &
    GetMemHandle() const {
        return mem_handle_;
    }

    uint8_t
    GetFastrakIdx() const {
        return fastrak_idx_;
    }

private:
    const MemoryHandle mem_handle_;
    const uint8_t fastrak_idx_;
};

nixl_status_t
SerializeMemoryMetadata(const nixlTcpxoLocalMemoryMetadata *mem_md, std::string &str);

// The caller owns `mem_md` and is responsible for `delete`ing it.
nixl_status_t
DeserializeMemoryMetadata(const std::string &str, nixlTcpxoLocalMemoryMetadata *&mem_md);

} // namespace tcpxo

#endif // NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_TCPXO_NIXL_MEMORY_METADATA_H
