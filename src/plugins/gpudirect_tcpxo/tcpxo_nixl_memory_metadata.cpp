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

#include "tcpxo_nixl_memory_metadata.h"

#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include "serdes/serdes.h"
#include "tcpxo_common.h"

namespace tcpxo {

nixl_status_t
SerializeMemoryMetadata(const nixlTcpxoLocalMemoryMetadata *mem_md, std::string &str) {
    nixlSerDes ser_des;
    MemoryHandle mem_handle = mem_md->GetMemHandle();
    uint8_t fastrak_idx = mem_md->GetFastrakIdx();

    RETURN_IF_NOT_NIXL_SUCCESS(
        ser_des.addBuf("dxs_handle", &mem_handle.reg_handle, sizeof(mem_handle.reg_handle)));
    RETURN_IF_NOT_NIXL_SUCCESS(
        ser_des.addBuf("start_addr", &mem_handle.start_addr, sizeof(mem_handle.start_addr)));
    RETURN_IF_NOT_NIXL_SUCCESS(ser_des.addBuf("size", &mem_handle.size, sizeof(mem_handle.size)));
    RETURN_IF_NOT_NIXL_SUCCESS(
        ser_des.addBuf("dmabuf_fd", &mem_handle.dmabuf_fd, sizeof(mem_handle.dmabuf_fd)));
    RETURN_IF_NOT_NIXL_SUCCESS(ser_des.addBuf("fastrak_idx", &fastrak_idx, sizeof(fastrak_idx)));
    str = ser_des.exportStr();
    return NIXL_SUCCESS;
}

nixl_status_t
DeserializeMemoryMetadata(const std::string &str, nixlTcpxoLocalMemoryMetadata *&mem_md) {
    nixlSerDes ser_des;
    RETURN_IF_NOT_NIXL_SUCCESS(ser_des.importStr(str));

    dxs::Reg reg_handle = dxs::kInvalidRegistration;
    RETURN_IF_NOT_NIXL_SUCCESS(ser_des.getBuf("dxs_handle", &reg_handle, sizeof(reg_handle)));

    void *start_addr = nullptr;
    RETURN_IF_NOT_NIXL_SUCCESS(ser_des.getBuf("start_addr", &start_addr, sizeof(start_addr)));

    size_t size = 0;
    RETURN_IF_NOT_NIXL_SUCCESS(ser_des.getBuf("size", &size, sizeof(size)));

    int dmabuf_fd = -1;
    RETURN_IF_NOT_NIXL_SUCCESS(ser_des.getBuf("dmabuf_fd", &dmabuf_fd, sizeof(dmabuf_fd)));

    uint8_t fastrak_idx = 0;
    RETURN_IF_NOT_NIXL_SUCCESS(ser_des.getBuf("fastrak_idx", &fastrak_idx, sizeof(fastrak_idx)));

    mem_md = new nixlTcpxoLocalMemoryMetadata(reg_handle, start_addr, size, dmabuf_fd, fastrak_idx);

    return NIXL_SUCCESS;
}

} // namespace tcpxo
