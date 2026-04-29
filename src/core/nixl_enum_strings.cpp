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

#include "nixl_types.h"

#include <string>

namespace nixlEnumStrings {

std::string
memTypeStr(const nixl_mem_t &mem) {
    switch (mem) {
    case DRAM_SEG:
        return "DRAM_SEG";
    case VRAM_SEG:
        return "VRAM_SEG";
    case BLK_SEG:
        return "BLK_SEG";
    case OBJ_SEG:
        return "OBJ_SEG";
    case FILE_SEG:
        return "FILE_SEG";
    }
    return "BAD_SEG";
}

std::string
xferOpStr(const nixl_xfer_op_t &op) {
    switch (op) {
    case NIXL_READ:
        return "READ";
    case NIXL_WRITE:
        return "WRITE";
    }
    return "BAD_OP";
}

std::string
statusStr(const nixl_status_t &status) {
    switch (status) {
    case NIXL_IN_PROG:
        return "NIXL_IN_PROG";
    case NIXL_SUCCESS:
        return "NIXL_SUCCESS";
    case NIXL_ERR_NOT_POSTED:
        return "NIXL_ERR_NOT_POSTED";
    case NIXL_ERR_INVALID_PARAM:
        return "NIXL_ERR_INVALID_PARAM";
    case NIXL_ERR_BACKEND:
        return "NIXL_ERR_BACKEND";
    case NIXL_ERR_NOT_FOUND:
        return "NIXL_ERR_NOT_FOUND";
    case NIXL_ERR_MISMATCH:
        return "NIXL_ERR_MISMATCH";
    case NIXL_ERR_NOT_ALLOWED:
        return "NIXL_ERR_NOT_ALLOWED";
    case NIXL_ERR_REPOST_ACTIVE:
        return "NIXL_ERR_REPOST_ACTIVE";
    case NIXL_ERR_UNKNOWN:
        return "NIXL_ERR_UNKNOWN";
    case NIXL_ERR_NOT_SUPPORTED:
        return "NIXL_ERR_NOT_SUPPORTED";
    case NIXL_ERR_REMOTE_DISCONNECT:
        return "NIXL_ERR_REMOTE_DISCONNECT";
    case NIXL_ERR_CANCELED:
        return "NIXL_ERR_CANCELED";
    case NIXL_ERR_NO_TELEMETRY:
        return "NIXL_ERR_NO_TELEMETRY";
    }
    return "BAD_STATUS";
}

} // namespace nixlEnumStrings
