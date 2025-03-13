/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _NIXL_TYPES_H
#define _NIXL_TYPES_H
#include <vector>
#include <string>
#include <unordered_map>

typedef std::unordered_map<std::string, std::string> nixl_b_params_t;
typedef std::unordered_map<std::string, std::vector<std::string>> nixl_notifs_t;
typedef std::string nixl_backend_t;

#define NIXL_NO_MSG stringWrapper("")
#define NIXL_INIT_AGENT ""

// FILE_SEG must be last
typedef enum {DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG} nixl_mem_t;

typedef enum {NIXL_READ,  NIXL_RD_NOTIF,
              NIXL_WRITE, NIXL_WR_NOTIF} nixl_xfer_op_t;

typedef enum {
    NIXL_IN_PROG = 1,
    NIXL_SUCCESS = 0,
    NIXL_ERR_INVALID_PARAM = -1,
    NIXL_ERR_BACKEND = -2,
    NIXL_ERR_NOT_FOUND = -3,
    NIXL_ERR_NYI = -4,
    NIXL_ERR_MISMATCH = -5,
    NIXL_ERR_BAD = -6,
    NIXL_ERR_NOT_ALLOWED = -7,
    NIXL_NOT_POSTED = -8
} nixl_status_t;


typedef struct {
    // Used in createBackend/prepXferFull/prepXferSide/GenNotif
    // as suggestion to limit the list of backends to be explored.
    std::vector<nixlBackendH*> suggestedBackends;

    nixl_xfer_op_t operation;
    bool validOp    = false;

    nixl_blob_t &notif_msg;
    bool validNotif = false;
} nixl_xfer_params_t;


class nixlSerDes;
class nixlBackendH;
class nixlXferReqH;
class nixlXferSideH;
class nixlAgentData;

#endif
