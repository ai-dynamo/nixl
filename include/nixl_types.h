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

/*** Enums of memory type, operation and status ***/

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
    NIXL_ERR_UNKNOWN = -6,
    NIXL_ERR_NOT_ALLOWED = -7,
    NIXL_ERR_NOT_POSTED = -8
} nixl_status_t;


/*** Types defined for NIXL API function arguments ***/

// std::string supports \0 natively, as long as c_str() is not called.
// To clarify the API, a wrapper around it is creatd. It can be looked
// as a void* of data, with specified length.
//class stringWrapper {
//    private:
//        std::string str;
//
//   public:
//        stringWrapper (const std::string& from_str = "")
//            : str(from_str) {}
//        size_t length() const { return str.length(); }
//        const char* data() const { return str.data(); }
//        std::string toString() const { return str; }
//}

#define NIXL_NO_MSG std::string("")
#define NIXL_INIT_AGENT ""

typedef struct {
    // Used in prepXferFull/prepXferSide/GenNotif as suggestion to limit
    // the list of backends to be explored.
    std::vector<nixlBackendH*> suggestedBackends;
} nixl_xfer_params_t;

typedef std::string nixl_backend_t;
typedef std::string nixl_blob_t;
typedef std::unordered_map<std::string, std::string> nixl_b_params_t;
typedef std::unordered_map<std::string, std::vector<nixl_blob_t>> nixl_notifs_t;

/*** Forward class declarations ***/
class nixlSerDes;
class nixlBackendH;
class nixlXferReqH;
class nixlXferSideH;
class nixlAgentData;

#endif
