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


/*** Forward declarations ***/
class nixlSerDes;
class nixlDlistH;
class nixlBackendH;
class nixlXferReqH;
class nixlAgentData;


/*** NIXL memory type, operation and status enums ***/

// FILE_SEG must be last
typedef enum {DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG} nixl_mem_t;

typedef enum {NIXL_READ, NIXL_WRITE} nixl_xfer_op_t;

typedef enum {
    NIXL_IN_PROG = 1,
    NIXL_SUCCESS = 0,
    NIXL_ERR_NOT_POSTED = -1,
    NIXL_ERR_INVALID_PARAM = -2,
    NIXL_ERR_BACKEND = -3,
    NIXL_ERR_NOT_FOUND = -4,
    NIXL_ERR_MISMATCH = -5,
    NIXL_ERR_NOT_ALLOWED = -6,
    NIXL_ERR_REPOST_ACTIVE = -7,
    NIXL_ERR_UNKNOWN = -8,
    NIXL_ERR_NOT_SUPPORTED = -9
} nixl_status_t;

// namespace to get string representation of different enums
namespace nixlEnumStrings {
    std::string memTypeStr(const nixl_mem_t &mem);
    std::string xferOpStr (const nixl_xfer_op_t &op);
    std::string statusStr (const nixl_status_t &status);
}


/*** NIXL typedefs and defines used in the API ***/

typedef std::string nixl_backend_t;
// std::string supports \0 natively, So it can be looked as a void* of data,
// with specified length. Giving it a new name to be clear in the API and
// preventing users to think it's a string and call c_str().
typedef std::string nixl_blob_t;
typedef std::vector<nixl_mem_t> nixl_mem_list_t;
typedef std::unordered_map<std::string, std::string> nixl_b_params_t;
typedef std::unordered_map<std::string, std::vector<nixl_blob_t>> nixl_notifs_t;

class nixlAgentOptionalArgs {
    public:
        // Used in createBackend / createXferReq / prepXferDlist
        //         makeXferReq   / GetNotifs     / GenNotif
        // As suggestion to limit the list of backends to be explored, and
        // the preference among them, first being the most preferred.
        std::vector<nixlBackendH*> backends;

        // Used in createXferReq / makeXferReq / postXferReq,
        // if a notification message is desired, and the corresponding indicator
        nixl_blob_t notifMsg;
        bool hasNotif = false;

        // Used in makeXferReq, to skip merging of consecutive descriptors
        bool skipDescMerge = false;
};
typedef nixlAgentOptionalArgs nixl_opt_args_t;

#define NIXL_INIT_AGENT ""

#endif
