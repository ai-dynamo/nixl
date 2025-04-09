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

/**
 * @enum   nixl_mem_t
 * @brief  An enumeration of segment types for NIXL
 *         FILE_SEG must be last
 */
typedef enum {DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG} nixl_mem_t;

/**
 * @enum   nixl_xfer_op_t
 * @brief  An enumeration of different transfer types for NIXL
 */
typedef enum {NIXL_READ, NIXL_WRITE} nixl_xfer_op_t;

/**
 * @enum   nixl_status_t
 * @brief  An enumeration of status values and error codes for NIXL
 */
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

/**
 * @namespace nixlEnumStrings
 * @brief     This namespace to get string representation
 *            of different enums
 */
namespace nixlEnumStrings {
    std::string memTypeStr(const nixl_mem_t &mem);
    std::string xferOpStr (const nixl_xfer_op_t &op);
    std::string statusStr (const nixl_status_t &status);
}


/*** NIXL typedefs and defines used in the API ***/

/**
 * @brief A typedef for a std::string to identify nixl backends
 */
typedef std::string nixl_backend_t;

/**
 * @brief A typedef for a std::string as nixl blob
 *        std::string supports \0 natively, so it can be looked as a void* of data,
 *        with specified length. Giving it a new name to be clear in the API and
 *        preventing users to think it's a string and call c_str().
 */
typedef std::string nixl_blob_t;

/**
 * @brief A typedef for a std::vector<nixl_mem_t> to create nixl_mem_list_t objects.
 */
typedef std::vector<nixl_mem_t> nixl_mem_list_t;

/**
 * @brief A typedef for a  std::unordered_map<std::string, std::string>
 *        to hold nixl_b_params_t .
 */
typedef std::unordered_map<std::string, std::string> nixl_b_params_t;

/**
 * @brief A typedef for a  std::unordered_map<std::string, std::vector<nixl_blob_t>>
 *        to hold nixl_notifs_t (nixl notifications)
 */
typedef std::unordered_map<std::string, std::vector<nixl_blob_t>> nixl_notifs_t;

/**
 * @class nixlAgentOptionalArgs
 * @brief A class for optional argument that can be provided to relevant agent methods.
 */
class nixlAgentOptionalArgs {
    public:
        /**
         * @var backends vector to specify a list of backend handles, to limit the list
         *      of backends to be considered. Used in registerMem / deregisterMem
         *      makeConnection / prepXferDlist / makeXferReq / createXferReq / GetNotifs / GenNotif
         */
        std::vector<nixlBackendH*> backends;

        /**
         * @var notifMsg A message to be used in createXferReq / makeXferReq / postXferReq,
         *               if a notification message is desired
         */
        nixl_blob_t notifMsg;
        /**
         * @var hasNotif boolean value to indicate that a notification is provided, or to
         *      remove notification during a repost. If set to false, notifMsg is not checked.
         */
        bool hasNotif = false;

        /**
         * @var makeXferReq boolean to skip merging consecutive descriptors, used in makeXferReq.
         */
        bool skipDescMerge = false;
};
/**
 * @brief A typedef for a nixlAgentOptionalArgs
 *        for providing extra optional arguments
 */
typedef nixlAgentOptionalArgs nixl_opt_args_t;

/**
 * @brief A define for an empty string, that indicates the descriptor list is being
 *        prepared for the local agent as an initiator in prepXferDlist method.
 */
#define NIXL_INIT_AGENT ""

#endif
