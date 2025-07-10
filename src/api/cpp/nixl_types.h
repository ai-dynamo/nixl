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
 * @enum   nixlMemType
 * @brief  An enumeration of segment types for NIXL
 *         FILE_SEG must be last
 */
enum nixlMemType { DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG };
/**
 * @deprecated Use nixlMemType instead
 */
using nixl_mem_t = nixlMemType;

/**
 * @enum   nixlXferOp
 * @brief  An enumeration of different transfer types for NIXL
 */
enum nixlXferOp { NIXL_READ, NIXL_WRITE };
/**
 * @deprecated Use nixlXferOp instead
 */
using nixl_xfer_op_t = nixlXferOp;

/**
 * @enum   nixlStatus
 * @brief  An enumeration of status values and error codes for NIXL
 */
enum nixlStatus {
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
    NIXL_ERR_NOT_SUPPORTED = -9,
    NIXL_ERR_REMOTE_DISCONNECT = -10
};
/**
 * @deprecated Use nixlStatus instead
 */
using nixl_status_t = nixlStatus;

/**
 * @enum nixlThreadSync
 * @brief An enumeration of supported synchronization modes for NIXL
 */
enum class nixlThreadSync {
    NIXL_THREAD_SYNC_NONE,
    NIXL_THREAD_SYNC_STRICT,
    NIXL_THREAD_SYNC_RW,
    NIXL_THREAD_SYNC_DEFAULT = NIXL_THREAD_SYNC_NONE,
};
/**
 * @deprecated Use nixlThreadSync instead
 */
using nixl_thread_sync_t = nixlThreadSync;

/**
 * @namespace nixlEnumStrings
 * @brief     This namespace to get string representation
 *            of different enums
 */
namespace nixlEnumStrings {
std::string
memTypeStr(const nixlMemType &mem);
std::string
xferOpStr(const nixlXferOp &op);
std::string
statusStr(const nixlStatus &status);
} // namespace nixlEnumStrings


/*** NIXL aliases and constant expressions used in the API ***/

/**
 * @brief An alias for a std::string to identify nixl backends
 */
using nixlBackend = std::string;
/**
 * @deprecated Use nixlBackend instead
 */
using nixl_backend_t = nixlBackend;

/**
 * @brief An alias for a std::string as nixl blob
 *        std::string supports \0 natively, so it can be looked as a void* of data,
 *        with specified length. Giving it a new name to be clear in the API and
 *        preventing users to think it's a string and call c_str().
 */
using nixlBlob = std::string;
/**
 * @deprecated Use nixlBlob instead
 */
using nixl_blob_t = nixlBlob;

/**
 * @brief An alias for a std::vector<nixlMemType> to create nixlMemList objects.
 */
using nixlMemList = std::vector<nixlMemType>;
/**
 * @deprecated Use nixlMemList instead
 */
using nixl_mem_list_t = nixlMemList;

/**
 * @brief An alias for a std::unordered_map<std::string, std::string>
 *        to hold nixlBParams
 */
using nixlBParams = std::unordered_map<std::string, std::string>;
/**
 * @deprecated Use nixlBParams instead
 */
using nixl_b_params_t = nixlBParams;

/**
 * @brief An alias for a  std::unordered_map<std::string, std::vector<nixlBlob>>
 *        to hold nixlNotifs (nixl notifications)
 */
using nixlNotifs = std::unordered_map<std::string, std::vector<nixlBlob>>;
/**
 * @deprecated Use nixlNotifs instead
 */
using nixl_notifs_t = nixlNotifs;

/**
 * @brief A constant to define the default communication port.
 */
constexpr int default_comm_port = 8888;

/**
 * @brief A constant to define the default metadata label for ETCD server key.
 *        Appended to the agent's key prefix to form the full key for metadata.
 */
extern const std::string default_metadata_label;

/**
 * @brief A constant to define the default partial metadata label for ETCD server key.
 *        Appended to the agent's key prefix to form the full key for partial metadata.
 */
extern const std::string default_partial_metadata_label;


/**
 * @enum nixlCost
 * @brief An enumeration of cost types for transfer cost estimation.
 */
enum class nixlCost {
    ANALYTICAL_BACKEND = 0, // Analytical backend cost estimate
};
/**
 * @deprecated Use nixlCost instead
 */
using nixl_cost_t = nixlCost;

/**
 * @struct nixlAgentOptionalArgs
 * @brief A structure for optional argument that can be provided to relevant agent methods.
 */
struct nixlAgentOptionalArgs {
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
    nixlBlob notifMsg;

    /**
     * @var hasNotif boolean value to indicate that a notification is provided, or to
     *      remove notification during a repost. If set to false, notifMsg is not checked.
     */
    bool hasNotif = false;

    /**
     * @var makeXferReq boolean to skip merging consecutive descriptors, used in makeXferReq.
     */
    bool skipDescMerge = false;

    /**
     * @var includeConnInfo boolean to include connection information in the metadata,
     *                      used in getLocalPartialMD.
     */
    bool includeConnInfo = false;

    /**
     * @var ipAddr Used to specify the IP address of a remote peer for metadata transfer.
     *                      used in sendLocalMD, fetchRemoteMD, invalidateLocalMD, sendLocalPartialMD.
     */
    std::string ipAddr;

    /**
     * @var port Used to specify the port of a remote peer, ipAddr must also be set
     *                      used in sendLocalMD, fetchRemoteMD, invalidateLocalMD, sendLocalPartialMD.
     */
    int port = default_comm_port;

    /**
     * @var metadataLabel Used to specify the label of the metadata to be sent/fetched
     *                    when working with ETCD metadata server. The label will be appended to the
     *                    agent's key prefix, and the full key will be used to store/fetch
     *                    the metadata key-value pair from the server.
     *                    Used in fetchRemoteMD, sendLocalPartialMD.
     *                    Note that sendLocalMD always uses default_metadata_label and ignores this parameter.
     *                    Note that invalidateLocalMD invalidates all labels and ignores this parameter.
     */
    std::string metadataLabel;

    /**
     * @var Backend custom parameter
     */
    nixlBlob customParam;
};
/**
 * @deprecated Use nixlAgentOptionalArgs directly instead
 */
using nixl_opt_args_t = nixlAgentOptionalArgs;

/**
 * @brief A define for an empty string, that indicates the descriptor list is being
 *        prepared for the local agent as an initiator in prepXferDlist method.
 */
#define NIXL_INIT_AGENT ""

#endif
