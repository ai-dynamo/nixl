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
#include <optional>
#include <cstdint>
#include <cstring>

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
enum nixl_mem_t {DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG};

/**
 * @enum   nixl_xfer_op_t
 * @brief  An enumeration of different transfer types for NIXL
 */
enum nixl_xfer_op_t {NIXL_READ, NIXL_WRITE};

/**
 * @enum   nixl_status_t
 * @brief  An enumeration of status values and error codes for NIXL
 */
enum nixl_status_t {
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
    NIXL_ERR_REMOTE_DISCONNECT = -10,
    NIXL_ERR_CANCELED = -11,
    NIXL_ERR_LAST = -11
};

/**
 * @enum nixl_thread_sync_t
 * @brief An enumeration of supported synchronization modes for NIXL
 */
enum class nixl_thread_sync_t {
    NIXL_THREAD_SYNC_NONE,
    NIXL_THREAD_SYNC_STRICT,
    NIXL_THREAD_SYNC_RW,
    NIXL_THREAD_SYNC_DEFAULT = NIXL_THREAD_SYNC_NONE,
};

/**
 * @enum nixl_telemetry_category_t
 * @brief An enumeration of main telemetry event categories for easy filtering and aggregation
 */
enum class nixl_telemetry_category_t {
    NIXL_TELEMETRY_MEMORY = 0, // Memory operations (register, deregister, allocation)
    NIXL_TELEMETRY_TRANSFER = 1, // Data transfer operations (read, write)
    NIXL_TELEMETRY_CONNECTION = 2, // Connection management (connect, disconnect)
    NIXL_TELEMETRY_BACKEND = 3, // Backend-specific operations
    NIXL_TELEMETRY_ERROR = 4, // Error events
    NIXL_TELEMETRY_PERFORMANCE = 5, // Performance metrics
    NIXL_TELEMETRY_SYSTEM = 6, // System-level events
    NIXL_TELEMETRY_CUSTOM = 7, // Custom/user-defined events
    NIXL_TELEMETRY_MAX = 8
};

/**
 * @namespace nixlEnumStrings
 * @brief     This namespace to get string representation
 *            of different enums
 */
namespace nixlEnumStrings {
    std::string memTypeStr(const nixl_mem_t &mem);
    std::string xferOpStr (const nixl_xfer_op_t &op);
    std::string statusStr (const nixl_status_t &status);
    std::string
    telemetryCategoryStr(const nixl_telemetry_category_t &category);
}


/*** NIXL typedefs and defines used in the API ***/

/**
 * @brief A typedef for a std::string to identify nixl backends
 */
using nixl_backend_t = std::string;

/**
 * @brief A typedef for a std::string as nixl blob
 *        std::string supports \0 natively, so it can be looked as a void* of data,
 *        with specified length. Giving it a new name to be clear in the API and
 *        preventing users to think it's a string and call c_str().
 */
using nixl_blob_t = std::string;

/**
 * @brief A typedef for a std::vector<nixl_mem_t> to create nixl_mem_list_t objects.
 */
using nixl_mem_list_t = std::vector<nixl_mem_t>;

/**
 * @brief A typedef for a  std::unordered_map<std::string, std::string>
 *        to hold nixl_b_params_t .
 */
using nixl_b_params_t = std::unordered_map<std::string, std::string>;

/**
 * @brief A typedef for a  std::unordered_map<std::string, std::vector<nixl_blob_t>>
 *        to hold nixl_notifs_t (nixl notifications)
 */
using nixl_notifs_t = std::unordered_map<std::string, std::vector<nixl_blob_t>>;

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

constexpr char TELEMETRY_PREFIX[] = "nixl_telemetry";
constexpr int TELEMETRY_VERSION = 1;

constexpr size_t MAX_EVENT_NAME_LEN = 32;

/**
 * @struct nixlTelemetryEvent
 * @brief A structure to hold individual telemetry event data for cyclic buffer storage
 */
struct nixlTelemetryEvent {
    uint64_t timestampUs_; // Event timestamp in microseconds
    nixl_telemetry_category_t category_; // Main event category for filtering
    char eventName_[MAX_EVENT_NAME_LEN]; // Detailed event name/identifier
    uint64_t value_; // Numeric value associated with the event
    nixlTelemetryEvent() = default;

    nixlTelemetryEvent(uint64_t timestamp_us,
                       nixl_telemetry_category_t category,
                       const std::string &event_name,
                       uint64_t value)
        : timestampUs_(timestamp_us),
          category_(category),
          value_(value) {
        strncpy(eventName_, event_name.c_str(), MAX_EVENT_NAME_LEN - 1);
        eventName_[MAX_EVENT_NAME_LEN - 1] = '\0';
    }
};

/**
 * @enum nixl_cost_t
 * @brief An enumeration of cost types for transfer cost estimation.
 */
enum class nixl_cost_t {
    ANALYTICAL_BACKEND = 0, // Analytical backend cost estimate
};

/**
 * @brief A typedef for std::optional<nixl_b_params_t> for querying memory results
 *        Validity of a nixl_query_resp_t can be checked by has_value() method,
 *        and if true, the dictionary can be accessed by value() method.
 */
using nixl_query_resp_t = std::optional<nixl_b_params_t>;


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
    nixl_blob_t customParam;
};
/**
 * @brief A typedef for a nixlAgentOptionalArgs
 *        for providing extra optional arguments
 */
using nixl_opt_args_t = nixlAgentOptionalArgs;

/**
 * @brief A define for an empty string, that indicates the descriptor list is being
 *        prepared for the local agent as an initiator in prepXferDlist method.
 */
#define NIXL_INIT_AGENT ""

#endif
