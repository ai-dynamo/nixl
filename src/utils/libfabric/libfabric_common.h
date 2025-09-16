/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H

#include <vector>
#include <string>
#include <unordered_set>
#include <cstring>

#include "nixl.h"

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>

// Libfabric configuration constants
#define NIXL_LIBFABRIC_DEFAULT_CONTROL_RAILS 1
#define NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE 8192
#define NIXL_LIBFABRIC_CQ_SREAD_TIMEOUT_SEC 1
#define NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD (128 * 1024) // 128KB
#define NIXL_LIBFABRIC_MAX_XFER_IDS 1024 // Maximum XFER_IDs per notification
#define LF_EP_NAME_MAX_LEN 56

// The immediate data associated with an RDMA operation is 32 bits and is divided as follows:
// | 4-bit MSG TYPE flag | 8-bit agent index | 20-bit XFER_ID |

// Optimized bit field constants (compile-time computed)
#define NIXL_MSG_TYPE_BITS 4
#define NIXL_AGENT_INDEX_BITS 8
#define NIXL_XFER_ID_BITS 20

// Pre-computed shift amounts for better performance
#define NIXL_MSG_TYPE_SHIFT 0
#define NIXL_AGENT_INDEX_SHIFT 4
#define NIXL_XFER_ID_SHIFT 12

// Pre-computed masks (compile-time constants)
#define NIXL_MSG_TYPE_MASK 0xFU // 0x0000000F (4 bits)
#define NIXL_AGENT_INDEX_MASK 0xFFU // 0x000000FF (8 bits)
#define NIXL_XFER_ID_MASK 0xFFFFFU // 0x000FFFFF (20 bits)

// Message type constants
#define NIXL_LIBFABRIC_MSG_CONNECT 0
#define NIXL_LIBFABRIC_MSG_ACK 1
#define NIXL_LIBFABRIC_MSG_NOTIFICTION 2
#define NIXL_LIBFABRIC_MSG_DISCONNECT 3
#define NIXL_LIBFABRIC_MSG_TRANSFER 4

// Single-operation immediate data extraction (no intermediate shifts)
#define NIXL_GET_MSG_TYPE_FROM_IMM(data) ((data) & NIXL_MSG_TYPE_MASK)
#define NIXL_GET_AGENT_INDEX_FROM_IMM(data) \
    (((data) >> NIXL_AGENT_INDEX_SHIFT) & NIXL_AGENT_INDEX_MASK)
#define NIXL_GET_XFER_ID_FROM_IMM(data) (((data) >> NIXL_XFER_ID_SHIFT) & NIXL_XFER_ID_MASK)

// Single-operation immediate data creation (minimal bit operations)
#define NIXL_MAKE_IMM_DATA(msg_type, agent_idx, xfer_id)                           \
    (((uint64_t)(msg_type) & NIXL_MSG_TYPE_MASK) |                                 \
     (((uint64_t)(agent_idx) & NIXL_AGENT_INDEX_MASK) << NIXL_AGENT_INDEX_SHIFT) | \
     (((uint64_t)(xfer_id) & NIXL_XFER_ID_MASK) << NIXL_XFER_ID_SHIFT))

/**
 * @brief Binary notification format to eliminate SerDes string operations
 *
 * This structure provides a fixed-size, binary format for notifications
 * to avoid expensive string serialization/deserialization operations.
 * Used for high-performance notification passing between agents.
 */
struct BinaryNotification {
    char agent_name[256]; // Fixed-size agent name (null-terminated)
    char message[1024]; // Fixed-size message (null-terminated)
    uint32_t xfer_id_count; // Number of XFER_IDs
    uint32_t
        xfer_ids[NIXL_LIBFABRIC_MAX_XFER_IDS]; // Fixed array of XFER_IDs (max 128 per notification)

    /** @brief Clear all fields to zero */
    void
    clear() {
        memset(this, 0, sizeof(BinaryNotification));
    }

    /** @brief Set agent name with bounds checking */
    void
    setAgentName(const std::string &name) {
        strncpy(agent_name, name.c_str(), sizeof(agent_name) - 1);
        agent_name[sizeof(agent_name) - 1] = '\0';
    }

    /** @brief Set message with bounds checking */
    void
    setMessage(const std::string &msg) {
        strncpy(message, msg.c_str(), sizeof(message) - 1);
        message[sizeof(message) - 1] = '\0';
    }

    /** @brief Add XFER_ID if space available */
    void
    addXferId(uint32_t xfer_id) {
        if (xfer_id_count < NIXL_LIBFABRIC_MAX_XFER_IDS) {
            xfer_ids[xfer_id_count++] = xfer_id;
        }
    }

    /** @brief Get agent name as string */
    std::string
    getAgentName() const {
        return std::string(agent_name);
    }

    /** @brief Get message as string */
    std::string
    getMessage() const {
        return std::string(message);
    }

    /** @brief Get all XFER_IDs as unordered set */
    std::unordered_set<uint32_t>
    getXferIds() const {
        std::unordered_set<uint32_t> result;
        for (uint32_t i = 0; i < xfer_id_count; ++i) {
            result.insert(xfer_ids[i]);
        }
        return result;
    }
};

// Global XFER_ID management
namespace LibfabricUtils {
// Pre-allocate XFER_IDs during initialization (NOT fast path)
std::vector<uint32_t>
preallocateXferIds(size_t count);
} // namespace LibfabricUtils

// Utility functions
namespace LibfabricUtils {
// Device discovery
std::vector<std::string>
getAvailableEfaDevices();
// String utilities
std::string
hexdump(const void *data);
} // namespace LibfabricUtils

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H
