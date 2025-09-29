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
#include <unordered_map>
#include <cstring>

#include "nixl.h"

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>

// Libfabric configuration constants
#define NIXL_LIBFABRIC_DEFAULT_CONTROL_RAILS 1
#define NIXL_LIBFABRIC_CQ_SREAD_TIMEOUT_SEC 1
#define NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD (128 * 1024) // 128KB
#define LF_EP_NAME_MAX_LEN 56

// Request pool configuration constants
#define NIXL_LIBFABRIC_CONTROL_REQUESTS_PER_RAIL 1024 // SEND/RECV operations (1:1 with buffers)
#define NIXL_LIBFABRIC_DATA_REQUESTS_PER_RAIL 1024 // WRITE/read operations (no buffers)
#define NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE 8192

// The immediate data associated with an RDMA operation is 32 bits and is divided as follows:
// | 4-bit MSG TYPE flag | 8-bit agent index | 16-bit XFER_ID | 4-bit SEQ_ID |

// Optimized bit field constants (compile-time computed)
#define NIXL_MSG_TYPE_BITS 4
#define NIXL_AGENT_INDEX_BITS 8
#define NIXL_XFER_ID_BITS 16
#define NIXL_SEQ_ID_BITS 4

// Pre-computed shift amounts for better performance
#define NIXL_MSG_TYPE_SHIFT 0
#define NIXL_AGENT_INDEX_SHIFT 4
#define NIXL_XFER_ID_SHIFT 12
#define NIXL_SEQ_ID_SHIFT 28

// Pre-computed masks (compile-time constants)
#define NIXL_MSG_TYPE_MASK 0xFU // 0x0000000F (4 bits)
#define NIXL_AGENT_INDEX_MASK 0xFFU // 0x000000FF (8 bits)
#define NIXL_XFER_ID_MASK 0xFFFFU // 0x0000FFFF (16 bits)
#define NIXL_SEQ_ID_MASK 0xFU // 0x0000000F (4 bits)

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
#define NIXL_GET_SEQ_ID_FROM_IMM(data) (((data) >> NIXL_SEQ_ID_SHIFT) & NIXL_SEQ_ID_MASK)

// Single-operation immediate data creation (minimal bit operations)
#define NIXL_MAKE_IMM_DATA(msg_type, agent_idx, xfer_id, seq_id)                   \
    (((uint64_t)(msg_type) & NIXL_MSG_TYPE_MASK) |                                 \
     (((uint64_t)(agent_idx) & NIXL_AGENT_INDEX_MASK) << NIXL_AGENT_INDEX_SHIFT) | \
     (((uint64_t)(xfer_id) & NIXL_XFER_ID_MASK) << NIXL_XFER_ID_SHIFT) |           \
     (((uint64_t)(seq_id) & NIXL_SEQ_ID_MASK) << NIXL_SEQ_ID_SHIFT))

/**
 * @brief Binary notification format with counter-based matching
 *
 * This structure provides a fixed-size, binary format for notifications
 */
struct BinaryNotification {
    char agent_name[256]; // Fixed-size agent name (null-terminated)
    char message[1024]; // Fixed-size message (binary data, not null-terminated)
    uint32_t message_length; // Actual length of message data
    uint16_t xfer_id; // 16-bit postXfer ID (unique per postXfer call)
    uint32_t expected_completions; // Total write requests for this xfer_id

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

    /** @brief Set message with bounds checking and proper binary data handling */
    void
    setMessage(const std::string &msg) {
        message_length = std::min(msg.length(), sizeof(message));
        memcpy(message, msg.data(), message_length);
        // Zero out remaining space for consistency
        if (message_length < sizeof(message)) {
            memset(message + message_length, 0, sizeof(message) - message_length);
        }
    }

    /** @brief Get agent name as string */
    std::string
    getAgentName() const {
        return std::string(agent_name);
    }

    /** @brief Get message as string using stored length for proper binary data handling */
    std::string
    getMessage() const {
        return std::string(message, message_length);
    }
};

// Global XFER_ID management
namespace LibfabricUtils {
// Get next unique XFER_ID
uint16_t
getNextXferId();
// Get next 4-bit SEQ_ID
uint8_t
getNextSeqId();
// Reset SEQ_ID counter for new postXfer
void
resetSeqId();
} // namespace LibfabricUtils

// Utility functions
namespace LibfabricUtils {
// Device discovery with fallback to sockets
std::pair<std::string, std::vector<std::string>>
getAvailableNetworkDevices();
// String utilities
std::string
hexdump(const void *data);
} // namespace LibfabricUtils

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H
