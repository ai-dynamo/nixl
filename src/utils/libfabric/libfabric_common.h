/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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
#include <cassert>

#include "nixl.h"

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>


// Libfabric configuration constants
#define NIXL_LIBFABRIC_DEFAULT_CONTROL_RAILS 1

// Sockets provider requires short timeout to maintain software progress during fi_cq_sread().
// Long timeouts block in poll(), preventing message processing. EFA uses hardware completions.
#define NIXL_LIBFABRIC_CQ_SREAD_TIMEOUT_MS 10
#define NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD (128 * 1024) // 128KB
#define LF_EP_NAME_MAX_LEN 56

// Request pool configuration constants
#define NIXL_LIBFABRIC_CONTROL_REQUESTS_PER_RAIL 4096 // SEND/RECV operations (1:1 with buffers)
#define NIXL_LIBFABRIC_DATA_REQUESTS_PER_RAIL 1024 // WRITE/read operations (no buffers)
#define NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE 8192
#define NIXL_LIBFABRIC_RECV_POOL_SIZE 1024 // Number of recv requests to pre-post per rail

// Retry configuration constants
#define NIXL_LIBFABRIC_MAX_RETRIES 10
#define NIXL_LIBFABRIC_EFA_RETRY_DELAY_US 100
#define NIXL_LIBFABRIC_DEFAULT_RETRY_DELAY_US 1000
#define NIXL_LIBFABRIC_BASE_RETRY_DELAY_US 1000 // Base 1ms delay between retries
#define NIXL_LIBFABRIC_MAX_RETRY_DELAY_US 100000 // Max 100ms delay between retries
#define NIXL_LIBFABRIC_LOG_INTERVAL_ATTEMPTS 100 // Log every N attempts to avoid spam

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
 * @brief Notification header for all fragments (10 bytes)
 *
 * This is present in every fragment and contains only the essential
 * fields needed for fragment identification and reassembly.
 */
struct BinaryNotificationHeader {
    uint16_t notif_xfer_id; // Transfer ID for matching notifications
    uint16_t notif_seq_id; // Fragment index (0, 1, 2...)
    uint16_t notif_seq_len; // Total number of fragments
    uint32_t payload_length; // Message bytes of this fragment
} __attribute__((packed));

/**
 * @brief Metadata for fragment 0 only (10 bytes)
 *
 * This contains metadata that is constant across all fragments,
 * so we only send it once in the first fragment.
 */
struct BinaryNotificationMetadata {
    uint32_t total_payload_length; // Total message bytes across all fragments
    uint32_t expected_completions; // Expected RDMA write completions
    uint16_t agent_name_length; // Actual length of agent_name
} __attribute__((packed));

/**
 * @brief Binary notification with variable-length encoding and fragmentation support
 *
 * The notification payload consists of agent_name + message, which is treated as a single
 * combined payload that can be fragmented across multiple network messages.
 *
 * Fragment 0 layout: [Header:10B] [Metadata:10B] [combined_payload_chunk:variable]
 * Fragment 1+ layout: [Header:10B] [combined_payload_chunk:variable]
 *
 * After reassembly, use metadata.agent_name_length to split the combined payload:
 *   - agent_name = combined_payload.substr(0, agent_name_length)
 *   - message = combined_payload.substr(agent_name_length)
 *
 * @note The __attribute__((packed)) ensures consistent byte layout across platforms,
 *       preventing padding-related data corruption during network serialization.
 */
class BinaryNotification {
private:
    BinaryNotificationHeader header_;
    BinaryNotificationMetadata metadata_; // Only valid for seq_id=0
    std::string payload_; // Chunk of (agent_name + message) combined payload

public:
    /** @brief Maximum fragment size for control messages */
    static constexpr size_t MAX_FRAGMENT_SIZE = NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE;

    /** @brief Constructor */
    BinaryNotification() {
        memset(&header_, 0, sizeof(header_));
        memset(&metadata_, 0, sizeof(metadata_));
    }

    /** @brief Set header fields */
    void
    setHeader(const BinaryNotificationHeader &header) {
        header_ = header;
    }

    /**
     * @brief Set metadata (only valid for fragment 0)
     * @param total_payload_length Total length of combined payload across all fragments
     * @param expected_completions Expected RDMA write completions
     * @param agent_name_length Length of agent_name within combined payload
     * @pre header_.notif_seq_id must be 0
     */
    void
    setMetadata(uint32_t total_payload_length,
                uint32_t expected_completions,
                uint16_t agent_name_length) {
        assert(header_.notif_seq_id == 0 && "setMetadata() can only be called for fragment 0");
        metadata_.total_payload_length = total_payload_length;
        metadata_.expected_completions = expected_completions;
        metadata_.agent_name_length = agent_name_length;
    }

    /**
     * @brief Set payload chunk for this fragment using move semantics
     * @param payload Chunk of (agent_name + message) combined payload (passed by value for move)
     * @note Also updates header_.payload_length to match the chunk size
     */
    void
    setPayload(std::string payload) {
        payload_ = std::move(payload);
        header_.payload_length = static_cast<uint32_t>(payload_.length());
    }

    /** @brief Get header (valid for all fragments) */
    const BinaryNotificationHeader &
    getHeader() const {
        return header_;
    }

    /**
     * @brief Get metadata (only valid for fragment 0)
     * @return Reference to metadata
     * @pre header_.notif_seq_id must be 0
     */
    const BinaryNotificationMetadata &
    getMetadata() const {
        assert(header_.notif_seq_id == 0 && "getMetadata() can only be called for fragment 0");
        return metadata_;
    }

    /** @brief Get payload chunk for this fragment */
    const std::string &
    getPayload() const {
        return payload_;
    }

    /** @brief Serialize to buffer for transmission */
    size_t
    serialize(void *buffer) const {
        char *ptr = static_cast<char *>(buffer);
        size_t offset = 0;

        // Write header (always present)
        memcpy(ptr + offset, &header_, sizeof(header_));
        offset += sizeof(header_);

        if (header_.notif_seq_id == 0) {
            // Fragment 0: write metadata
            memcpy(ptr + offset, &metadata_, sizeof(metadata_));
            offset += sizeof(metadata_);
        }

        // Write payload chunk (single memcpy)
        memcpy(ptr + offset, payload_.data(), payload_.size());
        offset += payload_.size();

        return offset;
    }

    /** @brief Deserialize from buffer */
    static void
    deserialize(const void *buffer, size_t size, BinaryNotification &notif_out) {
        const char *ptr = static_cast<const char *>(buffer);
        size_t offset = 0;

        // Read header
        memcpy(&notif_out.header_, ptr + offset, sizeof(notif_out.header_));
        offset += sizeof(notif_out.header_);

        if (notif_out.header_.notif_seq_id == 0) {
            // Fragment 0: read metadata
            memcpy(&notif_out.metadata_, ptr + offset, sizeof(notif_out.metadata_));
            offset += sizeof(notif_out.metadata_);
        }

        // Read payload chunk
        size_t remaining = size - offset;
        notif_out.payload_.assign(ptr + offset, remaining);
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
hexdump(const void *data, size_t size);
} // namespace LibfabricUtils

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H
