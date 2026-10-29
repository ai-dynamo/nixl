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
#include <span>
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
// Sockets provider requires short timeout to maintain software progress during fi_cq_sread().
// Long timeouts block in poll(), preventing message processing. EFA uses hardware completions.
#define NIXL_LIBFABRIC_CQ_SREAD_TIMEOUT_MS 10
#define NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD (128 * 1024) // 128KB
#define LF_EP_NAME_MAX_LEN 56
#define NIXL_LIBFABRC_DEFAULT_POST_QUEUE_SIZE (32 * 1024) // 32K MPSC entries

// Number of consecutive WRITE descriptors batched onto one rail with FI_MORE before flushing.
#define NIXL_LIBFABRIC_FI_MORE_BATCH_SIZE 16

// Request pool configuration constants
#define NIXL_LIBFABRIC_CONTROL_REQUESTS_PER_RAIL 4096 // SEND/RECV operations (for notifications)
#define NIXL_LIBFABRIC_DATA_REQUESTS_PER_RAIL 1024 // WRITE/READ operations
#define NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE 8192 // For SEND/RECV notifications
#define NIXL_LIBFABRIC_RECV_POOL_SIZE 1024 // Number of recv requests to pre-post per rail

// Retry configuration constants
#define NIXL_LIBFABRIC_LOG_INTERVAL_ATTEMPTS 100 // Log every N attempts to avoid spam

// Handshake timeout (seconds) for waiting on peer's inbound handshake
#define NIXL_LIBFABRIC_HANDSHAKE_TIMEOUT_S 60

// Handshake SerDes tag names
constexpr const char *NIXL_HANDSHAKE_TAG_VER = "ver";
// Sent by protocol-version-1 peers in place of "ver"; only read to recognise such a peer.
constexpr const char *NIXL_HANDSHAKE_TAG_V1_IDX = "idx";
constexpr const char *NIXL_HANDSHAKE_TAG_NAME = "name";
constexpr const char *NIXL_HANDSHAKE_TAG_HAS_CONN = "has_conn";
constexpr const char *NIXL_HANDSHAKE_TAG_CONN = "conn";
// Optional: the write tag the receiver puts in imm_data of its writes to us.
constexpr const char *NIXL_HANDSHAKE_TAG_WRITE_TAG = "wtag";
// Optional, after "wtag": that tag's generation, carried in the SEQ_ID bits.
constexpr const char *NIXL_HANDSHAKE_TAG_WRITE_GEN = "wgen";

// Wire-protocol version of the libfabric plugin, exchanged in the handshake. Bump it on any
// incompatible imm_data or control-message change. 1: 8-bit agent index in imm_data (sends
// "idx", no "ver"). 2: senders are resolved from the completion's source address.
#define NIXL_LIBFABRIC_PROTO_VERSION 2u

// The immediate data associated with an RDMA operation is 32 bits and is divided as follows:
// | 4-bit MSG TYPE flag | 8-bit WRITE TAG | 16-bit XFER_ID | 4-bit SEQ_ID |
// WRITE TAG is the tag the target assigned us in its handshake ("wtag"), or 0 to have the target
// resolve the sender from the source address. Targets assign tags only when their provider reports
// no source address for remote writes (verbs;ofi_rxm). With a tag, SEQ_ID carries the tag's
// generation ("wgen"), so the target can drop writes from a previous holder of the tag.

// Optimized bit field constants (compile-time computed)
#define NIXL_MSG_TYPE_BITS 4
#define NIXL_IMM_RESERVED_BITS 8
#define NIXL_XFER_ID_BITS 16
#define NIXL_SEQ_ID_BITS 4

// Pre-computed shift amounts for better performance
#define NIXL_MSG_TYPE_SHIFT 0
#define NIXL_IMM_RESERVED_SHIFT 4
#define NIXL_XFER_ID_SHIFT 12
#define NIXL_SEQ_ID_SHIFT 28

// Pre-computed masks (compile-time constants)
#define NIXL_MSG_TYPE_MASK 0xFU // 0x0000000F (4 bits)
#define NIXL_IMM_RESERVED_MASK 0xFFU // 0x000000FF (8 bits)
#define NIXL_XFER_ID_MASK 0xFFFFU // 0x0000FFFF (16 bits)
#define NIXL_SEQ_ID_MASK 0xFU // 0x0000000F (4 bits)

// Message type constants
#define NIXL_LIBFABRIC_MSG_XFER_ERROR 1
#define NIXL_LIBFABRIC_MSG_NOTIFICTION 2
#define NIXL_LIBFABRIC_MSG_TRANSFER 4
// Peer-id handshake message. Sent when a connection is set up, and as a reply
// to a peer's fresh handshake. Carries protocol version, sender name, and
// optionally connection info. See libfabric_handshake.cpp for wire-format details.
#define NIXL_LIBFABRIC_MSG_HANDSHAKE 8

// Single-operation immediate data extraction (no intermediate shifts)
#define NIXL_GET_MSG_TYPE_FROM_IMM(data) ((data) & NIXL_MSG_TYPE_MASK)
#define NIXL_GET_IMM_RESERVED(data) (((data) >> NIXL_IMM_RESERVED_SHIFT) & NIXL_IMM_RESERVED_MASK)
#define NIXL_GET_WRITE_TAG_FROM_IMM(data) NIXL_GET_IMM_RESERVED(data)
#define NIXL_GET_XFER_ID_FROM_IMM(data) (((data) >> NIXL_XFER_ID_SHIFT) & NIXL_XFER_ID_MASK)
#define NIXL_GET_SEQ_ID_FROM_IMM(data) (((data) >> NIXL_SEQ_ID_SHIFT) & NIXL_SEQ_ID_MASK)
#define NIXL_GET_WRITE_GEN_FROM_IMM(data) NIXL_GET_SEQ_ID_FROM_IMM(data)

// Single-operation immediate data creation (minimal bit operations)
#define NIXL_MAKE_IMM_DATA(msg_type, xfer_id, seq_id)                    \
    (((uint64_t)(msg_type) & NIXL_MSG_TYPE_MASK) |                       \
     (((uint64_t)(xfer_id) & NIXL_XFER_ID_MASK) << NIXL_XFER_ID_SHIFT) | \
     (((uint64_t)(seq_id) & NIXL_SEQ_ID_MASK) << NIXL_SEQ_ID_SHIFT))

// Same, with the target-assigned write tag (0 = none) in the WRITE TAG field.
#define NIXL_MAKE_TAGGED_IMM_DATA(msg_type, write_tag, xfer_id, seq_id) \
    (NIXL_MAKE_IMM_DATA(msg_type, xfer_id, seq_id) |                    \
     (((uint64_t)(write_tag) & NIXL_IMM_RESERVED_MASK) << NIXL_IMM_RESERVED_SHIFT))

// Largest write tag; tags are 1-based, 0 means "no tag".
#define NIXL_LIBFABRIC_MAX_WRITE_TAG NIXL_IMM_RESERVED_MASK
// Largest write-tag generation; generations are 1-based, 0 means "not stamped".
#define NIXL_LIBFABRIC_MAX_WRITE_GEN NIXL_SEQ_ID_MASK

#define NIXL_LIBFABRIC_CQ_BATCH_SIZE 16

// Giga (decimal) constant
constexpr inline uint64_t NIXL_LIBFABRIC_GIGA = 1000ull * 1000ull * 1000ull;

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
 * @brief Payload of a NIXL_LIBFABRIC_MSG_XFER_ERROR control message (4 bytes)
 *
 * The transfer this refers to is identified by the xfer_id embedded in the immediate data, so the
 * payload only carries how many of the transfer's writes actually completed. The initiator's count
 * is authoritative: the target lowers its expected completion count to final_completions so that it
 * waits for exactly the writes that will arrive, and no longer waits for the ones that failed.
 */
struct XferErrorPayload {
    uint32_t final_completions; // Writes that completed on the initiator and will reach the target
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

// helper type for hashing integer pair
template<typename T, typename U = T> struct pair_hash {
    inline size_t
    operator()(const std::pair<T, U> &int_pair) const {
        return std::hash<T>()(int_pair.first) ^ std::hash<U>()(int_pair.second);
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
/** @brief One libfabric domain as reported by the fi_getinfo() that selects the provider. */
struct nixlLibfabricDeviceInfo {
    /** @brief libfabric domain name, the name a rail is created on. */
    std::string domain_name;
    /**
     * @brief PCIe address in hwloc's "%x:%02x:%02x.%x" format, or empty if libfabric reported none
     * (verbs leaves bus_attr at FI_BUS_UNKNOWN; the topology layer resolves it through hwloc).
     */
    std::string pcie_address;
    /** @brief Link speed in bits/s, or 0 if libfabric reported none. */
    size_t link_speed = 0;
};

/**
 * @brief Selects a provider and returns its usable domains, with fallback to tcp/sockets.
 * @return Core provider name and one entry per rail, in rail order; {"none", {}} if none found.
 */
std::pair<std::string, std::vector<nixlLibfabricDeviceInfo>>
getAvailableNetworkDevices();

/** @brief Core provider of a possibly layered prov_name: "verbs;ofi_rxm" -> "verbs". */
std::string
baseProviderName(const std::string &prov_name);

/** @brief prov_name -> domain names, as gathered from fi_getinfo. */
using nixlProviderDeviceMap = std::unordered_map<std::string, std::vector<std::string>>;

/**
 * @brief Provider-specific settings for one core libfabric provider.
 *
 * Table order (see @ref knownProviders) is the preference order.
 */
struct nixlLibfabricProviderInfo {
    /** @brief Core provider name, as it appears before any ';' in a prov_name. */
    const char *name;
    /** @brief Human-readable name for the discovery log line. */
    const char *description;
    /** @brief Extra capability bits added to a rail's hints->caps. */
    uint64_t extra_caps;
    /** @brief hints->domain_attr->mr_mode for a rail. */
    uint64_t mr_mode;
    /** @brief hints->domain_attr->mr_key_size for a rail; 0 lets the provider choose. */
    size_t mr_key_size;
    /**
     * @brief Whether a rail names this provider in hints->fabric_attr->prov_name. Needed for verbs,
     * whose domains psm3 also offers under the same names.
     */
    bool needs_prov_name_hint;
    /** @brief Whether to do hwloc topology discovery and accelerator-to-NIC grouping. */
    bool needs_full_topology;
    /** @brief See @ref providerNeedsAllRailProgress. */
    bool needs_all_rail_progress;
    /** @brief See @ref providerNeedsWriteTag. */
    bool needs_write_tag;
    /** @brief Use only the first domain: every domain reaches every peer (tcp, sockets). */
    bool single_domain;
    /** @brief Provider-specific domain selection, or nullptr for @ref selectDomainsGeneric. */
    std::vector<std::string> (*select_domains)(const nixlProviderDeviceMap &);
};

/** @brief The provider table, in preference order (highest first). */
std::span<const nixlLibfabricProviderInfo *const>
knownProviders();

/** @brief Table entry for a core or layered @p prov_name, or nullptr if unknown. */
const nixlLibfabricProviderInfo *
findProviderInfo(const std::string &prov_name);

/** @brief Settings used for a provider with no table entry (the plain-RDMA defaults of efa). */
const nixlLibfabricProviderInfo &
defaultProviderInfo();

/**
 * @brief True when every rail must be progressed, not just the rails this agent posted on.
 *
 * A provider layered over a connection-oriented core (verbs via ofi_rxm) connects lazily, and the
 * passive side must poll a rail's CQ for the peer's connection on that rail to be accepted.
 */
bool
providerNeedsAllRailProgress(const std::string &prov_name);

/**
 * @brief True when a remote-write completion does not identify its sender by source address.
 *
 * ofi_rxm leaves src_addr stale or FI_ADDR_NOTAVAIL for remote writes, so peers carry a
 * target-assigned tag in imm_data instead.
 */
bool
providerNeedsWriteTag(const std::string &prov_name);

/** @brief See @ref nixlLibfabricProviderInfo::needs_full_topology. */
bool
providerNeedsFullTopology(const std::string &prov_name);

/**
 * @brief The verbs domains usable as rails.
 *
 * Keeps only "verbs;ofi_rxm" entries (verbs has no native RDM endpoint), drops the "-xrc" and
 * "-dgram" domain variants, and de-duplicates the per-capability repeats of each domain.
 *
 * @return Unique domain names, sorted.
 */
std::vector<std::string>
selectVerbsDomains(const nixlProviderDeviceMap &provider_device_map);

/**
 * @brief Domains listed under exactly the prov_name @p name, de-duplicated, in fi_getinfo order.
 *
 * Layered variants such as "efa;ofi_rxd" are not merged in: they reach the same NICs.
 */
std::vector<std::string>
selectDomainsGeneric(const nixlProviderDeviceMap &provider_device_map, const char *name);

/**
 * @brief The first knownProviders() entry with usable domains, and those domains (one for a
 *        single_domain provider).
 * @return Core provider name and its domains in rail order; {"none", {}} if none.
 */
std::pair<std::string, std::vector<std::string>>
selectProvider(const nixlProviderDeviceMap &provider_device_map);

/**
 * @brief Selects a provider from an fi_getinfo() list (see @ref selectProvider) and returns its
 *        domains with the PCIe address and link speed that the selected provider reported.
 * @return Core provider name and one entry per rail, in rail order; {"none", {}} if none found.
 */
std::pair<std::string, std::vector<nixlLibfabricDeviceInfo>>
selectNetworkDevices(const struct fi_info *info);
// String utilities
std::string
hexdump(const void *data, size_t size);

/** @brief Converts rail id vector to string. */
extern std::string
railIdsToString(const std::vector<size_t> &rail_ids);

/** @brief Retrieves the maximum NUMA node id. */
extern bool
getMaxNumaNode(int &node_id);

/** @brief Retrieves the number of configured NUMA nodes. */
extern bool
getNumConfiguredNumaNodes(int &node_count);
} // namespace LibfabricUtils

// Configuration helper functions
namespace LibfabricUtils {
/**
 * @brief Loads string from custom plugin parameters.
 * @note Can override from env var with name NIXL_LIBFABRIC_<upper-case key>.
 * @param custom_param The backend parameters.
 * @param key The key name.
 * @param[out] value The resulting value (valid only if call succeeds).
 * @return Result status.
 */
extern nixl_status_t
getCustomStringParam(const nixl_b_params_t &custom_params,
                     const std::string &key,
                     std::string &value);

/**
 * @brief Loads integer from custom plugin parameters.
 * @note Can override from env var with name NIXL_LIBFABRIC_<upper-case key>.
 * @param custom_param The backend parameters.
 * @param key The key name.
 * @param[out] value The resulting value (valid only if call succeeds).
 * @return Result status.
 */
extern nixl_status_t
getCustomIntParam(const nixl_b_params_t &custom_params, const std::string &key, size_t &value);
} // namespace LibfabricUtils

// CUDA context workaround temporary API for exposing to progress thread
namespace LibfabricUtils {

/**
 * @brief Mediator class for abstracting engine internals, while allowing consumers (e.g. progress
 * thread) to have access to the API without having an engine pointer.
 */
class nixlLibfaricCudaCtxMediator {
public:
    virtual ~nixlLibfaricCudaCtxMediator() {}

    virtual nixl_status_t
    cudaSetCtx(bool &use_cuda_addr_wa) = 0;

protected:
    nixlLibfaricCudaCtxMediator() {}
};

extern void
setCudaCtxMediator(std::unique_ptr<nixlLibfaricCudaCtxMediator> &&mediator);

extern void
clearCudaCtxMediator();

extern nixl_status_t
cudaSetCtx(bool &use_cuda_addr_wa);
} // namespace LibfabricUtils

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H
