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

// Wire-protocol version of the libfabric plugin, exchanged in the handshake. Bump it on any
// incompatible imm_data or control-message change. 1: 8-bit agent index in imm_data (sends
// "idx", no "ver"). 2: senders are resolved from the completion's source address.
#define NIXL_LIBFABRIC_PROTO_VERSION 2u

// The immediate data associated with an RDMA operation is 32 bits and is divided as follows:
// | 4-bit MSG TYPE flag | 8-bit RESERVED | 16-bit XFER_ID | 4-bit SEQ_ID |
// RESERVED held the 8-bit peer agent index in protocol version 1; it must now be zero.

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
// Peer-id handshake message. Sent once per (peer A, peer B) pair after
// connection setup. Carries assigned agent index, sender name, and optionally
// connection info. See libfabric_handshake.cpp for wire-format details.
#define NIXL_LIBFABRIC_MSG_HANDSHAKE 8

// Single-operation immediate data extraction (no intermediate shifts)
#define NIXL_GET_MSG_TYPE_FROM_IMM(data) ((data) & NIXL_MSG_TYPE_MASK)
#define NIXL_GET_IMM_RESERVED(data) (((data) >> NIXL_IMM_RESERVED_SHIFT) & NIXL_IMM_RESERVED_MASK)
#define NIXL_GET_XFER_ID_FROM_IMM(data) (((data) >> NIXL_XFER_ID_SHIFT) & NIXL_XFER_ID_MASK)
#define NIXL_GET_SEQ_ID_FROM_IMM(data) (((data) >> NIXL_SEQ_ID_SHIFT) & NIXL_SEQ_ID_MASK)

// Single-operation immediate data creation (minimal bit operations)
#define NIXL_MAKE_IMM_DATA(msg_type, xfer_id, seq_id)                    \
    (((uint64_t)(msg_type) & NIXL_MSG_TYPE_MASK) |                       \
     (((uint64_t)(xfer_id) & NIXL_XFER_ID_MASK) << NIXL_XFER_ID_SHIFT) | \
     (((uint64_t)(seq_id) & NIXL_SEQ_ID_MASK) << NIXL_SEQ_ID_SHIFT))

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
/**
 * @brief One libfabric domain as device discovery saw it.
 *
 * Everything the topology layer needs about a NIC, harvested during the single fi_getinfo() that
 * selects the provider. There used to be a second fi_getinfo() in
 * nixlLibfabricTopology::buildPcieToLibfabricMapping() purely to read these two fields back off the
 * same devices, with different hints -- which meant the two queries could disagree about which
 * domains exist, and needed a per-provider flag to paper over one provider's reaction to the
 * difference. One query cannot disagree with itself.
 */
struct nixlLibfabricDeviceInfo {
    /** @brief libfabric domain name, i.e. the name a rail is created on. */
    std::string domain_name;

    /**
     * @brief PCIe address in hwloc's "%x:%02x:%02x.%x" normalisation, or empty.
     *
     * Empty when libfabric reported no usable bus info -- verbs leaves bus_attr at FI_BUS_UNKNOWN.
     * Resolving that fallback needs hwloc, so it happens in the topology layer rather than here.
     */
    std::string pcie_address;

    /** @brief Link speed in bits/s from nic->link_attr, or 0 if libfabric reported none. */
    size_t link_speed = 0;
};

/**
 * @brief Selects a provider and returns its usable domains, with fallback to tcp/sockets.
 *
 * @return The chosen core provider name and one entry per domain that will become a rail, in rail
 *         order. {"none", {}} when nothing usable was found.
 */
std::pair<std::string, std::vector<nixlLibfabricDeviceInfo>>
getAvailableNetworkDevices();

/**
 * @brief The core provider part of a libfabric prov_name.
 *
 * A prov_name may name a stack of layered providers, e.g. "verbs;ofi_rxm". Everything before
 * the first ';' is the core provider. Returns @p prov_name unchanged when it is not layered.
 */
std::string
baseProviderName(const std::string &prov_name);

/** @brief prov_name → domain names, as gathered from fi_getinfo. */
using nixlProviderDeviceMap = std::unordered_map<std::string, std::vector<std::string>>;

/**
 * @brief Everything the plugin knows about one core libfabric provider.
 *
 * The provider analogue of libfabric_hmem.cpp's accelerator vendor table, and it exists for the
 * same reason: provider-specific behaviour used to be seven independent `provider == "..."`
 * comparisons spread across device discovery, rail setup, topology discovery and CQ progress, so
 * adding a provider meant finding all seven. Now it is one table entry.
 *
 * Table order is the preference order used by @ref getAvailableNetworkDevices.
 */
struct nixlLibfabricProviderInfo {
    /** @brief Core provider name, as it appears before any ';' in a libfabric prov_name. */
    const char *name;

    /** @brief Human-readable name for the discovery log line. */
    const char *description;

    /** @brief Extra capability bits OR'd into a rail's hints->caps. FI_RMA_EVENT for cxi. */
    uint64_t extra_caps;

    /** @brief hints->domain_attr->mr_mode for a rail of this provider. */
    uint64_t mr_mode;

    /** @brief hints->domain_attr->mr_key_size for a rail. 0 leaves the choice to the provider. */
    size_t mr_key_size;

    /**
     * @brief Whether a rail must name this provider in hints->fabric_attr->prov_name.
     *
     * Normally the domain name identifies one provider on its own. verbs is the exception: an
     * InfiniBand device is offered by both verbs and psm3 under the same domain name, and psm3 is
     * returned first, so without the hint a rail comes up on whichever libfabric happened to list
     * first. Asking for the core name is correct even though the RDM endpoint arrives as
     * "verbs;ofi_rxm" -- verbs has no native RDM endpoint, so libfabric layers ofi_rxm on top of
     * the name asked for.
     */
    bool needs_prov_name_hint;

    /**
     * @brief Whether rails are chosen by PCIe proximity to an accelerator.
     *
     * True for the RDMA providers (efa, cxi, verbs), which do hwloc topology discovery and
     * accelerator-to-NIC grouping. False for tcp and sockets, which reach every peer through the
     * host stack, so there is nothing for grouping to optimise.
     */
    bool needs_full_topology;

    /**
     * @brief Whether @ref providerNeedsAllRailProgress applies -- see that function for why.
     */
    bool needs_all_rail_progress;

    /**
     * @brief Whether to use only the first domain found rather than one rail per domain.
     *
     * True for tcp and sockets, where every domain reaches every peer and extra rails would only
     * multiply host-stack copies.
     */
    bool single_domain;

    /**
     * @brief Provider-specific domain selection, or nullptr for the generic rule.
     *
     * The generic rule takes every domain whose prov_name has this entry's @ref name as its core
     * provider, de-duplicated, in the order fi_getinfo returned them. Set this only for a provider
     * that needs more; see @ref selectVerbsDomains.
     */
    std::vector<std::string> (*select_domains)(const nixlProviderDeviceMap &);
};

/**
 * @brief The provider table, in preference order (highest first).
 */
const std::vector<const nixlLibfabricProviderInfo *> &
knownProviders();

/**
 * @brief Table entry for @p prov_name, or nullptr if it is not one the plugin knows.
 *
 * Accepts a layered prov_name ("verbs;ofi_rxm") as well as a core name ("verbs").
 */
const nixlLibfabricProviderInfo *
findProviderInfo(const std::string &prov_name);

/**
 * @brief Settings for a provider with no table entry: the plain-RDMA defaults that suited efa.
 *
 * Lets callers hold a non-null entry unconditionally instead of testing for one, so an unlisted
 * provider still gets a working rail rather than a special code path.
 */
const nixlLibfabricProviderInfo &
defaultProviderInfo();

/**
 * @brief A rail's position on the fabric, for pairing local rails with a peer's rails.
 *
 * Rails are addressed by index, and a transfer pairs local rail *k* with the peer's rail *k*. That is
 * only correct if both agents enumerate rails onto the same fabrics in the same order, which nothing
 * guarantees: libfabric domain names are host-local (one host calls a NIC `mlx5_0`, another calls the
 * same model `rocep153s0f0`), and the cabling need not follow the naming. When the assumption breaks,
 * rail *k* posts to a peer NIC on a different segment; the address is unreachable rather than
 * invalid, so the write retries -FI_EAGAIN forever instead of failing.
 *
 * This is the identity used to pair rails by where they actually are instead. It is derived from the
 * endpoint name the peer already sends during connection setup, so nothing new goes on the wire.
 */
struct nixlLibfabricFabricId {
    /** @brief Whether @ref addr holds a usable identity. False for address formats we cannot read. */
    bool valid = false;
    /** @brief IPv4 address of the rail's NIC, network byte order. */
    uint32_t addr = 0;
    /** @brief Netmask of the local interface owning @ref addr, or 0 if unknown (peer identities). */
    uint32_t mask = 0;
};

/**
 * @brief Extracts a @ref nixlLibfabricFabricId from a libfabric endpoint name.
 *
 * @param addr_format The provider's fi_info::addr_format. Only the sockaddr formats can be read;
 *                    anything else yields an invalid id, which makes pairing fall back to index
 *                    order. efa and cxi use opaque formats and are unaffected -- they also do not
 *                    need this, since their rails are enumerated consistently across hosts.
 * @param ep_name Endpoint name bytes as produced by fi_getname().
 * @param len Length of @p ep_name in bytes.
 */
nixlLibfabricFabricId
fabricIdFromEpName(uint32_t addr_format, const void *ep_name, size_t len);

/**
 * @brief Fills in @ref nixlLibfabricFabricId::mask by looking @p id's address up in getifaddrs().
 *
 * Only meaningful for a *local* rail: the netmask is what decides which peer addresses share this
 * rail's segment, and only the owning host knows it. Leaves mask at 0 when the address is not on any
 * local interface, which makes @ref sameFabric fall back to an exact-address comparison.
 */
void
resolveLocalFabricMask(nixlLibfabricFabricId &id);

/**
 * @brief Whether a peer rail at @p peer sits on the same fabric segment as local rail @p local.
 *
 * Uses the local rail's netmask, since that is the only one available. Both ids must be valid.
 */
bool
sameFabric(const nixlLibfabricFabricId &local, const nixlLibfabricFabricId &peer);

/**
 * @brief True when every rail must be progressed, not just the rails this agent posted on.
 *
 * A provider layered over a connection-oriented core (verbs, via ofi_rxm) establishes its
 * per-peer MSG connection lazily on the first send, and the *passive* side has to poll that
 * rail's completion queue for the connection to be accepted. An agent that has only received
 * has no rail of its own to progress, so restricting progress to rail 0 plus the rails with
 * outstanding local requests leaves any higher rail permanently unaccepted, and the peer that
 * posted on it never completes. Connectionless RDM providers (efa, cxi, tcp) need no accept.
 */
bool
providerNeedsAllRailProgress(const std::string &prov_name);

/**
 * @brief True when this provider does hwloc topology discovery and accelerator-to-NIC grouping.
 *
 * See @ref nixlLibfabricProviderInfo::needs_full_topology.
 */
bool
providerNeedsFullTopology(const std::string &prov_name);

/**
 * @brief Selects the verbs domains usable as nixl rails from a prov_name → domains map.
 *
 * The @ref nixlLibfabricProviderInfo::select_domains hook for verbs, which needs more filtering
 * than the generic rule gives:
 *
 *  - verbs has no native RDM endpoint, so its prov_name is the composite "verbs;ofi_rxm".
 *    Never compare it equal to "verbs".
 *  - fi_getinfo returns one entry per (domain x capability variant), so the same domain comes
 *    back several times and would otherwise become several rails on one NIC.
 *  - the ofi_rxd variant re-exposes the same NICs under their DGRAM "-dgram" domains.
 *
 * Exposed rather than kept file-local so it can be unit tested without a fabric.
 *
 * @return Unique domain names in sorted order, or empty if verbs has nothing usable.
 */
std::vector<std::string>
selectVerbsDomains(const nixlProviderDeviceMap &provider_device_map);

/**
 * @brief Every domain offered under exactly the prov_name @p name, de-duplicated, in fi_getinfo
 * order.
 *
 * The generic @ref nixlLibfabricProviderInfo::select_domains rule. Deliberately not sorted, unlike
 * selectVerbsDomains(): a rail's index has to be stable across runs and across peers, and for a
 * provider whose domains all arrive under one prov_name key the fi_getinfo order already is. Sorting
 * would be equally stable but would renumber the rails of every existing efa and cxi deployment.
 *
 * Exactly @p name, not every key with that base name, as before the provider table existed: a
 * layered variant such as "efa;ofi_rxd" reaches the same NICs through slower emulation, so merging
 * it would add duplicate rails, and since provider_device_map is unordered it would also make rail
 * order depend on hash order.
 *
 * Exposed rather than kept file-local so it can be unit tested without a fabric.
 */
std::vector<std::string>
selectDomainsGeneric(const nixlProviderDeviceMap &provider_device_map, const char *name);
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

/*
 * The accelerator-context mediator that used to live here is gone. It existed so the utils-layer
 * progress thread could reach a binder owned by the plugin without seeing its type; now the shim
 * (nfi_hmem.h) is in the utils layer itself and nfi_hmem_scope_push() needs no back-pointer to
 * anything. See the note in libfabric_backend.cpp for why the binder disappeared too.
 */


#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H
