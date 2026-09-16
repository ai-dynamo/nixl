/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SERVICE_DATA_H
#define NIXL_SERVICE_DATA_H

#include "nixl.h"
#include "common/nixl_log.h"
#include "nixl_service_types.h"
#include "marshal/marshal_backend.h"
#include "spsc_queue.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <bit>
#include <bitset>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <variant>

#include <cuda_runtime.h>

// Both peers of a transfer must agree on this (see marshalLayoutFingerprint), so the
// service fixes it rather than exposing it through nixlServiceAgentConfig.
constexpr size_t default_chunked_payload_size = 64UL * 1024 * 1024;
constexpr size_t max_descriptor_size = 64UL * 1024 * 1024 * 1024;
static_assert(max_descriptor_size % default_chunked_payload_size == 0);
constexpr size_t max_chunks_per_desc = max_descriptor_size / default_chunked_payload_size;

namespace nixlService {
size_t
recommendServiceMemSize(const nixl_marshal_config_t &mode,
                        uint32_t max_concurrent_transfers,
                        size_t chunked_payload_size);
} // namespace nixlService

namespace CompressedObjectLayout {
constexpr size_t header_entry_size = sizeof(uint64_t);
constexpr size_t header_size = max_chunks_per_desc * header_entry_size;
} // namespace CompressedObjectLayout

/**
 * @class cudaStream
 * @brief  RAII wrapper around a non-blocking cudaStream_t.
 */
class cudaStream {
public:
    cudaStream() {
        const cudaError_t status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamCreateWithFlags failed: ") +
                                     cudaGetErrorString(status));
        }
    }

    ~cudaStream() noexcept {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    cudaStream(const cudaStream &) = delete;
    cudaStream &
    operator=(const cudaStream &) = delete;

    cudaStream(cudaStream &&other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    cudaStream &
    operator=(cudaStream &&other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t
    get() const noexcept {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

/**
 * @struct slotT
 * @brief  Describes an allocated slot handed out by a slotPool.
 *
 * @details
 * @c stream is a non-owning handle.  Ownership of the underlying cudaStream_t
 * lives in the issuing @c slotPool (see @c slotPool::streams_), and the handle
 * remains valid for the lifetime of that pool.
 */
class slotPool;

struct slotT {
    slotPool *pool = nullptr;
    uintptr_t baseAddr;
    /* Size fields:
     * slotSize is the total size for the slot in the service memory (slotPool),
     * pointed to by baseAddr.
     * chunkSize is the corresponding size in the user buffer. Each chunk is processed to a slot and
     * vice versa. workspaceSize is included in slotSize (0 <= *workspaceSize < slotSize), needed
     * for certain marshals e.g. compression.
     */
    size_t slotSize;
    std::optional<size_t> workspaceSize;
    size_t chunkSize;
    cudaStream_t stream;
    nixl_mem_t type;

    slotT() = default;
    slotT(slotPool *slot_pool,
          uintptr_t base_addr,
          size_t slot_size,
          cudaStream_t stream,
          nixl_mem_t type,
          size_t chunk_size,
          std::optional<size_t> workspace_size = std::nullopt);

    [[nodiscard]] nixlBasicDesc
    toDesc(size_t len) const noexcept;

    [[nodiscard]] nixlMarshal::runtimeBuffer
    toRuntimeBuffer() const;

    [[nodiscard]] nixlMarshal::process_slot_input_options_t
    getProcessSlotInputOptions() const;

private:
    int deviceId_;
};

enum class remote_slot_state_t { NOT_ALLOCATED, BUSY, FREE };

enum class local_slot_state_t {
    BUSY_MARSHAL,
    READY_TO_SEND,
    BUSY_NIXL,
    BUSY_NIXL_READ,
    BUSY_NIXL_HEADER,
    FREE
};

enum class compressed_header_state_t { NOT_STARTED, BUSY_NIXL, DONE };

enum class nixl_service_xfer_state_t {
    PRE_START,
    WAIT_CTS,
    IN_PROGRESS,
    // READ-only: mid-transfer cancellation is in progress (draining in-flight work before
    // freeing state). Ordered after IN_PROGRESS and before DONE so the existing
    // `state >= IN_PROGRESS` checks used by the WRITE progress engine continue to hold for
    // a cancelling/failed READ request (WRITE never enters these two states).
    CANCELLING,
    DONE,
    // READ-only: the request ended in a terminal error (e.g. an RNAK admission rejection).
    FAILED
};

// TODO-Yoav: tie in a more robust way (template <nixl_marshal_mode_t> class Backend)
enum class nixl_marshal_mode_t { STAGING, COMPRESSION, DELTA, DIRECT, NUM_BACKENDS };

inline nixl_marshal_mode_t
marshalBackendFromConfig(const nixl_marshal_config_t &cfg) noexcept {
    if (std::holds_alternative<nixlMarshalCompressConfig>(cfg)) {
        return nixl_marshal_mode_t::COMPRESSION;
    }
    if (std::holds_alternative<nixlMarshalStagingConfig>(cfg)) {
        return nixl_marshal_mode_t::STAGING;
    }
    if (std::holds_alternative<nixlMarshalDeltaConfig>(cfg)) {
        return nixl_marshal_mode_t::DELTA;
    }
    return nixl_marshal_mode_t::DIRECT;
}

static constexpr size_t slots_per_xfer = 2;

class chunkIteratorH {
protected:
    const nixl_xfer_dlist_t descList;
    const size_t chunkSize;
    const size_t totalChunks;
    size_t currentChunkGlobal;
    size_t currentChunkLocal;
    size_t currentDesc;
    nixl_marshal_mode_t backend;

public:
    chunkIteratorH(const nixl_xfer_dlist_t &desc_list,
                   size_t chunk_size,
                   size_t total_chunks,
                   nixl_marshal_mode_t backend)
        : descList(desc_list),
          chunkSize(chunk_size),
          totalChunks(total_chunks),
          currentChunkGlobal(0),
          currentChunkLocal(0),
          currentDesc(0),
          backend(backend) {}

    [[nodiscard]] std::byte *
    get() noexcept {
        if (currentChunkGlobal >= totalChunks) {
            return nullptr;
        }
        return reinterpret_cast<std::byte *>(descList[currentDesc].addr +
                                             currentChunkLocal * chunkSize);
    }

    bool
    operator++(int) noexcept {
        currentChunkGlobal++;
        if (currentChunkGlobal >= totalChunks) {
            return false;
        }
        currentChunkLocal++;
        if (currentChunkLocal >= (descList[currentDesc].len + chunkSize - 1) / chunkSize) {
            currentChunkLocal = 0;
            currentDesc++;
        }
        NIXL_ASSERT(currentDesc < static_cast<size_t>(descList.descCount()));
        return true;
    }

    [[nodiscard]] size_t
    currentChunkSize() noexcept {
        if (currentChunkGlobal >= totalChunks) {
            return 0;
        }
        return std::min(
            chunkSize,
            static_cast<size_t>(descList[currentDesc].len - currentChunkLocal * chunkSize));
    }

    [[nodiscard]] size_t
    getTotalChunks() const noexcept {
        return totalChunks;
    }

    [[nodiscard]] size_t
    getCurrentChunkLocal() const noexcept {
        return currentChunkLocal;
    }

    [[nodiscard]] size_t
    getCurrentDesc() const noexcept {
        return currentDesc;
    }

    [[nodiscard]] nixl_marshal_mode_t
    getBackend() const noexcept {
        return backend;
    }

    [[nodiscard]] nixl_mem_t
    getMemType() const noexcept {
        return descList.getType();
    }

    // Absolute byte offset of (desc_index, chunk_index) from the start of this iterator's
    // descriptor list, using this iterator's own chunk size. Useful for recovering the byte
    // offset of a chunk already visited (e.g. from a previously-captured descIndex/
    // chunkIndex pair), without needing to re-walk the iterator itself.
    [[nodiscard]] size_t
    getByteOffset(size_t desc_index, size_t chunk_index) const noexcept {
        size_t offset = 0;
        for (size_t i = 0; i < desc_index; ++i) {
            offset += descList[i].len;
        }
        return offset + chunk_index * chunkSize;
    }
};

struct postedNotifPayload {
    size_t xferId;
    size_t slotIndex;
    size_t originalSize; // size of data pulled from user buffer.
    size_t wireSize; // size of the marshalled data transferred through the slot.
    size_t descIndex; // index of the desc in the desc list.
    size_t chunkIndex; // index of the chunk in the desc.
    std::string metadata; // metadata produced by the marshal layer.
    nixl_marshal_mode_t backend;

    postedNotifPayload(size_t xfer_id,
                       size_t slot_index,
                       size_t original_size,
                       size_t wire_size,
                       size_t desc_index,
                       size_t chunk_index,
                       std::string md,
                       nixl_marshal_mode_t backend);
    explicit postedNotifPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

struct slotWorkItem {
    slotT slot;
    local_slot_state_t state = local_slot_state_t::FREE;
    size_t slotIndex = 0;
    std::optional<postedNotifPayload> postedNotif = std::nullopt;

    slotWorkItem() = default;
    slotWorkItem(slotT slot, size_t slot_index);
};

struct nixlServiceXferReqH {
    struct nonDirectDataH {
        const std::string remoteAgent;
        const std::string serializedDstDescList;
        const size_t xferId;
        nixl_service_xfer_state_t state;
        chunkIteratorH chunkIterator;
        std::array<std::optional<slotWorkItem>, slots_per_xfer> localSlots;
        std::array<std::unique_ptr<nixlMarshal::outbound_async_handle_t>, slots_per_xfer>
            outboundAsyncHandles;
        // Remote slots are not const since they are determined by the remote agent and
        // passed in the CTS notification.
        std::array<nixlBasicDesc, slots_per_xfer> remoteSlotDescriptors;
        std::array<nixlMarshal::mem_space_t, slots_per_xfer> remoteSlotMemTypes;
        std::array<remote_slot_state_t, slots_per_xfer> remoteSlotStates;
        std::array<nixlXferReqH *, slots_per_xfer> nixlXferReqs;
        size_t rslotsReceived = 0;
        std::string notifMsg = "";
        // READ-serving only (always 0, unused for WRITE): the generation this agent most
        // recently sent for each remote slot, bumped by trySend() right before dispatching
        // a new fill. handleRRSlot() compares an incoming RRSLOT's generation against this
        // to tell a stale/duplicate ack for an earlier fill of the same slot apart from the
        // ack for the fill currently in flight.
        std::array<uint64_t, slots_per_xfer> remoteSlotGenerations{};

        struct asymmetricPreTransferState {
            nixl_xfer_dlist_t remoteDstList;
            size_t currentDescIndex = 0;
            size_t chunksInCurrentDesc = 0;
            size_t descCount = 0;
            std::array<uint64_t, max_chunks_per_desc> compressedSizes{};
            std::array<std::optional<size_t>, slots_per_xfer> slotChunkIndices{};
            size_t nextPayloadOffset = CompressedObjectLayout::header_size;
            size_t nextChunkToWrite = 0;
            size_t completedPayloadWrites = 0;
            bool fillBlocked = false;
            nixlXferReqH *headerXferReq = nullptr;
            compressed_header_state_t headerState = compressed_header_state_t::NOT_STARTED;
            // This field preserves the result to return once all in-flight work has drained
            // (maybe with errors). It differs from the "state" field as that one tracks execution.
            nixl_status_t completionStatus = NIXL_IN_PROG;

            asymmetricPreTransferState(nixl_xfer_dlist_t remote_dst_list,
                                       size_t chunks_in_current_desc)
                : remoteDstList(std::move(remote_dst_list)),
                  chunksInCurrentDesc(chunks_in_current_desc),
                  descCount(static_cast<size_t>(remoteDstList.descCount())) {}
        };

        std::optional<asymmetricPreTransferState> asymmetricPreTransfer;

        // Bitmask of free source-slot indexes (1 = free).
        std::bitset<slots_per_xfer> freeSlotIndexes = ~std::bitset<slots_per_xfer>{0};
    };

    // The direct (<= direct_desc_threshold) sub-transfer's handle for a WRITE, a direct-only
    // READ, or a fully-direct request. For a READ with a marshal part (mixed or
    // marshal-only), the direct child instead lives on the receive context's own
    // directChild (see inboundXferReqH) so progressService() can reach it, and this stays
    // null - req_hndl is then just a thin token carrying readReceiveXferId.
    nixlXferReqH *xferReq;
    std::optional<nonDirectDataH> nonDirectData;
    nixl_marshal_opt_args_t marshalOptArgs;
    // Direction this handle was created for. NIXL_WRITE for all handles today; READ handles
    // set this so postXferReq/getXferStatus/releaseXferReq can branch on direction without
    // relying on the (direction-agnostic) marshalOptArgs alternative.
    nixl_xfer_op_t op = NIXL_WRITE;
    const nixl_marshal_phase_t phase;
    // For a READ handle (op == NIXL_READ) with a marshal part only: the key into
    // nixlServiceAgentData::readReceiveReqs_ identifying this READ's receive context. Empty
    // for a WRITE handle, or for a direct-only READ (no marshal part, hence no context).
    // remoteAgent is deliberately not duplicated here - it is available from the receive
    // context itself (inboundXferReqH::remoteAgent) once looked up by this id.
    std::optional<size_t> readReceiveXferId;
    std::optional<nixl_xfer_telem_t> telemetry;

    nixlServiceXferReqH(
        const nixl_xfer_dlist_t &src_desc_list,
        const std::string &serialized_dst_desc_list,
        const std::string &remote_agent,
        size_t chunk_size,
        size_t xfer_id,
        size_t total_chunks,
        const nixl_marshal_opt_args_t &marshal_opt_args,
        nixl_marshal_mode_t backend,
        nixl_marshal_phase_t marshal_phase = nixl_marshal_phase_t::PRE_AND_POST_TRANSFER);

    explicit nixlServiceXferReqH(
        nixl_marshal_phase_t marshal_phase = nixl_marshal_phase_t::PRE_AND_POST_TRANSFER)
        : xferReq(nullptr),
          marshalOptArgs(nixlMarshalDirectOptArgs{}),
          phase(marshal_phase) {}
};

struct nixlMarshalDeltaReceiverRefArgs {
    std::byte *ref = nullptr;
    nixl_mem_t memType{};
    size_t elementSize = 0;
};

// Sender-side counterpart of nixlMarshalDeltaReceiverRefArgs: for a marshalled READ, the
// peer is the one encoding, so RREQ ships the peer's own delta reference (this struct)
// rather than the initiator's receiver reference.
struct nixlMarshalDeltaSenderRefArgs {
    std::byte *ref = nullptr;
    nixl_mem_t memType{};
    size_t elementSize = 0;
};

struct inboundXferReqH {
    const std::string remoteAgent;
    const nixl_xfer_dlist_t dstList;
    const size_t xferId;
    nixl_service_xfer_state_t state;
    const nixl_marshal_phase_t phase;
    bool markedForDeletion = false;
    std::array<slotWorkItem, slots_per_xfer> localSlots;
    std::array<std::unique_ptr<nixlMarshal::inbound_async_handle_t>, slots_per_xfer> asyncHandles;
    std::optional<nixlMarshalDeltaReceiverRefArgs> receiverDeltaRef;

    // --- READ-receive-only bookkeeping (userInitiated == true); left at their defaults for
    //     a WRITE-inbound request built by handleRTS ---

    // Serialized SOURCE dlist (the peer's address space), captured at createXferReq() time
    // and sent by genRREQ() when postXferReq() is later called by the user.
    const std::string serializedSrcList = "";
    size_t totalChunks = 0;
    size_t decodedChunks = 0;
    // The user notification for this READ, delivered to the peer once the whole logical
    // request (direct + marshal parts) completes. std::optional (not a bare string) so a
    // present-but-empty notification can be told apart from "no notification at all".
    std::optional<nixl_blob_t> notif;
    // True only for a request built by createXferReq(NIXL_READ, ...) on the initiator (a
    // *receive* context the user holds a handle to); false for a WRITE-inbound request built
    // by handleRTS (which the user never sees a handle for).
    bool userInitiated = false;
    // The direct (<= direct_desc_threshold) sub-transfer's handle for a mixed READ, moved
    // here (rather than living on the outer nixlServiceXferReqH, which stays a thin token
    // for any READ with a marshal part) so progressService()'s end-of-tick scan can poll
    // it - see progressReadReceive(). Null for a marshal-only READ.
    nixlXferReqH *directChild = nullptr;
    // Completion of directChild above and the marshal (slot-based) sub-transfer
    // respectively; the logical READ is done only when both are. directDone is derived
    // from directChild at construction (null => true, as there is nothing to wait for) and,
    // for a mixed READ, later updated by progressReadReceive() as directChild's own status
    // is observed.
    bool directDone = true;
    bool marshalDone = false;
    // Failure status returned by getXferStatus() when state is FAILED. This is separate from
    // state, which tracks request execution and cleanup.
    nixl_status_t terminalStatus = NIXL_IN_PROG;
    // True once handleRAbortAck confirms the peer has fully quiesced this READ (stopped
    // touching this agent's receive slots) after an RABORT. Distinguishes "still waiting for
    // the peer to quiesce" from "peer quiesced, now draining this agent's own in-flight
    // decodes" - both of which are state == CANCELLING; progressService()'s end-of-tick
    // scan only frees this request once both remoteQuiesced and every local slot is idle.
    bool remoteQuiesced = false;
    // Per-slot generation, incremented each time a slot is (re)filled, so a stale/duplicate
    // RRSLOT ack cannot free a slot that has since been reused for a later chunk.
    std::array<uint64_t, slots_per_xfer> slotGenerations{};
    // Per-slot expected decoded size, set from rPostedPayload::originalSize when a decode is
    // submitted; compared against the actual decoded size on completion so a corrupted or
    // mismatched chunk fails the logical request instead of landing a wrong-sized write.
    std::array<size_t, slots_per_xfer> slotExpectedSizes{};
    nixlServiceXferReqH *parent = nullptr;

    struct asymmetricPostTransferState {
        nixl_xfer_dlist_t remoteSrcList;
        size_t currentDescIndex = 0;
        size_t chunksInCurrentDesc = 0;
        size_t descCount = 0;
        std::array<uint64_t, max_chunks_per_desc> compressedSizes{};
        std::array<size_t, max_chunks_per_desc> payloadOffsets{};
        size_t nextChunkToRead = 0;
        std::array<std::optional<size_t>, slots_per_xfer> slotChunkIndices{};
        std::array<nixlXferReqH *, slots_per_xfer> nixlXferReqs{};
        nixlXferReqH *headerXferReq = nullptr;
        compressed_header_state_t headerState = compressed_header_state_t::NOT_STARTED;

        asymmetricPostTransferState(nixl_xfer_dlist_t remote_src_list,
                                    size_t chunks_in_current_desc)
            : remoteSrcList(std::move(remote_src_list)),
              chunksInCurrentDesc(chunks_in_current_desc),
              descCount(static_cast<size_t>(remoteSrcList.descCount())) {}
    };

    std::optional<asymmetricPostTransferState> asymmetricPostTransfer;

    // WRITE-inbound constructor: built by handleRTS() when a peer pushes data to us.
    inboundXferReqH(
        const std::string &remote_agent,
        nixl_xfer_dlist_t &&dst_list,
        const std::array<slotWorkItem, slots_per_xfer> &local_slots,
        size_t xfer_id,
        std::optional<nixlMarshalDeltaReceiverRefArgs> receiver_delta_ref = std::nullopt);

    // READ receive constructor: built by createXferReq(NIXL_READ, ...) on the initiator.
    // Always sets userInitiated = true, state = PRE_START, and marshalDone = false;
    // directDone is derived from direct_child (null => no direct sub-part => directDone
    // starts true, as for a marshal-only READ).
    inboundXferReqH(
        const std::string &remote_agent,
        nixl_xfer_dlist_t &&dst_list,
        const std::array<slotWorkItem, slots_per_xfer> &local_slots,
        size_t xfer_id,
        size_t total_chunks,
        std::string serialized_src_list,
        nixlXferReqH *direct_child,
        std::optional<nixl_blob_t> notif,
        nixlServiceXferReqH *parent,
        nixl_marshal_phase_t marshal_phase,
        std::optional<nixlMarshalDeltaReceiverRefArgs> receiver_delta_ref = std::nullopt);
};

class slotPool {
private:
    uintptr_t baseAddr_;
    uintptr_t registeredDescAddr_;
    size_t slotSize_;
    size_t numSlots_;
    size_t chunkSize_;
    std::optional<size_t> workspaceSize_;
    // TODO-Eyal: create struct Layout{slotSize, chunkSize, workspaceSize}
    nixl_mem_t type_;
    /**
     * @brief  RAII-owned CUDA streams, one per slot.
     */
    std::vector<cudaStream> streams_;

    /**
     * @brief  LIFO free list of work items.
     *
     * @details
     * Allocation pops the back; free pushes it back.  When non-empty, every
     * element's @c stream handle refers to one of the streams in @c streams_.
     */
    std::vector<slotT> freeList_;

    void
    moveFrom(slotPool &&other) noexcept;

public:
    slotPool(uintptr_t base_addr,
             uintptr_t registered_desc_addr,
             size_t slot_size,
             size_t num_slots,
             size_t chunk_size,
             nixl_mem_t type,
             std::optional<size_t> workspace_size = std::nullopt);

    slotPool(const slotPool &) = delete;
    slotPool &
    operator=(const slotPool &) = delete;

    slotPool(slotPool &&) noexcept;
    slotPool &
    operator=(slotPool &&) noexcept;

    ~slotPool() = default;

    [[nodiscard]] std::optional<slotT>
    allocateSlot() noexcept;

    void
    freeSlot(slotT work_item) noexcept;

    [[nodiscard]] size_t
    getNumSlots() const noexcept;

    [[nodiscard]] size_t
    getSlotSize() const noexcept;

    [[nodiscard]] size_t
    getChunkSize() const noexcept;

    [[nodiscard]] uintptr_t
    getRegisteredDescAddr() const noexcept;

    [[nodiscard]] nixl_mem_t
    getType() const noexcept;
};

struct rtsNotifPayload {
    size_t xferId;
    std::string serializedDstList;
    size_t slotSize;
    std::optional<nixlMarshalDeltaReceiverRefArgs> deltaOptArgs;

    rtsNotifPayload(size_t xfer_id, const std::string &dst_list, size_t slot_size);
    explicit rtsNotifPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

struct ctsNotifPayload {
    size_t xferId;
    std::array<nixlBasicDesc, slots_per_xfer> receiverSlotDescriptors;
    std::array<nixlMarshal::mem_space_t, slots_per_xfer> receiverMemSpaces;

    ctsNotifPayload(
        size_t xfer_id,
        const std::array<nixlBasicDesc, slots_per_xfer> &receiver_slot_descriptors,
        const std::array<nixlMarshal::mem_space_t, slots_per_xfer> &receiver_mem_spaces);
    explicit ctsNotifPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

struct rslotNotifPayload {
    size_t xferId;
    size_t slotIndex;

    rslotNotifPayload(size_t xfer_id, size_t slot_index);
    explicit rslotNotifPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

struct deleteNotifPayload {
    size_t xferId;

    explicit deleteNotifPayload(size_t xfer_id);
    explicit deleteNotifPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

/**
 * @struct marshalLayoutFingerprint
 * @brief Fixed-width descriptor of a marshal-mode configuration and its physical slot
 *        layout, embedded in RREQ so a peer can validate compatibility with its own
 *        configuration before serving a READ.
 *        `slotSize` alone (as used by the existing RTS/CTS handshake) is not a sufficient
 *        compatibility check, since two different (chunkedPayloadSize, algo) configurations
 *        can coincidentally yield the same physical slot size. This checks layout/algorithm
 *        compatibility only, not a protocol version - see handleRREQ's doc comment for the
 *        matched-build assumption this implies.
 *
 * @details
 * Every field is a fixed-width scalar in a stable order, so the struct can be
 * (de)serialized without any variable-length framing.
 */
struct marshalLayoutFingerprint {
    // nixl_marshal_config_t variant index (Direct=0, Staging=1, Delta=2, Compress=3).
    uint32_t mode = 0;
    // nixl_marshal_compress_algo_t value; only meaningful when mode is Compress.
    uint32_t algo = 0;
    uint64_t chunkedPayloadSize = 0;
    uint64_t chunkSize = 0;
    // Usable bytes per slot for marshalled data, i.e. slot size excluding the workspace tail.
    uint64_t wireDataCapacity = 0;
    // nixl_mem_t value of the registered service memory.
    uint32_t memType = 0;

    [[nodiscard]] bool
    operator==(const marshalLayoutFingerprint &other) const noexcept {
        return mode == other.mode && algo == other.algo &&
            chunkedPayloadSize == other.chunkedPayloadSize && chunkSize == other.chunkSize &&
            wireDataCapacity == other.wireDataCapacity && memType == other.memType;
    }

    [[nodiscard]] bool
    operator!=(const marshalLayoutFingerprint &other) const noexcept {
        return !(*this == other);
    }
};

/**
 * @brief Serialize a fingerprint to a fixed-width byte blob. The result carries no framing
 *        or prefix of its own; callers embed it inline within a larger message (e.g. RREQ).
 */
[[nodiscard]] nixl_blob_t
serializeFingerprint(const marshalLayoutFingerprint &fingerprint) noexcept;

/**
 * @brief Deserialize a fingerprint from a standalone byte blob.
 * @throw std::runtime_error if `bytes` is shorter than a fingerprint's fixed-width encoding,
 *        so a truncated/malformed message is rejected rather than read out of bounds.
 */
[[nodiscard]] marshalLayoutFingerprint
deserializeFingerprint(std::string_view bytes);

/**
 * @struct rReqPayload
 * @brief "Read Request" message, sent by the initiator to the peer: the READ counterpart
 *        of rtsNotifPayload. Unlike RTS, the initiator already knows its own receive slots
 *        and advertises them here directly, so no CTS round trip is needed.
 */
struct rReqPayload {
    size_t xferId;
    std::string serializedSrcList;
    std::array<nixlBasicDesc, slots_per_xfer> recvSlotDescriptors;
    std::array<nixlMarshal::mem_space_t, slots_per_xfer> recvMemSpaces;
    marshalLayoutFingerprint fingerprint;
    nixl_marshal_compress_data_type_t dataType = nixl_marshal_compress_data_type_t::FLOAT16;
    // Present only for a delta-mode READ; set after construction, mirroring how
    // rtsNotifPayload::deltaOptArgs is populated for WRITE.
    std::optional<nixlMarshalDeltaSenderRefArgs> deltaOptArgs;

    rReqPayload(size_t xfer_id,
                const std::string &serialized_src_list,
                const std::array<nixlBasicDesc, slots_per_xfer> &recv_slot_descriptors,
                const std::array<nixlMarshal::mem_space_t, slots_per_xfer> &recv_mem_spaces,
                const marshalLayoutFingerprint &fingerprint);
    explicit rReqPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

// A fixed-underlying enum can carry an unnamed value received from the wire. Reject it before
// installing remote request state or passing the selection to the compression backend.
[[nodiscard]] constexpr bool
isValidCompressionDataType(nixl_marshal_compress_data_type_t data_type) noexcept {
    switch (data_type) {
    case nixl_marshal_compress_data_type_t::CHAR:
    case nixl_marshal_compress_data_type_t::UCHAR:
    case nixl_marshal_compress_data_type_t::FLOAT16:
    case nixl_marshal_compress_data_type_t::FLOAT8_E4M3:
        return true;
    case nixl_marshal_compress_data_type_t::NUM_COMPRESS_DATA_TYPES:
        return false;
    }
    return false;
}

/**
 * @struct rPostedPayload
 * @brief "Read Posted" message, sent by the peer to the initiator, piggybacked on the slot
 *        NIXL_WRITE: the READ counterpart of postedNotifPayload. Carries an absolute
 *        destination byte offset instead of descIndex/chunkIndex, since the peer's chunk
 *        iterator walks the source descriptor list, which may not match the initiator's
 *        local chunking.
 */
struct rPostedPayload {
    size_t xferId;
    size_t slotIndex;
    size_t originalSize; // size of data pulled from the peer's source buffer.
    size_t wireSize; // size of the marshalled data transferred through the slot.
    size_t dstByteOffset; // absolute byte offset into the initiator's destination descriptor list.
    std::string metadata; // metadata produced by the marshal layer.
    nixl_marshal_mode_t backend;

    rPostedPayload(size_t xfer_id,
                   size_t slot_index,
                   size_t original_size,
                   size_t wire_size,
                   size_t dst_byte_offset,
                   std::string md,
                   nixl_marshal_mode_t backend);
    explicit rPostedPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

/**
 * @struct rrSlotPayload
 * @brief "Read Ready Slot" message, sent by the initiator to the peer: the READ
 *        counterpart of rslotNotifPayload. Echoes a per-slot generation so a
 *        stale/duplicate ack cannot free a slot that has since been reused for a later
 *        chunk.
 */
struct rrSlotPayload {
    size_t xferId;
    size_t slotIndex;
    uint64_t slotGeneration;

    rrSlotPayload(size_t xfer_id, size_t slot_index, uint64_t slot_generation);
    explicit rrSlotPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

/**
 * @struct rAbortPayload
 * @brief "Read Abort" message, sent by the initiator to the peer: requests the peer
 *        quiesce and cancel a served READ mid-transfer.
 */
struct rAbortPayload {
    size_t xferId;

    explicit rAbortPayload(size_t xfer_id);
    explicit rAbortPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

/**
 * @struct rAbortAckPayload
 * @brief "Read Abort Ack" message, sent by the peer to the initiator: sent only after the
 *        peer has fully quiesced (drained all in-flight writes/CUDA work for the aborted
 *        transfer) and freed its slots, so the initiator may safely free its own recv
 *        slots.
 */
struct rAbortAckPayload {
    size_t xferId;

    explicit rAbortAckPayload(size_t xfer_id);
    explicit rAbortAckPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

/**
 * @struct rNakPayload
 * @brief "Read Not Acknowledged" message, sent by the peer to the initiator: admission
 *        rejection for an RREQ (e.g. fingerprint mismatch, unregistered/unauthorized
 *        source region, or slot pool exhaustion). `errorCode` is a nixl_status_t value
 *        narrowed to int32_t for the wire.
 */
struct rNakPayload {
    size_t xferId;
    int32_t errorCode;

    rNakPayload(size_t xfer_id, int32_t error_code);
    explicit rNakPayload(std::string_view notif);
    [[nodiscard]] nixl_blob_t
    serialize() const noexcept;
};

using service_notif_payload_t = std::variant<rtsNotifPayload,
                                             ctsNotifPayload,
                                             rslotNotifPayload,
                                             deleteNotifPayload,
                                             postedNotifPayload,
                                             rReqPayload,
                                             rPostedPayload,
                                             rrSlotPayload,
                                             rAbortPayload,
                                             rAbortAckPayload,
                                             rNakPayload>;
using service_req_ref_t = std::variant<std::reference_wrapper<nixlServiceXferReqH>,
                                       std::reference_wrapper<inboundXferReqH>>;

struct serviceNotifWorkItem {
    std::string senderAgent = "";
    std::shared_ptr<service_notif_payload_t> payload = nullptr;
};

struct activeSlotWorkItem {
    service_req_ref_t req;
    std::optional<std::reference_wrapper<slotWorkItem>> slot;
};

class nixlServiceAgentData {
protected:
    using deleted_writes_t = std::unordered_map<std::string, std::set<size_t>>;

    friend class nixlServiceAgent;
    friend marshalLayoutFingerprint
    makeFingerprint(const nixlServiceAgentData &data, const slotPool &pool);

    nixlAgent *agent_ = nullptr;
    // This agent's own name, set once by nixlServiceAgent's delegating constructor. Used to
    // detect and reject a loopback READ (remote_agent == self) at createXferReq() time.
    std::string localAgentName_;

    nixl_marshal_config_t mode_;
    /** @brief  Payload bytes a single staging slot commits to. Fixed by the service. */
    size_t chunkedPayloadSize_;
    // Collection is unconditional, exposing to the user is conditional
    const bool captureTelemetry_;
    std::array<std::shared_ptr<nixlMarshal::backend>,
               static_cast<size_t>(nixl_marshal_mode_t::NUM_BACKENDS)>
        backends_;
    std::list<slotPool> localStagingPools_;
    std::optional<slotPool> sourceSlotsPool_;

    /* The sender allocates the xfer id.
    Thus outbound reqs are keyed by xfer id, and inbound reqs are keyed by sender agent name and
    xfer id. */
    size_t nextOutboundXferId_;
    std::unordered_map<size_t, std::unique_ptr<nixlServiceXferReqH>> outboundXferReqs_;
    std::unordered_map<std::string, std::unordered_map<size_t, std::unique_ptr<inboundXferReqH>>>
        inboundXferReqs_;

    /* READ request maps, role-separated from the WRITE maps above so a READ and a WRITE
    between the same pair of agents can never collide on xfer id:
    - readReceiveReqs_: my own READs, keyed by my xfer id (I am the initiator/sink, so my own
      counter is unique - same reasoning as outboundXferReqs_).
    - readServeReqs_: peers' READs from me, keyed by (initiator agent, their xfer id) - I am
      not the initiator here, so the id alone is not unique across peers (same reasoning as
      inboundXferReqs_). */
    std::unordered_map<size_t, std::unique_ptr<inboundXferReqH>> readReceiveReqs_;
    std::unordered_map<std::string,
                       std::unordered_map<size_t, std::unique_ptr<nixlServiceXferReqH>>>
        readServeReqs_;

    /* Queue of slots with active asynchronous work. */
    std::queue<activeSlotWorkItem> activeSlotQueue_;
    /* The queue for service notifs. The producer is the progress thread, the consumer is the main
     * thread */
    spscQueue<serviceNotifWorkItem, spsc_size> serviceNotifQueue_;

    /**
     * @brief  The memory requirements for a single slot of the marshal layer.
     *         This is used to calculate the maximum safe source chunk size.
     */
    nixlMarshal::memoryRequirements marshalSlotMemoryRequirements_;

    /**
     * @brief  Callback for service notifications - starting with _NIXLS_.
     *         The callback is called by the progress thread, and pushes an item to the
     * serviceNotifQueue_.
     *
     * @param  sender_agent The name of the sender agent
     * @param  notif The notification to handle (raw, unpeeled)
     */
    void
    serviceNotifCallback(const std::string &sender_agent, const nixl_blob_t &notif);

    void
    rtsCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    ctsCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    postedCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    rslotCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    deleteCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    rreqCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    rpostedCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    rrslotCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    rabortCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    rabortAckCallback(const std::string &sender_agent, const nixl_blob_t &notif);
    void
    rnakCallback(const std::string &sender_agent, const nixl_blob_t &notif);

    /**
     * @brief  Handle a "Request To Send" notification.
     *
     * @param  sender_agent The name of the sender agent
     * @param  rts_payload The payload to handle
     */
    void
    handleRTS(const std::string &sender_agent, const rtsNotifPayload &rts_payload);

    /**
     * @brief  Handle a "Clear To Send" notification.
     *
     * @param  sender_agent The name of the sender agent
     * @param  cts_payload The payload to handle
     */
    void
    handleCTS(const std::string &sender_agent, const ctsNotifPayload &cts_payload);

    /**
     * @brief  Handle a "Posted" notification, sent by the sender after a nixl transfer is
     * completed.
     *
     * @param  sender_agent The name of the sender agent
     * @param  posted_payload The payload to handle
     */
    void
    handlePosted(const std::string &sender_agent, const postedNotifPayload &posted_payload);

    /**
     * @brief  Handle a "Ready Slot" notification, sent by the receiver after a local slot is
     * filled.
     *
     * @param  source_agent The name of the source agent (the receiver of the xfer)
     * @param  rslot_payload The payload to handle
     */
    void
    handleRSlot(const std::string &source_agent, const rslotNotifPayload &rslot_payload);

    /**
     * @brief  Handle a "Delete" notification, sent by the sender after a transfer is finished.
     *
     * @param  sender_agent The name of the sender agent
     * @param  delete_payload The payload to handle
     */
    void
    handleDelete(const std::string &sender_agent,
                 const deleteNotifPayload &delete_payload,
                 deleted_writes_t &deleted_writes);

    /**
     * @brief  Handle a "Read Request" notification: the initiator is asking this agent to
     *         serve a READ. On success, allocates send slots and starts serving the transfer;
     *         on failure (fingerprint mismatch, unauthorized source region, or slot pool
     *         exhaustion), replies with an RNAK.
     *
     * @note   ASSUMPTION: this serves any well-formed RREQ from a trusted peer running a
     *         matched build. There is no registration-based authorization of the source
     *         region - any descriptor list the peer sends is served as-is - and the
     *         fingerprint checks layout/algo compatibility, not protocol version; both
     *         agents must run the same build. READ trusts the peer at least as much as
     *         WRITE already does, just via a different message (RREQ's source list vs.
     *         RTS's destination list).
     *
     * @param  sender_agent The name of the initiator agent
     * @param  rreq_payload The payload to handle
     */
    void
    handleRREQ(const std::string &sender_agent, const rReqPayload &rreq_payload);

    /**
     * @brief  Handle a "Read Posted" notification, sent by the peer piggybacked on the slot
     *         NIXL_WRITE: submits the inbound marshal (decode) operation for the received
     *         slot into the initiator's destination buffer.
     *
     * @param  sender_agent The name of the peer (source) agent
     * @param  rposted_payload The payload to handle
     */
    void
    handleRPosted(const std::string &sender_agent, const rPostedPayload &rposted_payload);

    /**
     * @brief  Handle a "Read Ready Slot" notification, sent by the initiator to the peer: a
     *         receive slot has been drained and its remote counterpart may be reused for
     *         the next chunk. When this was the last chunk, marks the serving request DONE;
     *         progressService()'s end-of-tick scan of readServeReqs_ frees it once every
     *         local slot has actually gone idle - an RRSLOT's arrival reflects the peer's
     *         decode, not this agent's own send completion, so it must not be erased inline.
     *
     * @param  source_agent The name of the initiator agent (the sink of the READ)
     * @param  rrslot_payload The payload to handle
     */
    void
    handleRRSlot(const std::string &source_agent, const rrSlotPayload &rrslot_payload);

    /**
     * @brief  Handle a "Read Abort" notification, sent by the initiator: it is cancelling a
     *         READ we are serving. If already drained (or there was nothing in flight to
     *         drain), acknowledges immediately; otherwise transitions to CANCELLING,
     *         stopping fillLocalSlot from starting new work on this request's slots.
     *         progressService()'s end-of-tick scan of readServeReqs_ acknowledges and frees
     *         it once every local slot has gone idle.
     *
     * @param  sender_agent The name of the initiator agent
     * @param  rabort_payload The payload to handle
     */
    void
    handleRAbort(const std::string &sender_agent, const rAbortPayload &rabort_payload);

    /**
     * @brief  Handle a "Read Abort Ack" notification, sent by the peer: it has fully
     *         quiesced the aborted READ, so this agent may now safely free its own receive
     *         slots once any of its own still-in-flight decodes also finish draining. Sets
     *         remoteQuiesced on the receive context; progressService()'s end-of-tick scan
     *         of readReceiveReqs_ frees and finalizes it once every local slot is also idle.
     *
     * @param  sender_agent The name of the peer agent
     * @param  raback_payload The payload to handle
     */
    void
    handleRAbortAck(const std::string &sender_agent, const rAbortAckPayload &raback_payload);

    /**
     * @brief  Handle a "Read Not Acknowledged" notification, sent by the peer: it rejected
     *         an RREQ; store the terminal status so getXferStatus() can surface it.
     *
     * @param  sender_agent The name of the peer agent
     * @param  rnak_payload The payload to handle
     */
    void
    handleRNak(const std::string &sender_agent, const rNakPayload &rnak_payload);

    /**
     * @brief  Generate a "Delete" notification for a given transfer request.
     *
     * @param  remote_agent The name of the remote agent
     * @param  xfer_id The ID of the transfer request
     * @return nixl_status_t error code in case of failure
     */
    nixl_status_t
    genDelete(const std::string &remote_agent, size_t xfer_id);

    /**
     * @brief  Allocate a local staging slot group for outbound transfer setup.
     *
     * @return Allocated slot group on success, std::nullopt on failure.
     */
    [[nodiscard]] std::optional<std::list<slotT>>
    allocateSlotGroup();

    // TODO-Eyal: move this function to the anonymous namespace
    /**
     * @brief  Free a local staging slot group.
     *
     * @param  slot_group The slot group to free
     */
    void
    freeSlotGroup(const std::array<slotWorkItem, slots_per_xfer> &slot_group) noexcept;
    // TODO-Eyal: Unify the slot-group container across the API

    /**
     * @brief Fill one local staging slot from the current outbound chunk.
     *
     * @param work_item An active slot from the current work queue being processed.
     * @param processed_queue Output queue for work items that should remain pending
     *                        after this progress tick.
     */
    void
    fillLocalSlot(const activeSlotWorkItem &work_item,
                  std::queue<activeSlotWorkItem> &processed_queue);

    /**
     * @brief Poll completion of an outbound slot processing operation.
     *
     * @param work_item An active slot from the current work queue being processed.
     * @param processed_queue Output queue for work items that should remain pending
     *                        after this progress tick.
     */
    void
    pollOutboundSlotCompletion(const activeSlotWorkItem &work_item,
                               std::queue<activeSlotWorkItem> &processed_queue);

    /**
     * @brief Try to transfer a filled local slot to the remote agent.
     *        Note: this function is called after the slot is filled,
     *        but it is not guaranteed that the remote slot is free.
     *        The service notifications update the remote slot state.
     *
     * @param work_item An active slot from the current work queue being processed.
     * @param processed_queue Output queue for work items that should remain pending
     *                        after this progress tick.
     */
    void
    trySend(const activeSlotWorkItem &work_item, std::queue<activeSlotWorkItem> &processed_queue);

    /**
     * @brief Write one ready compressed PRE chunk directly to its ordered DRAM extent.
     *
     * @param work_item An active slot from the current work queue being processed.
     * @param processed_queue Output queue for work items that should remain pending
     *                        after this progress tick.
     */
    void
    trySendAsymmetricPre(const activeSlotWorkItem &work_item,
                         std::queue<activeSlotWorkItem> &processed_queue);

    /**
     * @brief Fail an asymmetric PRE request and release an idle local slot.
     *
     * @param req_data The request containing the PRE state.
     * @param slot The slot to release.
     * @param status The terminal failure status.
     */
    void
    failAsymmetricPreSlot(nixlServiceXferReqH::nonDirectDataH &req_data,
                          slotWorkItem &slot,
                          nixl_status_t status);

    /**
     * @brief Poll completion of a posted NIXL transfer for a slot.
     *
     * @param work_item An active slot from the current work queue being processed.
     * @param processed_queue Output queue for work items that should remain pending
     *                        after this progress tick.
     */
    void
    pollNixlXferCompletion(const activeSlotWorkItem &work_item,
                           std::queue<activeSlotWorkItem> &processed_queue);

    /**
     * @brief Finish one PRE payload write and start the header write after the final payload.
     *
     * @param work_item The slot whose payload write completed.
     * @param processed_queue Output queue for work items that should remain pending.
     * @param status Completion status of the payload write.
     */
    void
    handleAsymmetricPrePayloadCompletion(const activeSlotWorkItem &work_item,
                                         std::queue<activeSlotWorkItem> &processed_queue,
                                         nixl_status_t status);

    /**
     * @brief Poll completion of the final compressed-object header write.
     *
     * @param work_item An active slot from the current work queue being processed.
     * @param processed_queue Output queue for work items that should remain pending
     *                        after this progress tick.
     */
    void
    pollHeaderXferCompletion(const activeSlotWorkItem &work_item,
                             std::queue<activeSlotWorkItem> &processed_queue);

    nixl_status_t
    startAsymmetricPost(inboundXferReqH &inbound_req);

    void
    validateAndIndexAsymmetricPostHeader(inboundXferReqH &inbound_req, size_t max_payload_size);

    void
    pollAsymmetricPostHeaderCompletion(const activeSlotWorkItem &work_item,
                                       std::queue<activeSlotWorkItem> &processed_queue);

    void
    tryReadAsymmetricPost(const activeSlotWorkItem &work_item,
                          std::queue<activeSlotWorkItem> &processed_queue);

    void
    pollAsymmetricPostReadCompletion(const activeSlotWorkItem &work_item,
                                     std::queue<activeSlotWorkItem> &processed_queue);

    /**
     * @brief Poll completion of an inbound marshal operation for a local slot.
     *
     * @param work_item An active slot from the current work queue being processed.
     * @param processed_queue Output queue for work items that should remain pending
     *                        after this progress tick.
     */
    void
    pollInboundSlotCompletion(const activeSlotWorkItem &work_item,
                              std::queue<activeSlotWorkItem> &processed_queue);

    /**
     * @brief Poll and advance a READ receive context's direct child once per progress tick,
     *        then try to finalize the request. A no-op unless the context is IN_PROGRESS
     *        with a not-yet-done direct child (mixed READ only - marshal-only starts with
     *        directDone already true and directChild null). Idempotent: safe to call every
     *        tick regardless of the context's current state.
     *
     * @param ctx The READ receive context to advance.
     */
    void
    progressReadReceive(inboundXferReqH &ctx);

    /** @brief Progress the active slots that were queued at the start of this pass. */
    void
    progressActiveSlots();

    /** @brief Free slots and erase WRITE receive requests marked by DELETE notifications. */
    void
    cleanupDeletedWrites(deleted_writes_t &deleted_writes);

    /** @brief Finalize completed or fully drained READ serve requests. */
    void
    cleanupReadServes();

    /** @brief Progress READ receive contexts and erase fully drained cancellations. */
    void
    progressReadReceives();

    /** @brief Reclaim asymmetric PRE slots after all compression and writes have drained. */
    void
    cleanupAsymmetricPreTransfers();

    /** @brief Make all releases observed so far visible to the slot allocator. */
    void
    progressAndCleanup(deleted_writes_t &deleted_writes);

    /** @brief Process queued service notifications in FIFO order. */
    void
    processServiceNotifications(deleted_writes_t &deleted_writes);

public:
    nixlServiceAgentData(const nixl_marshal_config_t &mode,
                         size_t chunked_payload_size,
                         bool capture_telemetry,
                         std::shared_ptr<nixlMarshal::backend> backend);

    /**
     * @brief  Progress the service work queue and drive service progress.
     *         Called by nixlServiceAgent::getNotifs() and getXferStatus().
     *
     * @return nixl_status_t
     */
    nixl_status_t
    progressService();

    /**
     * @brief  Generate a "Request To Send" message for a given transfer request.
     *         This should be sent once per transfer request, before the actual data is sent.
     *
     * @param  xfer_req The transfer request handle
     * @return nixl_status_t error code in case of failure
     */
    nixl_status_t
    genRTS(nixlServiceXferReqH &xfer_req);

    /**
     * @brief  Generate a "Clear To Send" message for a given remote agent.
     *         This should be sent once both the receiver's staging chunks are not in use.
     *
     * @param  remote_agent The name of the remote agent
     * @return nixl_status_t error code in case of failure
     */
    nixl_status_t
    genCTS(const std::string &remote_agent,
           size_t xfer_id,
           const std::array<slotWorkItem, slots_per_xfer> &slots);

    /**
     * @brief  Generate a "Ready Slot" message for a given remote agent.
     *         Sent by the receiver once a certain slot is clear to be filled.
     *
     * @param  remote_agent The name of the remote agent
     * @param  xfer_id The ID of the transfer request
     * @param  slot_index The index of the slot to be filled
     * @return nixl_status_t error code in case of failure
     */
    nixl_status_t
    genRSlot(const std::string &remote_agent, size_t xfer_id, size_t slot_index);

    /**
     * @brief  Generate a "Read Request" message for a given READ handle. This is the READ
     *         counterpart of genRTS: it should be sent once per READ, when postXferReq() is
     *         called. Unlike genRTS, no CTS reply is expected - the receive context (looked
     *         up via `xfer_req.readReceiveXferId`) already advertises its own recv slots
     *         directly. Takes the outer handle (rather than the receive context alone,
     *         mirroring genRTS's signature) because the optional delta sender ref lives in
     *         `xfer_req.marshalOptArgs`, not on the receive context.
     *
     * @param  xfer_req The READ handle returned by createXferReq(NIXL_READ, ...)
     * @return nixl_status_t error code in case of failure
     */
    nixl_status_t
    genRREQ(nixlServiceXferReqH &xfer_req);

    /**
     * @brief  Generate a "Read Ready Slot" message for a given remote agent. This is the READ
     *         counterpart of genRSlot: sent by the initiator once a given receive slot is
     *         clear to be filled with the next chunk.
     *
     * @param  remote_agent The name of the remote (serving) agent
     * @param  xfer_id The ID of the READ transfer request
     * @param  slot_index The index of the slot to be filled
     * @param  slot_generation The current generation of the slot (see slotGenerations)
     * @return nixl_status_t error code in case of failure
     */
    nixl_status_t
    genRRSlot(const std::string &remote_agent,
              size_t xfer_id,
              size_t slot_index,
              uint64_t slot_generation);

    /**
     * @brief  Generate a "Read Abort" message for a given remote agent: requests the peer
     *         quiesce and cancel a READ it is serving. Sent by releaseXferReq() on a
     *         mid-transfer release of a READ handle.
     *
     * @param  remote_agent The name of the remote (serving) agent
     * @param  xfer_id The ID of the READ transfer request
     * @return nixl_status_t error code in case of failure
     */
    nixl_status_t
    genRAbort(const std::string &remote_agent, size_t xfer_id);

    /**
     * @brief  Generate a "Read Abort Ack" message for a given remote agent: confirms this
     *         agent has fully quiesced (freed all local slots) for a READ it was serving,
     *         so the initiator may now safely free its own receive slots. Sent by
     *         handleRAbort() once drained, or immediately if there was nothing to drain.
     *
     * @param  remote_agent The name of the remote (initiator) agent
     * @param  xfer_id The ID of the READ transfer request
     * @return nixl_status_t error code in case of failure
     */
    nixl_status_t
    genRAbortAck(const std::string &remote_agent, size_t xfer_id);

    /**
     * @brief  If a READ receive context has finished both its direct and marshal
     *         sub-transfers, finalizes it (transitions to DONE) and delivers the user
     *         notification, if one was requested, to the peer. A no-op otherwise, including
     *         when the request is already done - safe to call from multiple sites (this
     *         file's pollInboundSlotCompletion and progressReadReceive), whichever of
     *         directDone/marshalDone happens to become true last.
     *
     * @param  inbound_req The READ receive context to check/finalize.
     */
    void
    tryCompleteReadReceive(inboundXferReqH &inbound_req);

    /**
     * @brief  Return one leased slot and erase it from its xfer.
     *
     * @param  req_data The request we want to unlease a slot from.
     * @param  work_item The slot we want to unlease.
     */
    void
    freeSingleSourceSlot(nixlServiceXferReqH::nonDirectDataH &req_data,
                         slotWorkItem &work_item) noexcept;
};

/**
 * @brief  Build the capability/layout fingerprint that identifies `data`'s marshal
 *         configuration together with `pool`'s physical layout, for embedding in an
 *         outgoing RREQ.
 *
 * @param  data The service agent data whose marshal configuration is being described.
 * @param  pool A slot pool whose physical layout (chunk size, slot size, workspace) is
 *              being described.
 * @return The resulting fingerprint.
 */
[[nodiscard]] marshalLayoutFingerprint
makeFingerprint(const nixlServiceAgentData &data, const slotPool &pool);

#endif // NIXL_SERVICE_DATA_H
