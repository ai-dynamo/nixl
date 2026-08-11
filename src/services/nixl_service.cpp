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
/**
 * @file nixl_service.cpp
 * @brief Initial scaffold for nixlServiceAgent.  All public methods are
 *        currently unimplemented and return NIXL_ERR_NOT_SUPPORTED so the
 *        surface links cleanly while the service implementation is built
 *        up incrementally.
 */
#include "nixl_service.h"
#include "nixl_service_types.h"
#include "nixl_types.h"
#include "marshal/marshal_backend.h"
#include "marshal/staging/staging_backend.h"
#include "marshal/delta/delta_backend.h"
#ifdef NIXL_HAVE_NVCOMP
#include "marshal/compression/compression_backend.h"
#endif
#include "nixl_service_data.h"
#include "backend/backend_aux.h"
#include "nixl_log.h"
#include "serdes/serdes.h"
#include "absl/cleanup/cleanup.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

using namespace nixlMarshal;

namespace {

constexpr size_t
alignUp(size_t value, size_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Both peers of a transfer must agree on this (see marshalLayoutFingerprint), so the
// service fixes it rather than exposing it through nixlServiceAgentConfig.
constexpr size_t default_chunked_payload_size = 128UL * 1024 * 1024;

constexpr char nixl_s_prefix[] = "_NIXLS_";
constexpr char nixl_srts_prefix[] = "_NIXLS_RTS_";
constexpr char nixl_scts_prefix[] = "_NIXLS_CTS_";
constexpr char nixl_sposted_prefix[] = "_NIXLS_POSTED_";
constexpr char nixl_srslot_prefix[] = "_NIXLS_RSLOT_";
constexpr char nixl_sdelete_prefix[] = "_NIXLS_DELETE_";
// READ protocol: the peer serves the transfer by pushing data back to the initiator.
constexpr char nixl_srreq_prefix[] = "_NIXLS_RREQ_";
constexpr char nixl_srposted_prefix[] = "_NIXLS_RPOSTED_";
constexpr char nixl_srrslot_prefix[] = "_NIXLS_RRSLOT_";
constexpr char nixl_srabort_prefix[] = "_NIXLS_RABORT_";
constexpr char nixl_srabortack_prefix[] = "_NIXLS_RABACK_";
constexpr char nixl_srnak_prefix[] = "_NIXLS_RNAK_";

template<class... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template<typename T>
void
writeScalar(char *&cursor, const T &value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(cursor, &value, sizeof(T));
    cursor += sizeof(T);
}

void
writeBytes(char *&cursor, const void *data, size_t size) noexcept {
    if (size == 0) {
        return;
    }
    std::memcpy(cursor, data, size);
    cursor += size;
}

template<typename T>
T
readScalar(const char *&cursor, const char *end) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    NIXL_ASSERT(cursor + sizeof(T) <= end);
    T value{};
    std::memcpy(&value, cursor, sizeof(T));
    cursor += sizeof(T);
    return value;
}

void
readBytes(const char *&cursor, const char *end, void *data, size_t size) noexcept {
    NIXL_ASSERT(cursor + size <= end);
    if (size == 0) {
        return;
    }
    std::memcpy(data, cursor, size);
    cursor += size;
}

std::string
readString(const char *&cursor, const char *end, size_t size) noexcept {
    NIXL_ASSERT(cursor + size <= end);
    std::string value(cursor, size);
    cursor += size;
    return value;
}

// Fixed-width byte size of a serialized marshalLayoutFingerprint: 3 uint32_t fields
// (mode, algo, memType) plus 3 uint64_t fields (chunkedPayloadSize, chunkSize,
// wireDataCapacity). No padding, since fields are written individually via writeScalar
// rather than as a single struct-sized memcpy.
constexpr size_t fingerprint_wire_size = 3 * sizeof(uint32_t) + 3 * sizeof(uint64_t);

void
writeFingerprint(char *&cursor, const marshalLayoutFingerprint &fingerprint) noexcept {
    writeScalar(cursor, fingerprint.mode);
    writeScalar(cursor, fingerprint.algo);
    writeScalar(cursor, fingerprint.chunkedPayloadSize);
    writeScalar(cursor, fingerprint.chunkSize);
    writeScalar(cursor, fingerprint.wireDataCapacity);
    writeScalar(cursor, fingerprint.memType);
}

// Bounds-checked: throws (rather than asserting) if fewer than fingerprint_wire_size bytes
// remain between `cursor` and `end`, so a truncated/malformed message can be rejected
// cleanly by the caller instead of crashing the process on untrusted remote input.
marshalLayoutFingerprint
readFingerprint(const char *&cursor, const char *end) {
    if (static_cast<size_t>(end - cursor) < fingerprint_wire_size) {
        throw std::runtime_error("readFingerprint: truncated buffer");
    }
    marshalLayoutFingerprint fingerprint;
    fingerprint.mode = readScalar<uint32_t>(cursor, end);
    fingerprint.algo = readScalar<uint32_t>(cursor, end);
    fingerprint.chunkedPayloadSize = readScalar<uint64_t>(cursor, end);
    fingerprint.chunkSize = readScalar<uint64_t>(cursor, end);
    fingerprint.wireDataCapacity = readScalar<uint64_t>(cursor, end);
    fingerprint.memType = readScalar<uint32_t>(cursor, end);
    return fingerprint;
}

std::shared_ptr<std::vector<ChunkDivision::segment>>
getOutboundSegments(const outboundSlotCompletionData &completion) {
    if (completion.size != marshal_derived_size) {
        return ChunkDivision::defaultSegments(completion.size);
    }

    const auto chunk_division_it =
        std::find_if(completion.options.begin(), completion.options.end(), [](const auto &option) {
            return std::holds_alternative<ChunkDivision::processSlotOutput>(option);
        });
    if (chunk_division_it == completion.options.end()) {
        throw std::runtime_error("outbound completion missing chunk division output");
    }

    const auto &chunk_division = std::get<ChunkDivision::processSlotOutput>(*chunk_division_it);
    if (!chunk_division.segments) {
        throw std::runtime_error("outbound completion chunk division is null");
    }
    return chunk_division.segments;
}

/**
 * @brief  Dummy backend used when no marshalling is requested. Both processSlot overrides throw
 *         before constructing any asyncHandle, so no concrete handle subclass is needed here.
 */
class nixlMarshalBackendDirect final : public backend {
private:
    struct passkey {
        explicit passkey() = default;
    };

public:
    static std::shared_ptr<nixlMarshalBackendDirect>
    createBackend() {
        return std::make_shared<nixlMarshalBackendDirect>(passkey{});
    }

    explicit nixlMarshalBackendDirect(passkey) {}

    const std::vector<mem_space_t> &
    getSupportedMemSpaces() const override {
        static const std::vector<mem_space_t> k_none = {};
        return k_none;
    }

    std::unique_ptr<inbound_async_handle_t>
    inboundProcessSlot(const slotBuffers & /*buffers*/,
                       const std::string & /*metadata*/,
                       const process_slot_input_options_t & /*opts*/) override {
        throw std::runtime_error("Not implemented");
    }

    std::unique_ptr<outbound_async_handle_t>
    outboundProcessSlot(const slotBuffers & /*buffers*/,
                        const process_slot_input_options_t & /*opts*/) override {
        throw std::runtime_error("Not implemented");
    }

    memoryRequirements
    getSlotMemoryRequirements() const noexcept override {
        return memoryRequirements{{}};
    }
};

inline mem_space_t
memSpaceFromNixlMem(nixl_mem_t mem) {
    switch (mem) {
    case VRAM_SEG:
        return mem_space_t::DEVICE;
    case DRAM_SEG:
        return mem_space_t::HOST;
    default:
        throw std::runtime_error("Invalid memory type");
    }
}

inline nixl_mem_t
nixlMemFromMemSpace(nixlMarshal::mem_space_t mem_space) {
    switch (mem_space) {
    case nixlMarshal::mem_space_t::DEVICE:
        return VRAM_SEG;
    case nixlMarshal::mem_space_t::HOST:
        return DRAM_SEG;
    default:
        throw std::runtime_error("Invalid memory space");
    }
}

inline runtimeBuffer
runtimeBufferFromDesc(const nixlBlobDesc &desc, nixl_mem_t mem) {
    return runtimeBuffer(absl::Span<std::byte>(reinterpret_cast<std::byte *>(desc.addr), desc.len),
                         memSpaceFromNixlMem(mem));
}

void
addReferenceOption(process_slot_input_options_t &opts,
                   std::byte *ref,
                   size_t offset,
                   size_t size,
                   nixl_mem_t mem_type,
                   size_t element_size) {
    opts[option_t::READ_ONLY_REFERENCE_STRUCTURED_MEMORY] =
        ReadOnlyReferenceStructuredMemory::processSlotInput{
            runtimeBuffer(absl::Span<std::byte>(ref + offset, size), memSpaceFromNixlMem(mem_type)),
            element_size};
}

// True for a READ-serving request draining to quiescence (CANCELLING after an RABORT, or
// DONE once every chunk is acked). The serve-side slot handlers must let any in-flight
// async op finish, then mark the slot idle and drop it rather than re-queuing it - only
// then does "every slot FREE" (checked at end-of-tick) safely imply the request can be
// freed without a stale queue entry later dereferencing it.
bool
isTerminalReadServe(nixl_xfer_op_t op, nixl_service_xfer_state_t state) noexcept {
    return op == NIXL_READ &&
        (state == nixl_service_xfer_state_t::CANCELLING ||
         state == nixl_service_xfer_state_t::DONE);
}

const nixlMarshalDeltaOptArgs *
getDeltaOptArgs(const nixl_marshal_opt_args_t &marshal_opt_args) {
    if (const auto *delta = std::get_if<nixlMarshalDeltaOptArgs>(&marshal_opt_args)) {
        return delta;
    }
    if (const auto *compress = std::get_if<nixlMarshalCompressOptArgs>(&marshal_opt_args)) {
        if (compress->delta.has_value()) {
            return &compress->delta.value();
        }
    }
    return nullptr;
}

std::shared_ptr<backend>
makeMarshalBackend(const nixl_marshal_config_t &mode, size_t chunked_payload_size) {
    return std::visit(overloaded{
                          [](const nixlMarshalDirectConfig &) -> std::shared_ptr<backend> {
                              return nixlMarshalBackendDirect::createBackend();
                          },
                          [](const nixlMarshalStagingConfig &cfg) -> std::shared_ptr<backend> {
                              return stagingBackend::createBackend(cfg);
                          },
                          [](const nixlMarshalDeltaConfig &cfg) -> std::shared_ptr<backend> {
                              return deltaBackend::createBackend(cfg);
                          },
#ifdef NIXL_HAVE_NVCOMP
                          [chunked_payload_size](
                              const nixlMarshalCompressConfig &cfg) -> std::shared_ptr<backend> {
                              return compressionBackend::createBackend(cfg, chunked_payload_size);
                          },
#else
                          [](const nixlMarshalCompressConfig &) -> std::shared_ptr<backend> {
                              throw std::invalid_argument(
                                  "Compression marshal backend requires nvCOMP support");
                          },
#endif
                      },
                      mode);
}

std::optional<nixl_marshal_opt_args_t>
getValidMarshalOptArgs(const nixl_service_opt_args_t *extra_params,
                       const nixl_marshal_config_t &mode) {
    if (extra_params != nullptr && extra_params->marshalOptArgs.has_value()) {
        const auto &marshal_opt_args = extra_params->marshalOptArgs.value();
        if (!std::holds_alternative<nixlMarshalDirectOptArgs>(marshal_opt_args) &&
            marshal_opt_args.index() != mode.index()) {
            return std::nullopt;
        }
        return marshal_opt_args;
    }

    return std::visit(overloaded{
                          [](const nixlMarshalDirectConfig &) -> nixl_marshal_opt_args_t {
                              return nixlMarshalDirectOptArgs{};
                          },
                          [](const nixlMarshalStagingConfig &) -> nixl_marshal_opt_args_t {
                              return nixlMarshalStagingOptArgs{};
                          },
                          [](const nixlMarshalDeltaConfig &) -> nixl_marshal_opt_args_t {
                              return nixlMarshalDeltaOptArgs{};
                          },
                          [](const nixlMarshalCompressConfig &) -> nixl_marshal_opt_args_t {
                              return nixlMarshalCompressOptArgs{};
                          },
                      },
                      mode);
}

// Resolves the notification to carry on a READ receive context from `current` (the value
// captured at an earlier create/post) and `extra_params`, mirroring the base nixlAgent's own
// precedence (see nixl_agent.cpp's createXferReq/postXferReq): a present extra_params->notif
// always wins; failing that, the legacy hasNotif/notifMsg pair; a null extra_params leaves
// `current` untouched (so a bare postXferReq() after a createXferReq(..., extra_params) does
// not drop the create-time notification); a non-null extra_params carrying neither clears it.
std::optional<nixl_blob_t>
resolveReadNotif(std::optional<nixl_blob_t> current, const nixl_opt_args_t *extra_params) {
    if (extra_params == nullptr) {
        return current;
    }
    if (extra_params->notif.has_value()) {
        return extra_params->notif;
    }
    if (extra_params->hasNotif) {
        return extra_params->notifMsg;
    }
    return std::nullopt;
}

auto
makeSlotGroupCleanup(std::list<slotT> &slot_group) {
    return absl::MakeCleanup([&slot_group]() {
        while (!slot_group.empty()) {
            const auto slot = slot_group.front();
            slot_group.pop_front();
            NIXL_ASSERT(slot.pool != nullptr);
            slot.pool->freeSlot(slot);
        }
    });
}

std::array<slotWorkItem, slots_per_xfer>
slotGroupToWorkItems(const std::list<slotT> &slot_group) {
    NIXL_ASSERT(slot_group.size() == slots_per_xfer);
    std::array<slotWorkItem, slots_per_xfer> slot_work_items;
    size_t i = 0;
    for (const auto &slot : slot_group) {
        slot_work_items[i] = slotWorkItem{slot, i};
        ++i;
    }
    return slot_work_items;
}

int
countChunksAndVerifyMatch(const nixl_xfer_dlist_t &local_desc_list,
                          const nixl_xfer_dlist_t &remote_desc_list,
                          size_t chunk_size) {
    auto desc_count = local_desc_list.descCount();
    if (desc_count != remote_desc_list.descCount()) {
        return -1;
    }
    size_t total_chunks = 0;
    for (int i = 0; i < desc_count; ++i) {
        auto desc_len = local_desc_list[i].len;
        if (desc_len != remote_desc_list[i].len) {
            return -1;
        }
        total_chunks += (desc_len + chunk_size - 1) / chunk_size;
    }
    return static_cast<int>(total_chunks);
}

// Resolves an absolute byte offset (relative to the start of dst_list) plus a size into a
// single concrete address range, or returns std::nullopt if the range does not fall
// entirely within one descriptor of dst_list. Used to validate a remotely-supplied offset
// before it is ever dereferenced, rather than trusting a peer's descIndex/chunkIndex pair.
std::optional<nixlBasicDesc>
resolveAbsoluteOffset(const nixl_xfer_dlist_t &dst_list, size_t byte_offset, size_t size) {
    size_t cumulative = 0;
    for (int i = 0; i < dst_list.descCount(); ++i) {
        const auto &desc = dst_list[i];
        if (byte_offset < cumulative + desc.len) {
            const size_t offset_in_desc = byte_offset - cumulative;
            // Rearranged from "offset_in_desc + size > desc.len" so the (remotely-supplied,
            // unvalidated) size is never added to anything: offset_in_desc < desc.len is
            // already established above, so desc.len - offset_in_desc cannot underflow,
            // whereas offset_in_desc + size could silently wrap around for a size close to
            // SIZE_MAX and pass this check when it should not.
            if (size > desc.len - offset_in_desc) {
                return std::nullopt;
            }
            return nixlBasicDesc(desc.addr + offset_in_desc, size, desc.devId);
        }
        cumulative += desc.len;
    }
    return std::nullopt;
}

constexpr size_t direct_desc_threshold = 64 * 1024 * 1024;

struct splitDescLists {
    nixl_xfer_dlist_t directLocal;
    nixl_xfer_dlist_t directRemote;
    nixl_xfer_dlist_t marshalLocal;
    nixl_xfer_dlist_t marshalRemote;
    bool valid = true;

    splitDescLists(nixl_mem_t local_type, nixl_mem_t remote_type)
        : directLocal(local_type),
          directRemote(remote_type),
          marshalLocal(local_type),
          marshalRemote(remote_type) {}
};

splitDescLists
splitSmallDescPairs(const nixl_xfer_dlist_t &local_desc_list,
                    const nixl_xfer_dlist_t &remote_desc_list) {
    splitDescLists split(local_desc_list.getType(), remote_desc_list.getType());
    const auto desc_count = local_desc_list.descCount();
    if (desc_count != remote_desc_list.descCount() || desc_count == 0) {
        split.valid = false;
        return split;
    }

    for (int i = 0; i < desc_count; ++i) {
        const auto desc_len = local_desc_list[i].len;
        if (desc_len != remote_desc_list[i].len) {
            split.valid = false;
            return split;
        }

        if (desc_len <= direct_desc_threshold) {
            split.directLocal.addDesc(local_desc_list[i]);
            split.directRemote.addDesc(remote_desc_list[i]);
        } else {
            split.marshalLocal.addDesc(local_desc_list[i]);
            split.marshalRemote.addDesc(remote_desc_list[i]);
        }
    }

    return split;
}

template<typename PayloadT>
void
notifCallbackTemplate(const char *prefix,
                      const std::string &sender_agent,
                      const nixl_blob_t &notif,
                      spscQueue<serviceNotifWorkItem, spsc_size> &service_notif_queue) {
    NIXL_ASSERT(notif.rfind(prefix, 0) == 0);
    // PayloadT's deserializing constructor throws on truncated/malformed input (e.g.
    // readFingerprint()); this callback runs on the backend's own notification-delivery
    // thread, so an uncaught exception here would escape across that boundary instead of
    // just dropping one bad message.
    std::optional<PayloadT> payload;
    try {
        payload.emplace(notif);
    }
    catch (const std::exception &e) {
        NIXL_WARN << "nixlServiceAgentData: dropping malformed notification with prefix '" << prefix
                  << "' from '" << sender_agent << "': " << e.what();
        return;
    }
    if (!service_notif_queue.push(serviceNotifWorkItem{
            sender_agent, std::make_shared<service_notif_payload_t>(std::move(*payload))})) {
        // TODO-Eyal: handle error.
        NIXL_ASSERT(false);
    }
}

} // namespace

nixl_blob_t
serializeFingerprint(const marshalLayoutFingerprint &fingerprint) noexcept {
    nixl_blob_t bytes(fingerprint_wire_size, '\0');
    char *cursor = bytes.data();
    writeFingerprint(cursor, fingerprint);
    NIXL_ASSERT(cursor == bytes.data() + fingerprint_wire_size);
    return bytes;
}

marshalLayoutFingerprint
deserializeFingerprint(std::string_view bytes) {
    const char *cursor = bytes.data();
    const char *end = bytes.data() + bytes.size();
    auto fingerprint = readFingerprint(cursor, end);
    NIXL_ASSERT(cursor == end);
    return fingerprint;
}

rtsNotifPayload::rtsNotifPayload(size_t xfer_id, const std::string &dst_list, size_t slot_size)
    : xferId(xfer_id),
      serializedDstList(dst_list),
      slotSize(slot_size) {
    NIXL_ASSERT(!serializedDstList.empty());
    NIXL_ASSERT(slotSize > 0);
}

rtsNotifPayload::rtsNotifPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_srts_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_srts_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    slotSize = readScalar<size_t>(cursor, end);
    const auto dst_list_size = readScalar<size_t>(cursor, end);
    const auto has_delta = readScalar<bool>(cursor, end);
    if (has_delta) {
        const auto ref = reinterpret_cast<std::byte *>(readScalar<uintptr_t>(cursor, end));
        const auto mem_type = readScalar<nixl_mem_t>(cursor, end);
        const auto element_size = readScalar<size_t>(cursor, end);
        deltaOptArgs = nixlMarshalDeltaReceiverRefArgs{ref, mem_type, element_size};
    }
    serializedDstList = readString(cursor, end, dst_list_size);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
rtsNotifPayload::serialize() const noexcept {
    const auto prefix_len = std::char_traits<char>::length(nixl_srts_prefix);
    const auto has_delta = deltaOptArgs.has_value();
    const auto serialized_dst_list_size = serializedDstList.size();
    const auto delta_size = has_delta ?
        sizeof(uintptr_t) + sizeof(deltaOptArgs->memType) + sizeof(deltaOptArgs->elementSize) :
        0;
    const auto base_fields_size =
        sizeof(xferId) + sizeof(slotSize) + sizeof(serialized_dst_list_size) + sizeof(has_delta);
    const auto total_size = prefix_len + base_fields_size + delta_size + serialized_dst_list_size;
    nixl_blob_t rts_msg(total_size, '\0');
    char *cursor = rts_msg.data();
    writeBytes(cursor, nixl_srts_prefix, prefix_len);
    writeScalar(cursor, xferId);
    writeScalar(cursor, slotSize);
    writeScalar(cursor, serialized_dst_list_size);
    writeScalar(cursor, has_delta);
    if (has_delta) {
        const auto ref = reinterpret_cast<uintptr_t>(deltaOptArgs->ref);
        writeScalar(cursor, ref);
        writeScalar(cursor, deltaOptArgs->memType);
        writeScalar(cursor, deltaOptArgs->elementSize);
    }
    writeBytes(cursor, serializedDstList.data(), serialized_dst_list_size);
    NIXL_ASSERT(cursor == rts_msg.data() + total_size);
    [[maybe_unused]] rtsNotifPayload deserialized(rts_msg);
    NIXL_ASSERT(deserialized.xferId == xferId);
    NIXL_ASSERT(deserialized.serializedDstList == serializedDstList);
    NIXL_ASSERT(deserialized.slotSize == slotSize);
    NIXL_ASSERT(deserialized.deltaOptArgs.has_value() == deltaOptArgs.has_value());
    if (deltaOptArgs.has_value()) {
        NIXL_ASSERT(deserialized.deltaOptArgs->ref == deltaOptArgs->ref);
        NIXL_ASSERT(deserialized.deltaOptArgs->memType == deltaOptArgs->memType);
        NIXL_ASSERT(deserialized.deltaOptArgs->elementSize == deltaOptArgs->elementSize);
    }
    return rts_msg;
}

ctsNotifPayload::ctsNotifPayload(
    size_t xfer_id,
    const std::array<nixlBasicDesc, slots_per_xfer> &receiver_slot_descriptors,
    const std::array<nixlMarshal::mem_space_t, slots_per_xfer> &receiver_mem_spaces)
    : xferId(xfer_id),
      receiverSlotDescriptors(receiver_slot_descriptors),
      receiverMemSpaces(receiver_mem_spaces) {}

ctsNotifPayload::ctsNotifPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_scts_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_scts_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    readBytes(cursor, end, receiverSlotDescriptors.data(), sizeof(receiverSlotDescriptors));
    readBytes(cursor, end, receiverMemSpaces.data(), sizeof(receiverMemSpaces));
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
ctsNotifPayload::serialize() const noexcept {
    const auto prefix_len = std::char_traits<char>::length(nixl_scts_prefix);
    const auto total_size =
        prefix_len + sizeof(xferId) + sizeof(receiverSlotDescriptors) + sizeof(receiverMemSpaces);
    nixl_blob_t cts_msg(total_size, '\0');
    char *cursor = cts_msg.data();
    writeBytes(cursor, nixl_scts_prefix, prefix_len);
    writeScalar(cursor, xferId);
    writeBytes(cursor, receiverSlotDescriptors.data(), sizeof(receiverSlotDescriptors));
    writeBytes(cursor, receiverMemSpaces.data(), sizeof(receiverMemSpaces));
    NIXL_ASSERT(cursor == cts_msg.data() + total_size);
    [[maybe_unused]] ctsNotifPayload deserialized(cts_msg);
    NIXL_ASSERT(deserialized.xferId == xferId);
    NIXL_ASSERT(deserialized.receiverSlotDescriptors == receiverSlotDescriptors);
    NIXL_ASSERT(deserialized.receiverMemSpaces == receiverMemSpaces);
    return cts_msg;
}

rslotNotifPayload::rslotNotifPayload(size_t xfer_id, size_t slot_index)
    : xferId(xfer_id),
      slotIndex(slot_index) {}

rslotNotifPayload::rslotNotifPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_srslot_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_srslot_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    slotIndex = readScalar<size_t>(cursor, end);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
rslotNotifPayload::serialize() const noexcept {
    const auto prefix_len = std::char_traits<char>::length(nixl_srslot_prefix);
    const auto total_size = prefix_len + sizeof(xferId) + sizeof(slotIndex);
    nixl_blob_t rslot_msg(total_size, '\0');
    char *cursor = rslot_msg.data();
    writeBytes(cursor, nixl_srslot_prefix, prefix_len);
    writeScalar(cursor, xferId);
    writeScalar(cursor, slotIndex);
    NIXL_ASSERT(cursor == rslot_msg.data() + total_size);
    [[maybe_unused]] rslotNotifPayload deserialized(rslot_msg);
    NIXL_ASSERT(deserialized.xferId == xferId);
    NIXL_ASSERT(deserialized.slotIndex == slotIndex);
    return rslot_msg;
}

deleteNotifPayload::deleteNotifPayload(size_t xfer_id) : xferId(xfer_id) {}

deleteNotifPayload::deleteNotifPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_sdelete_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_sdelete_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
deleteNotifPayload::serialize() const noexcept {
    const auto prefix_len = std::char_traits<char>::length(nixl_sdelete_prefix);
    const auto total_size = prefix_len + sizeof(xferId);
    nixl_blob_t delete_msg(total_size, '\0');
    char *cursor = delete_msg.data();
    writeBytes(cursor, nixl_sdelete_prefix, prefix_len);
    writeScalar(cursor, xferId);
    NIXL_ASSERT(cursor == delete_msg.data() + total_size);
    [[maybe_unused]] deleteNotifPayload deserialized(delete_msg);
    NIXL_ASSERT(deserialized.xferId == xferId);
    return delete_msg;
}

postedNotifPayload::postedNotifPayload(
    size_t xfer_id,
    size_t slot_index,
    size_t original_size,
    std::shared_ptr<std::vector<nixlMarshal::ChunkDivision::segment>> posted_segments,
    size_t desc_index,
    size_t chunk_index,
    std::string md)
    : xferId(xfer_id),
      slotIndex(slot_index),
      originalSize(original_size),
      postedSegments(std::move(posted_segments)),
      descIndex(desc_index),
      chunkIndex(chunk_index),
      metadata(std::move(md)) {}

postedNotifPayload::postedNotifPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_sposted_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_sposted_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    slotIndex = readScalar<size_t>(cursor, end);
    originalSize = readScalar<size_t>(cursor, end);
    const auto posted_segment_count = readScalar<size_t>(cursor, end);
    NIXL_ASSERT(posted_segment_count > 0);
    postedSegments =
        std::make_shared<std::vector<nixlMarshal::ChunkDivision::segment>>(posted_segment_count);
    readBytes(
        cursor, end, postedSegments->data(), postedSegments->size() * sizeof((*postedSegments)[0]));
    descIndex = readScalar<size_t>(cursor, end);
    chunkIndex = readScalar<size_t>(cursor, end);
    const auto metadata_size = readScalar<size_t>(cursor, end);
    metadata = readString(cursor, end, metadata_size);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
postedNotifPayload::serialize() const noexcept {
    NIXL_ASSERT(postedSegments && !postedSegments->empty());
    const auto prefix_len = std::char_traits<char>::length(nixl_sposted_prefix);
    const auto posted_segment_count = postedSegments->size();
    const auto metadata_size = metadata.size();
    const auto segment_size = posted_segment_count * sizeof((*postedSegments)[0]);
    const auto base_fields_size = sizeof(xferId) + sizeof(slotIndex) + sizeof(originalSize) +
        sizeof(posted_segment_count) + sizeof(descIndex) + sizeof(chunkIndex) +
        sizeof(metadata_size);
    const auto total_size = prefix_len + base_fields_size + segment_size + metadata_size;
    nixl_blob_t posted_msg(total_size, '\0');
    char *cursor = posted_msg.data();
    writeBytes(cursor, nixl_sposted_prefix, prefix_len);
    writeScalar(cursor, xferId);
    writeScalar(cursor, slotIndex);
    writeScalar(cursor, originalSize);
    writeScalar(cursor, posted_segment_count);
    writeBytes(cursor, postedSegments->data(), segment_size);
    writeScalar(cursor, descIndex);
    writeScalar(cursor, chunkIndex);
    writeScalar(cursor, metadata_size);
    writeBytes(cursor, metadata.data(), metadata_size);
    NIXL_ASSERT(cursor == posted_msg.data() + total_size);
    [[maybe_unused]] postedNotifPayload deserialized(posted_msg);
    NIXL_ASSERT(deserialized.xferId == xferId);
    NIXL_ASSERT(deserialized.slotIndex == slotIndex);
    NIXL_ASSERT(deserialized.originalSize == originalSize);
    NIXL_ASSERT(deserialized.postedSegments->size() == postedSegments->size());
    for (size_t i = 0; i < postedSegments->size(); ++i) {
        NIXL_ASSERT((*deserialized.postedSegments)[i].offset == (*postedSegments)[i].offset);
        NIXL_ASSERT((*deserialized.postedSegments)[i].size == (*postedSegments)[i].size);
    }
    NIXL_ASSERT(deserialized.descIndex == descIndex);
    NIXL_ASSERT(deserialized.chunkIndex == chunkIndex);
    NIXL_ASSERT(deserialized.metadata == metadata);
    return posted_msg;
}

rReqPayload::rReqPayload(
    size_t xfer_id,
    const std::string &serialized_src_list,
    const std::array<nixlBasicDesc, slots_per_xfer> &recv_slot_descriptors,
    const std::array<nixlMarshal::mem_space_t, slots_per_xfer> &recv_mem_spaces,
    const marshalLayoutFingerprint &fingerprint)
    : xferId(xfer_id),
      serializedSrcList(serialized_src_list),
      recvSlotDescriptors(recv_slot_descriptors),
      recvMemSpaces(recv_mem_spaces),
      fingerprint(fingerprint) {
    NIXL_ASSERT(!serializedSrcList.empty());
}

rReqPayload::rReqPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_srreq_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_srreq_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    readBytes(cursor, end, recvSlotDescriptors.data(), sizeof(recvSlotDescriptors));
    readBytes(cursor, end, recvMemSpaces.data(), sizeof(recvMemSpaces));
    fingerprint = readFingerprint(cursor, end);
    const auto has_delta = readScalar<bool>(cursor, end);
    if (has_delta) {
        const auto ref = reinterpret_cast<std::byte *>(readScalar<uintptr_t>(cursor, end));
        const auto mem_type = readScalar<nixl_mem_t>(cursor, end);
        const auto element_size = readScalar<size_t>(cursor, end);
        deltaOptArgs = nixlMarshalDeltaSenderRefArgs{ref, mem_type, element_size};
    }
    const auto src_list_size = readScalar<size_t>(cursor, end);
    serializedSrcList = readString(cursor, end, src_list_size);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
rReqPayload::serialize() const noexcept {
    const auto prefix_len = std::char_traits<char>::length(nixl_srreq_prefix);
    const auto has_delta = deltaOptArgs.has_value();
    const auto src_list_size = serializedSrcList.size();
    const auto delta_size = has_delta ?
        sizeof(uintptr_t) + sizeof(deltaOptArgs->memType) + sizeof(deltaOptArgs->elementSize) :
        0;
    const auto base_fields_size = sizeof(xferId) + sizeof(recvSlotDescriptors) +
        sizeof(recvMemSpaces) + fingerprint_wire_size + sizeof(has_delta) + sizeof(src_list_size);
    const auto total_size = prefix_len + base_fields_size + delta_size + src_list_size;
    nixl_blob_t rreq_msg(total_size, '\0');
    char *cursor = rreq_msg.data();
    writeBytes(cursor, nixl_srreq_prefix, prefix_len);
    writeScalar(cursor, xferId);
    writeBytes(cursor, recvSlotDescriptors.data(), sizeof(recvSlotDescriptors));
    writeBytes(cursor, recvMemSpaces.data(), sizeof(recvMemSpaces));
    writeFingerprint(cursor, fingerprint);
    writeScalar(cursor, has_delta);
    if (has_delta) {
        const auto ref = reinterpret_cast<uintptr_t>(deltaOptArgs->ref);
        writeScalar(cursor, ref);
        writeScalar(cursor, deltaOptArgs->memType);
        writeScalar(cursor, deltaOptArgs->elementSize);
    }
    writeScalar(cursor, src_list_size);
    writeBytes(cursor, serializedSrcList.data(), src_list_size);
    NIXL_ASSERT(cursor == rreq_msg.data() + total_size);
    return rreq_msg;
}

rPostedPayload::rPostedPayload(
    size_t xfer_id,
    size_t slot_index,
    size_t original_size,
    std::shared_ptr<std::vector<nixlMarshal::ChunkDivision::segment>> posted_segments,
    size_t dst_byte_offset,
    std::string md)
    : xferId(xfer_id),
      slotIndex(slot_index),
      originalSize(original_size),
      postedSegments(std::move(posted_segments)),
      dstByteOffset(dst_byte_offset),
      metadata(std::move(md)) {}

rPostedPayload::rPostedPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_srposted_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_srposted_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    slotIndex = readScalar<size_t>(cursor, end);
    originalSize = readScalar<size_t>(cursor, end);
    const auto posted_segment_count = readScalar<size_t>(cursor, end);
    NIXL_ASSERT(posted_segment_count > 0);
    postedSegments =
        std::make_shared<std::vector<nixlMarshal::ChunkDivision::segment>>(posted_segment_count);
    readBytes(
        cursor, end, postedSegments->data(), postedSegments->size() * sizeof((*postedSegments)[0]));
    dstByteOffset = readScalar<size_t>(cursor, end);
    const auto metadata_size = readScalar<size_t>(cursor, end);
    metadata = readString(cursor, end, metadata_size);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
rPostedPayload::serialize() const noexcept {
    NIXL_ASSERT(postedSegments && !postedSegments->empty());
    const auto prefix_len = std::char_traits<char>::length(nixl_srposted_prefix);
    const auto posted_segment_count = postedSegments->size();
    const auto metadata_size = metadata.size();
    const auto segment_size = posted_segment_count * sizeof((*postedSegments)[0]);
    const auto base_fields_size = sizeof(xferId) + sizeof(slotIndex) + sizeof(originalSize) +
        sizeof(posted_segment_count) + sizeof(dstByteOffset) + sizeof(metadata_size);
    const auto total_size = prefix_len + base_fields_size + segment_size + metadata_size;
    nixl_blob_t rposted_msg(total_size, '\0');
    char *cursor = rposted_msg.data();
    writeBytes(cursor, nixl_srposted_prefix, prefix_len);
    writeScalar(cursor, xferId);
    writeScalar(cursor, slotIndex);
    writeScalar(cursor, originalSize);
    writeScalar(cursor, posted_segment_count);
    writeBytes(cursor, postedSegments->data(), segment_size);
    writeScalar(cursor, dstByteOffset);
    writeScalar(cursor, metadata_size);
    writeBytes(cursor, metadata.data(), metadata_size);
    NIXL_ASSERT(cursor == rposted_msg.data() + total_size);
    return rposted_msg;
}

rrSlotPayload::rrSlotPayload(size_t xfer_id, size_t slot_index, uint64_t slot_generation)
    : xferId(xfer_id),
      slotIndex(slot_index),
      slotGeneration(slot_generation) {}

rrSlotPayload::rrSlotPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_srrslot_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_srrslot_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    slotIndex = readScalar<size_t>(cursor, end);
    slotGeneration = readScalar<uint64_t>(cursor, end);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
rrSlotPayload::serialize() const noexcept {
    const auto prefix_len = std::char_traits<char>::length(nixl_srrslot_prefix);
    const auto total_size =
        prefix_len + sizeof(xferId) + sizeof(slotIndex) + sizeof(slotGeneration);
    nixl_blob_t rrslot_msg(total_size, '\0');
    char *cursor = rrslot_msg.data();
    writeBytes(cursor, nixl_srrslot_prefix, prefix_len);
    writeScalar(cursor, xferId);
    writeScalar(cursor, slotIndex);
    writeScalar(cursor, slotGeneration);
    NIXL_ASSERT(cursor == rrslot_msg.data() + total_size);
    return rrslot_msg;
}

rAbortPayload::rAbortPayload(size_t xfer_id) : xferId(xfer_id) {}

rAbortPayload::rAbortPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_srabort_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_srabort_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
rAbortPayload::serialize() const noexcept {
    const auto prefix_len = std::char_traits<char>::length(nixl_srabort_prefix);
    const auto total_size = prefix_len + sizeof(xferId);
    nixl_blob_t rabort_msg(total_size, '\0');
    char *cursor = rabort_msg.data();
    writeBytes(cursor, nixl_srabort_prefix, prefix_len);
    writeScalar(cursor, xferId);
    NIXL_ASSERT(cursor == rabort_msg.data() + total_size);
    return rabort_msg;
}

rAbortAckPayload::rAbortAckPayload(size_t xfer_id) : xferId(xfer_id) {}

rAbortAckPayload::rAbortAckPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_srabortack_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_srabortack_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
rAbortAckPayload::serialize() const noexcept {
    const auto prefix_len = std::char_traits<char>::length(nixl_srabortack_prefix);
    const auto total_size = prefix_len + sizeof(xferId);
    nixl_blob_t raback_msg(total_size, '\0');
    char *cursor = raback_msg.data();
    writeBytes(cursor, nixl_srabortack_prefix, prefix_len);
    writeScalar(cursor, xferId);
    NIXL_ASSERT(cursor == raback_msg.data() + total_size);
    return raback_msg;
}

rNakPayload::rNakPayload(size_t xfer_id, int32_t error_code)
    : xferId(xfer_id),
      errorCode(error_code) {}

rNakPayload::rNakPayload(std::string_view notif) {
    NIXL_ASSERT(notif.rfind(nixl_srnak_prefix, 0) == 0);
    const auto prefix_len = std::char_traits<char>::length(nixl_srnak_prefix);
    const char *cursor = notif.data() + prefix_len;
    const char *end = notif.data() + notif.size();
    xferId = readScalar<size_t>(cursor, end);
    errorCode = readScalar<int32_t>(cursor, end);
    NIXL_ASSERT(cursor == end);
}

nixl_blob_t
rNakPayload::serialize() const noexcept {
    const auto prefix_len = std::char_traits<char>::length(nixl_srnak_prefix);
    const auto total_size = prefix_len + sizeof(xferId) + sizeof(errorCode);
    nixl_blob_t rnak_msg(total_size, '\0');
    char *cursor = rnak_msg.data();
    writeBytes(cursor, nixl_srnak_prefix, prefix_len);
    writeScalar(cursor, xferId);
    writeScalar(cursor, errorCode);
    NIXL_ASSERT(cursor == rnak_msg.data() + total_size);
    return rnak_msg;
}

slotT::slotT(slotPool *slot_pool,
             uintptr_t base_addr,
             size_t slot_size,
             cudaStream_t stream,
             nixl_mem_t type,
             size_t chunk_size,
             std::optional<size_t> workspace_size)
    : pool(slot_pool),
      baseAddr(base_addr),
      slotSize(slot_size),
      workspaceSize(workspace_size),
      chunkSize(chunk_size),
      stream(stream),
      type(type) {
    cudaGetDevice(&deviceId_);
}

nixlBasicDesc
slotT::toDesc(size_t len) const noexcept {
    NIXL_ASSERT(len <= slotSize);
    return nixlBasicDesc(baseAddr, len, deviceId_);
}

runtimeBuffer
slotT::toRuntimeBuffer() const {
    const size_t workspace_size = workspaceSize.value_or(0);
    return runtimeBuffer(
        absl::Span<std::byte>(reinterpret_cast<std::byte *>(baseAddr), slotSize - workspace_size),
        memSpaceFromNixlMem(type));
}

nixlMarshal::process_slot_input_options_t
slotT::getProcessSlotInputOptions() const {
    nixlMarshal::process_slot_input_options_t options;
    options[nixlMarshal::option_t::USER_CUDA_STREAM] =
        nixlMarshal::UserCudaStream::processSlotInput{stream};
    const size_t workspace_size = workspaceSize.value_or(0);
    if (workspace_size) {
        options[nixlMarshal::option_t::WRITEABLE_WORKSPACE_MEMORY] =
            nixlMarshal::WriteableWorkspaceMemory::processSlotInput{
                runtimeBuffer(absl::Span<std::byte>(reinterpret_cast<std::byte *>(
                                                        baseAddr + slotSize - workspace_size),
                                                    workspace_size),
                              memSpaceFromNixlMem(type))};
    }
    return options;
}

slotWorkItem::slotWorkItem(slotT slot, size_t slot_index) : slot(slot), slotIndex(slot_index) {}

slotPool::slotPool(uintptr_t base_addr,
                   size_t slot_size,
                   size_t num_slots,
                   size_t chunk_size,
                   nixl_mem_t type,
                   std::optional<size_t> workspace_size)
    : baseAddr_(base_addr),
      slotSize_(slot_size),
      numSlots_(num_slots),
      chunkSize_(chunk_size),
      workspaceSize_(workspace_size),
      type_(type) {
    // TODO-Eyal: replace this assertion with something more friendly. It's for nvComp.
    NIXL_ASSERT(base_addr % 8 == 0);
    NIXL_ASSERT(slot_size % 8 == 0);
    streams_.reserve(num_slots);
    freeList_.reserve(num_slots);
    for (size_t i = 0; i < num_slots; ++i) {
        streams_.emplace_back();
        freeList_.push_back(slotT{this,
                                  baseAddr_ + static_cast<uintptr_t>(i) * slot_size,
                                  slotSize_,
                                  streams_.back().get(),
                                  type_,
                                  chunkSize_,
                                  workspaceSize_});
    }
}

std::optional<slotT>
slotPool::allocateSlot() noexcept {
    if (freeList_.empty()) {
        return std::nullopt;
    }
    slotT item = freeList_.back();
    freeList_.pop_back();
    return item;
}

void
slotPool::freeSlot(slotT work_item) noexcept {
    freeList_.push_back(work_item);
}

size_t
slotPool::getNumSlots() const noexcept {
    return numSlots_;
}

size_t
slotPool::getSlotSize() const noexcept {
    return slotSize_;
}

size_t
slotPool::getChunkSize() const noexcept {
    return chunkSize_;
}

uintptr_t
slotPool::getBaseAddr() const noexcept {
    return baseAddr_;
}

nixl_mem_t
slotPool::getType() const noexcept {
    return type_;
}

nixlServiceXferReqH::nixlServiceXferReqH(
    const nixl_xfer_dlist_t &src_desc_list,
    const std::string &serialized_dst_desc_list,
    const std::string &remote_agent,
    const std::array<slotWorkItem, slots_per_xfer> &local_slots,
    size_t xfer_id,
    size_t total_chunks,
    const nixl_marshal_opt_args_t &marshal_opt_args)
    : xferReq(nullptr),
      nonDirectData(nonDirectDataH{
          remote_agent,
          serialized_dst_desc_list,
          xfer_id,
          nixl_service_xfer_state_t::PRE_START,
          chunkIteratorH(src_desc_list, local_slots[0].slot.chunkSize, total_chunks),
          local_slots,
          std::array<std::unique_ptr<outbound_async_handle_t>, slots_per_xfer>{},
          nixlServiceAgent::trackCompressionRatio ? std::make_unique<compressionStats>() : nullptr,
          std::array<nixlBasicDesc, slots_per_xfer>{},
          std::array<nixlMarshal::mem_space_t, slots_per_xfer>{},
          std::array{remote_slot_state_t::NOT_ALLOCATED, remote_slot_state_t::NOT_ALLOCATED},
          std::array<nixlXferReqH *, slots_per_xfer>{}}),
      marshalOptArgs(marshal_opt_args) {}

inboundXferReqH::inboundXferReqH(const std::string &remote_agent,
                                 nixl_xfer_dlist_t &&dst_list,
                                 const std::array<slotWorkItem, slots_per_xfer> &local_slots,
                                 size_t xfer_id,
                                 std::optional<nixlMarshalDeltaReceiverRefArgs> receiver_delta_ref)
    : remoteAgent(remote_agent),
      dstList(std::move(dst_list)),
      xferId(xfer_id),
      state(nixl_service_xfer_state_t::IN_PROGRESS),
      localSlots(local_slots),
      receiverDeltaRef(receiver_delta_ref) {}

inboundXferReqH::inboundXferReqH(const std::string &remote_agent,
                                 nixl_xfer_dlist_t &&dst_list,
                                 const std::array<slotWorkItem, slots_per_xfer> &local_slots,
                                 size_t xfer_id,
                                 size_t total_chunks,
                                 std::string serialized_src_list,
                                 nixlXferReqH *direct_child,
                                 std::optional<nixl_blob_t> notif,
                                 std::optional<nixlMarshalDeltaReceiverRefArgs> receiver_delta_ref)
    : remoteAgent(remote_agent),
      dstList(std::move(dst_list)),
      xferId(xfer_id),
      state(nixl_service_xfer_state_t::PRE_START),
      localSlots(local_slots),
      receiverDeltaRef(receiver_delta_ref),
      serializedSrcList(std::move(serialized_src_list)),
      totalChunks(total_chunks),
      notif(std::move(notif)),
      userInitiated(true),
      directChild(direct_child),
      directDone(direct_child == nullptr),
      marshalDone(false) {}

nixlServiceAgentData::nixlServiceAgentData(const nixl_marshal_config_t &mode,
                                           size_t chunked_payload_size)
    : mode_(mode),
      chunkedPayloadSize_(chunked_payload_size),
      backend_(makeMarshalBackend(mode_, chunkedPayloadSize_)),
      nextOutboundXferId_(0) {
    // Direct mode stages nothing, so it has no per-slot requirements to query.
    if (!std::holds_alternative<nixlMarshalDirectConfig>(mode_)) {
        marshalSlotMemoryRequirements_ = backend_->getSlotMemoryRequirements();
    }
}

marshalLayoutFingerprint
makeFingerprint(const nixlServiceAgentData &data, const slotPool &pool) {
    marshalLayoutFingerprint fingerprint;
    fingerprint.mode = static_cast<uint32_t>(data.mode_.index());
    if (const auto *compress_cfg = std::get_if<nixlMarshalCompressConfig>(&data.mode_)) {
        fingerprint.algo = static_cast<uint32_t>(compress_cfg->algo);
    }
    fingerprint.chunkedPayloadSize =
        std::holds_alternative<nixlMarshalDirectConfig>(data.mode_) ? 0 : data.chunkedPayloadSize_;
    fingerprint.chunkSize = pool.getChunkSize();
    fingerprint.memType = static_cast<uint32_t>(pool.getType());

    size_t slot_workspace_size = 0;
    const auto it_workspace =
        data.marshalSlotMemoryRequirements_.opts.find(option_t::WRITEABLE_WORKSPACE_MEMORY);
    if (it_workspace != data.marshalSlotMemoryRequirements_.opts.end()) {
        if (const auto *ws_req =
                std::get_if<WriteableWorkspaceMemory::memoryRequirements>(&it_workspace->second)) {
            slot_workspace_size = ws_req->slotWorkspaceSize;
        }
    }
    NIXL_ASSERT(pool.getSlotSize() >= slot_workspace_size);
    fingerprint.wireDataCapacity = pool.getSlotSize() - slot_workspace_size;
    return fingerprint;
}

nixl_status_t
nixlServiceAgentData::progressService() {
    std::queue<activeSlotWorkItem> processed_queue;
    std::unordered_map<std::string, std::set<size_t>> deleted_reqs;
    for (auto notif_item = serviceNotifQueue_.tryPop(); notif_item.has_value();
         notif_item = serviceNotifQueue_.tryPop()) {
        const auto &item = notif_item.value();
        NIXL_ASSERT(item.payload != nullptr);
        std::visit(
            overloaded{[&](const rtsNotifPayload &p) { handleRTS(item.senderAgent, p); },
                       [&](const ctsNotifPayload &p) { handleCTS(item.senderAgent, p); },
                       [&](const postedNotifPayload &p) { handlePosted(item.senderAgent, p); },
                       [&](const rslotNotifPayload &p) { handleRSlot(item.senderAgent, p); },
                       [&](const deleteNotifPayload &p) {
                           handleDelete(item.senderAgent, p, deleted_reqs);
                           NIXL_ASSERT(deleted_reqs.count(item.senderAgent) == 1);
                           NIXL_ASSERT(deleted_reqs[item.senderAgent].size() >= 1);
                       },
                       [&](const rReqPayload &p) { handleRREQ(item.senderAgent, p); },
                       [&](const rPostedPayload &p) { handleRPosted(item.senderAgent, p); },
                       [&](const rrSlotPayload &p) { handleRRSlot(item.senderAgent, p); },
                       [&](const rAbortPayload &p) { handleRAbort(item.senderAgent, p); },
                       [&](const rAbortAckPayload &p) { handleRAbortAck(item.senderAgent, p); },
                       [&](const rNakPayload &p) { handleRNak(item.senderAgent, p); }},
            *item.payload);
    }
    while (!activeSlotQueue_.empty()) {
        auto &work_item = activeSlotQueue_.front();
        std::visit(overloaded{[&](std::reference_wrapper<nixlServiceXferReqH>) {
                                  switch (work_item.slot.get().state) {
                                  case local_slot_state_t::FREE:
                                      fillLocalSlot(work_item, processed_queue);
                                      break;
                                  case local_slot_state_t::BUSY_MARSHAL:
                                      pollOutboundSlotCompletion(work_item, processed_queue);
                                      break;
                                  case local_slot_state_t::READY_TO_SEND:
                                      trySend(work_item, processed_queue);
                                      break;
                                  case local_slot_state_t::BUSY_NIXL:
                                      pollNixlXferCompletion(work_item, processed_queue);
                                      break;
                                  }
                              },
                              [&](std::reference_wrapper<inboundXferReqH>) {
                                  NIXL_ASSERT(work_item.slot.get().state ==
                                              local_slot_state_t::BUSY_MARSHAL);
                                  pollInboundSlotCompletion(work_item, processed_queue);
                              }},
                   work_item.req);
        activeSlotQueue_.pop();
    }

    for (const auto &[sender_agent, xfer_ids] : deleted_reqs) {
        auto sender_it = inboundXferReqs_.find(sender_agent);
        if (sender_it == inboundXferReqs_.end()) {
            continue;
        }
        auto &sender_reqs = sender_it->second;
        for (size_t xfer_id : xfer_ids) {
            auto req_it = sender_reqs.find(xfer_id);
            if (req_it == sender_reqs.end()) {
                continue;
            }
            freeSlotGroup(req_it->second->localSlots);
            sender_reqs.erase(req_it);
        }
    }

    // Cleanup for READ serves that finished normally (handleRRSlot, state == DONE) or are
    // draining after an RABORT (handleRAbort, state == CANCELLING): neither "every chunk
    // acked" nor "abort requested" implies every local slot has actually gone idle yet, from
    // this agent's own point of view (an RRSLOT/RABORT is a remote signal, independent of
    // pollOutboundSlotCompletion/pollNixlXferCompletion noticing the matching local send as
    // complete) - so this may take several ticks. The authoritative state already lives on
    // each request, so this scans readServeReqs_ directly rather than tracking a separate
    // index of "terminal" keys that could desync from it.
    for (auto agent_it = readServeReqs_.begin(); agent_it != readServeReqs_.end();) {
        auto &xfer_reqs = agent_it->second;
        const std::string &initiator_agent = agent_it->first;
        for (auto xfer_it = xfer_reqs.begin(); xfer_it != xfer_reqs.end();) {
            NIXL_ASSERT(xfer_it->second->nonDirectData.has_value());
            auto &req_data = xfer_it->second->nonDirectData.value();
            if (req_data.state != nixl_service_xfer_state_t::DONE &&
                req_data.state != nixl_service_xfer_state_t::CANCELLING) {
                ++xfer_it;
                continue;
            }
            const bool fully_drained = std::all_of(
                req_data.localSlots.begin(), req_data.localSlots.end(), [](const slotWorkItem &s) {
                    return s.state == local_slot_state_t::FREE;
                });
            if (!fully_drained) {
                ++xfer_it;
                continue;
            }
            if (req_data.state == nixl_service_xfer_state_t::CANCELLING) {
                [[maybe_unused]] auto ack_ret = genRAbortAck(initiator_agent, xfer_it->first);
                // TODO: handle a genNotif failure here.
            }
            freeSlotGroup(req_data.localSlots);
            xfer_it = xfer_reqs.erase(xfer_it);
        }
        if (xfer_reqs.empty()) {
            agent_it = readServeReqs_.erase(agent_it);
        } else {
            ++agent_it;
        }
    }

    // Same idea, for this agent's own READs currently draining after an RABORT_ACK (see
    // handleRAbortAck): the peer has confirmed it is done touching these slots
    // (remoteQuiesced), but any of this agent's own in-flight decodes still need to drain
    // first. This scan is also the only place a still-in-progress READ's direct child (if
    // any) gets polled: it lives on this map-owned context rather than the caller-owned
    // outer handle, so only progressService() (also driven by getNotifs()) can reach it -
    // see progressReadReceive().
    for (auto xfer_it = readReceiveReqs_.begin(); xfer_it != readReceiveReqs_.end();) {
        auto &receive_req = *xfer_it->second;
        progressReadReceive(receive_req);
        if (receive_req.state != nixl_service_xfer_state_t::CANCELLING ||
            !receive_req.remoteQuiesced) {
            ++xfer_it;
            continue;
        }
        const bool fully_drained =
            std::all_of(receive_req.localSlots.begin(),
                        receive_req.localSlots.end(),
                        [](const slotWorkItem &s) { return s.state == local_slot_state_t::FREE; });
        if (!fully_drained) {
            ++xfer_it;
            continue;
        }
        freeSlotGroup(receive_req.localSlots);
        xfer_it = readReceiveReqs_.erase(xfer_it);
    }

    activeSlotQueue_.swap(processed_queue);
    return NIXL_SUCCESS;
}

void
nixlServiceAgentData::fillLocalSlot(const activeSlotWorkItem &work_item,
                                    std::queue<activeSlotWorkItem> &processed_queue) {
    auto &req_h = std::get<std::reference_wrapper<nixlServiceXferReqH>>(work_item.req).get();
    auto &req_data = req_h.nonDirectData.value();
    auto &slot = work_item.slot.get();
    NIXL_ASSERT(req_data.state >= nixl_service_xfer_state_t::IN_PROGRESS);
    if (req_data.state == nixl_service_xfer_state_t::DONE) {
        return;
    }
    if (req_data.state == nixl_service_xfer_state_t::CANCELLING) {
        // READ-serving only (WRITE-outbound never reaches CANCELLING): being drained
        // after an RABORT. Do not start new work on this now-idle slot - simply not
        // re-queuing it here is what lets it fall out of activeSlotQueue_. The end-of-tick
        // pass in progressService() notices once every slot of this request has reached
        // this point and finalizes (frees + sends RABORT_ACK).
        return;
    }
    auto src_addr = req_data.chunkIterator.get();
    auto chunk_size = req_data.chunkIterator.currentChunkSize();
    if (!src_addr) {
        // The sender is done pulling data from the user buffer, however there might still be
        // xfers/"processSlot" operations in progress.
        NIXL_ASSERT(chunk_size == 0);
        return;
    }
    NIXL_ASSERT(chunk_size > 0);
    auto slot_index = slot.slotIndex;
    const auto current_chunk_local = req_data.chunkIterator.getCurrentChunkLocal();
    postedNotifPayload posted_notif_payload(
        req_data.xferId,
        slot_index,
        chunk_size,
        // Placeholder {offset: 0, size: 0}; pollOutboundSlotCompletion replaces it with actual
        // segments.
        nixlMarshal::ChunkDivision::defaultSegments(0),
        req_data.chunkIterator.getCurrentDesc(),
        current_chunk_local,
        "");
    req_data.chunkIterator++;
    NIXL_ASSERT(slot.state == local_slot_state_t::FREE);
    slotBuffers buffers;
    buffers.src = runtimeBuffer(absl::Span<std::byte>(src_addr, chunk_size),
                                memSpaceFromNixlMem(req_data.chunkIterator.getMemType()));
    buffers.dst = slot.slot.toRuntimeBuffer();

    auto opts = slot.slot.getProcessSlotInputOptions();
    if (const auto *delta_opt_args = getDeltaOptArgs(req_h.marshalOptArgs)) {
        addReferenceOption(opts,
                           delta_opt_args->senderRef,
                           current_chunk_local * slot.slot.chunkSize,
                           chunk_size,
                           delta_opt_args->senderMemType,
                           delta_opt_args->elementSize);
    }
    req_data.outboundAsyncHandles[slot_index] = backend_->outboundProcessSlot(buffers, opts);
    slot.state = local_slot_state_t::BUSY_MARSHAL;
    slot.postedNotif = posted_notif_payload;
    processed_queue.push(work_item);
}

void
nixlServiceAgentData::pollOutboundSlotCompletion(const activeSlotWorkItem &work_item,
                                                 std::queue<activeSlotWorkItem> &processed_queue) {
    auto &slot = work_item.slot.get();
    auto slot_index = slot.slotIndex;
    auto &req_h = std::get<std::reference_wrapper<nixlServiceXferReqH>>(work_item.req).get();
    auto &req_data = req_h.nonDirectData.value();
    NIXL_ASSERT(req_data.state >= nixl_service_xfer_state_t::IN_PROGRESS);
    if (isTerminalReadServe(req_h.op, req_data.state)) {
        // Let the in-flight encode finish so the slot buffer is safe to reclaim, then drop
        // the now-idle slot (do not advance it to READY_TO_SEND / send it during a drain).
        NIXL_ASSERT(req_data.outboundAsyncHandles[slot_index] != nullptr);
        NIXL_ASSERT(slot.state == local_slot_state_t::BUSY_MARSHAL);
        if (!req_data.outboundAsyncHandles[slot_index]->checkForCompletion().has_value()) {
            processed_queue.push(work_item);
            return;
        }
        req_data.outboundAsyncHandles[slot_index].reset();
        slot.state = local_slot_state_t::FREE;
        return;
    }
    if (req_data.state == nixl_service_xfer_state_t::DONE) {
        return;
    }
    NIXL_ASSERT(req_data.outboundAsyncHandles[slot_index] != nullptr);
    NIXL_ASSERT(slot.state == local_slot_state_t::BUSY_MARSHAL);

    auto completion_data = req_data.outboundAsyncHandles[slot_index]->checkForCompletion();
    if (!completion_data.has_value()) {
        processed_queue.push(work_item);
    } else {
        NIXL_ASSERT(slot.postedNotif.has_value());
        auto &payload = slot.postedNotif.value();
        const auto &completion = completion_data.value();
        payload.postedSegments = getOutboundSegments(completion);
        payload.metadata = completion.metadata;
        if constexpr (nixlServiceAgent::trackCompressionRatio) {
            if (std::holds_alternative<nixlMarshalCompressOptArgs>(req_h.marshalOptArgs)) {
                NIXL_ASSERT(req_data.compressionStatsHandle != nullptr);
                size_t outbound_size = 0;
                for (const auto &segment : *payload.postedSegments) {
                    outbound_size += segment.size;
                }
                auto &stats = *req_data.compressionStatsHandle;
                const auto ratio =
                    static_cast<double>(outbound_size) / static_cast<double>(payload.originalSize);
                if (stats.originalSize == 0) {
                    stats.minRatio = ratio;
                    stats.maxRatio = ratio;
                } else {
                    stats.minRatio = std::min(stats.minRatio, ratio);
                    stats.maxRatio = std::max(stats.maxRatio, ratio);
                }
                stats.compressedSize += outbound_size;
                stats.weightedSumSquaredRatio +=
                    static_cast<double>(payload.originalSize) * ratio * ratio;
                stats.originalSize += payload.originalSize;
            }
        }
        slot.state = local_slot_state_t::READY_TO_SEND;
        processed_queue.push(work_item);
    }
}

void
nixlServiceAgentData::trySend(const activeSlotWorkItem &work_item,
                              std::queue<activeSlotWorkItem> &processed_queue) {
    auto &slot = work_item.slot.get();
    auto slot_index = slot.slotIndex;
    auto &req_h = std::get<std::reference_wrapper<nixlServiceXferReqH>>(work_item.req).get();
    auto &req_data = req_h.nonDirectData.value();
    NIXL_ASSERT(req_data.state >= nixl_service_xfer_state_t::IN_PROGRESS);
    if (isTerminalReadServe(req_h.op, req_data.state)) {
        // The encode is already complete (READY_TO_SEND); during a drain do not send it,
        // just drop the now-idle slot.
        NIXL_ASSERT(slot.state == local_slot_state_t::READY_TO_SEND);
        req_data.outboundAsyncHandles[slot_index].reset();
        slot.postedNotif = std::nullopt;
        slot.state = local_slot_state_t::FREE;
        return;
    }
    if (req_data.state == nixl_service_xfer_state_t::DONE) {
        return;
    }
    NIXL_ASSERT(slot.state == local_slot_state_t::READY_TO_SEND);
    if (req_data.remoteSlotStates[slot_index] == remote_slot_state_t::FREE) {
        NIXL_ASSERT(req_data.nixlXferReqs[slot_index] == nullptr);
        const auto completion =
            req_data.outboundAsyncHandles[slot_index]->checkForCompletion().value();

        nixl_xfer_dlist_t local_slot_dlist(slot.slot.type);
        nixl_xfer_dlist_t remote_slot_dlist(
            nixlMemFromMemSpace(req_data.remoteSlotMemTypes[slot_index]));
        const auto local_slot_desc = slot.slot.toDesc(slot.slot.slotSize);
        auto outbound_segments = getOutboundSegments(completion);
        for (const auto &segment : *outbound_segments) {
            local_slot_dlist.addDesc(nixlBasicDesc(
                local_slot_desc.addr + segment.offset, segment.size, local_slot_desc.devId));
            remote_slot_dlist.addDesc(
                nixlBasicDesc(req_data.remoteSlotDescriptors[slot_index].addr + segment.offset,
                              segment.size,
                              req_data.remoteSlotDescriptors[slot_index].devId));
        }

        nixlXferReqH *slot_xfer_req = nullptr;
        nixl_opt_args_t extra_params;
        NIXL_ASSERT(slot.postedNotif.has_value());
        const auto &payload = slot.postedNotif.value();
        if (req_h.op == NIXL_READ) {
            // Bump this slot's generation before sending, so a stale/duplicate RRSLOT
            // ack for an earlier fill of this same slot (echoing an older generation) can
            // be told apart from the ack for this fill - see handleRRSlot.
            ++req_data.remoteSlotGenerations[slot_index];
            // RPOSTED carries an absolute destination byte offset rather than
            // payload.descIndex/chunkIndex directly: the receiver's descriptor list is
            // guaranteed to have the same per-descriptor lengths as this agent's source
            // list (validated when the READ was created), but not necessarily the same
            // chunk size, so the offset must be self-contained rather than an index the
            // receiver has to reinterpret using its own chunking.
            const auto dst_byte_offset =
                req_data.chunkIterator.getByteOffset(payload.descIndex, payload.chunkIndex);
            rPostedPayload rposted_payload(payload.xferId,
                                           payload.slotIndex,
                                           payload.originalSize,
                                           payload.postedSegments,
                                           dst_byte_offset,
                                           payload.metadata);
            extra_params.notif = rposted_payload.serialize();
        } else {
            extra_params.notif = payload.serialize();
        }
        auto create_ret = agent_->createXferReq(NIXL_WRITE,
                                                local_slot_dlist,
                                                remote_slot_dlist,
                                                req_data.remoteAgent,
                                                slot_xfer_req,
                                                &extra_params);
        if (create_ret != NIXL_SUCCESS) {
            // TODO-Eyal: handle error.
            NIXL_ASSERT(false);
            return;
        }

        auto post_ret = agent_->postXferReq(slot_xfer_req, &extra_params);
        if (post_ret < NIXL_SUCCESS) {
            // TODO-Eyal: handle error.
            NIXL_ASSERT(false);
            return;
        }

        req_data.nixlXferReqs[slot_index] = slot_xfer_req;
        slot.state = local_slot_state_t::BUSY_NIXL;
        slot.postedNotif = std::nullopt;
        req_data.remoteSlotStates[slot_index] = remote_slot_state_t::BUSY;
        processed_queue.push(work_item);
    } else {
        processed_queue.push(work_item);
    }
}

void
nixlServiceAgentData::pollNixlXferCompletion(const activeSlotWorkItem &work_item,
                                             std::queue<activeSlotWorkItem> &processed_queue) {
    auto &slot = work_item.slot.get();
    auto slot_index = slot.slotIndex;
    auto &req_h = std::get<std::reference_wrapper<nixlServiceXferReqH>>(work_item.req).get();
    auto &req_data = req_h.nonDirectData.value();
    NIXL_ASSERT(req_data.state >= nixl_service_xfer_state_t::IN_PROGRESS);
    if (isTerminalReadServe(req_h.op, req_data.state)) {
        // Let the in-flight NIXL_WRITE finish and release it, then drop the now-idle slot.
        // A failure status is tolerated here (not asserted): the request is already
        // terminal, so the send's outcome no longer matters, only that it quiesces.
        NIXL_ASSERT(slot.state == local_slot_state_t::BUSY_NIXL);
        NIXL_ASSERT(req_data.nixlXferReqs[slot_index] != nullptr);
        if (agent_->getXferStatus(req_data.nixlXferReqs[slot_index]) == NIXL_IN_PROG) {
            processed_queue.push(work_item);
            return;
        }
        [[maybe_unused]] auto release_ret =
            agent_->releaseXferReq(req_data.nixlXferReqs[slot_index]);
        NIXL_ASSERT(release_ret == NIXL_SUCCESS);
        req_data.nixlXferReqs[slot_index] = nullptr;
        slot.state = local_slot_state_t::FREE;
        return;
    }
    if (req_data.state == nixl_service_xfer_state_t::DONE) {
        return;
    }
    NIXL_ASSERT(slot.state == local_slot_state_t::BUSY_NIXL);
    NIXL_ASSERT(req_data.nixlXferReqs[slot_index] != nullptr);

    auto status = agent_->getXferStatus(req_data.nixlXferReqs[slot_index]);
    if (status == NIXL_IN_PROG) {
        processed_queue.push(work_item);
        return;
    }

    if (status != NIXL_SUCCESS) {
        // TODO-Eyal: handle transfer error status.
        NIXL_ASSERT(false);
        return;
    }

    auto release_ret = agent_->releaseXferReq(req_data.nixlXferReqs[slot_index]);
    NIXL_ASSERT(release_ret == NIXL_SUCCESS);
    req_data.nixlXferReqs[slot_index] = nullptr;
    slot.state = local_slot_state_t::FREE;
    processed_queue.push(work_item);
}

void
nixlServiceAgentData::pollInboundSlotCompletion(const activeSlotWorkItem &work_item,
                                                std::queue<activeSlotWorkItem> &processed_queue) {
    auto &slot = work_item.slot.get();
    auto slot_index = slot.slotIndex;
    auto &inbound_req = std::get<std::reference_wrapper<inboundXferReqH>>(work_item.req).get();
    if (inbound_req.markedForDeletion) {
        return;
    }
    if (inbound_req.userInitiated) {
        if (inbound_req.state == nixl_service_xfer_state_t::DONE) {
            // Fully finalized (or about to be, at end-of-tick); nothing left to do.
            return;
        }
        // Unlike WRITE, a READ receive context can be draining an in-flight decode after
        // reaching a terminal state: CANCELLING from a mid-transfer release, or FAILED
        // from a codec-integrity mismatch on a sibling slot. Either way, this decode must
        // still be allowed to complete and free its slot below - it just should not feed
        // into the logical READ's own completion bookkeeping anymore (handled further down).
        NIXL_ASSERT(inbound_req.state == nixl_service_xfer_state_t::IN_PROGRESS ||
                    inbound_req.state == nixl_service_xfer_state_t::CANCELLING ||
                    inbound_req.state == nixl_service_xfer_state_t::FAILED);
    } else {
        NIXL_ASSERT(inbound_req.state == nixl_service_xfer_state_t::IN_PROGRESS);
    }
    NIXL_ASSERT(slot.state == local_slot_state_t::BUSY_MARSHAL);
    NIXL_ASSERT(inbound_req.asyncHandles[slot_index] != nullptr);

    auto completion_data = inbound_req.asyncHandles[slot_index]->checkForCompletion();
    if (!completion_data.has_value()) {
        processed_queue.push(work_item);
        return;
    }

    inbound_req.asyncHandles[slot_index].reset();
    slot.state = local_slot_state_t::FREE;

    if (!inbound_req.userInitiated) {
        auto gen_rslot_ret = genRSlot(inbound_req.remoteAgent, inbound_req.xferId, slot_index);
        if (gen_rslot_ret != NIXL_SUCCESS) {
            // TODO: handle error.
            NIXL_ASSERT(false);
            return;
        }
        return;
    }

    if (inbound_req.state != nixl_service_xfer_state_t::IN_PROGRESS) {
        // Draining (CANCELLING) or already failed: this decode's completion is just
        // freeing its slot, not contributing to the logical READ's progress anymore. The
        // end-of-tick pass in progressService() notices once every slot has drained like
        // this.
        return;
    }

    // READ-receive: verify the decode produced exactly the size that was posted, so a
    // corrupted or mismatched chunk fails the logical request rather than silently landing
    // a wrong-sized write in the destination buffer.
    if (completion_data->size != inbound_req.slotExpectedSizes[slot_index]) {
        inbound_req.state = nixl_service_xfer_state_t::FAILED;
        inbound_req.terminalStatus = NIXL_ERR_UNKNOWN;
        return;
    }
    inbound_req.decodedChunks++;
    auto gen_rrslot_ret = genRRSlot(inbound_req.remoteAgent,
                                    inbound_req.xferId,
                                    slot_index,
                                    inbound_req.slotGenerations[slot_index]);
    if (gen_rrslot_ret != NIXL_SUCCESS) {
        // TODO: handle error.
        NIXL_ASSERT(false);
        return;
    }
    if (inbound_req.decodedChunks == inbound_req.totalChunks) {
        inbound_req.marshalDone = true;
        tryCompleteReadReceive(inbound_req);
    }
}

void
nixlServiceAgentData::tryCompleteReadReceive(inboundXferReqH &inbound_req) {
    if (inbound_req.state != nixl_service_xfer_state_t::IN_PROGRESS) {
        return;
    }
    if (!inbound_req.directDone || !inbound_req.marshalDone) {
        return;
    }
    inbound_req.state = nixl_service_xfer_state_t::DONE;
    if (inbound_req.notif.has_value()) {
        // Deliver to the peer, mirroring where the direct child's notification would have
        // gone had it not been withheld at post time (see postXferReq): the base nixlAgent
        // always addresses a transfer's notification to remote_agent, regardless of
        // direction.
        auto notif_ret = agent_->genNotif(inbound_req.remoteAgent, *inbound_req.notif);
        if (notif_ret != NIXL_SUCCESS) {
            // TODO: handle error.
            NIXL_ASSERT(false);
        }
    }
}

void
nixlServiceAgentData::progressReadReceive(inboundXferReqH &ctx) {
    if (ctx.state == nixl_service_xfer_state_t::IN_PROGRESS && ctx.directChild != nullptr &&
        !ctx.directDone) {
        const nixl_status_t direct_status = agent_->getXferStatus(ctx.directChild);
        if (direct_status == NIXL_IN_PROG) {
            return;
        }
        if (direct_status != NIXL_SUCCESS) {
            // The direct sub-transfer failed: that becomes the logical READ's terminal
            // error.
            ctx.state = nixl_service_xfer_state_t::FAILED;
            ctx.terminalStatus = direct_status;
            return;
        }
        ctx.directDone = true;
    }
    // Whichever of directDone/marshalDone becomes true last finalizes the request and
    // delivers the notification; this may be that moment (a no-op otherwise, including for
    // a marshal-only READ, which reaches DONE via pollInboundSlotCompletion instead).
    tryCompleteReadReceive(ctx);
}

std::optional<std::list<slotT>>
nixlServiceAgentData::allocateSlotGroup() {
    std::list<slotT> slot_group;
    // TODO-Eyal: fix once slots are RAII.
    auto cleanup = makeSlotGroupCleanup(slot_group);

    auto find_slot = [&]() -> std::optional<slotT> {
        for (auto &pool : localStagingPools_) {
            auto slot = pool.allocateSlot();
            if (slot.has_value()) {
                return slot;
            }
        }
        return std::nullopt;
    };

    for (size_t i = 0; i < slots_per_xfer; ++i) {
        auto pending_slot = find_slot();
        if (!pending_slot.has_value()) {
            return std::nullopt;
        }
        slot_group.push_back(*pending_slot);
    }
    std::move(cleanup).Cancel();
    return slot_group;
}

void
nixlServiceAgentData::freeSlotGroup(
    const std::array<slotWorkItem, slots_per_xfer> &slot_group) noexcept {
    for (const auto &slot : slot_group) {
        slot.slot.pool->freeSlot(slot.slot);
    }
}

void
nixlServiceAgentData::serviceNotifCallback(const std::string &sender_agent,
                                           const nixl_blob_t &notif) {
    NIXL_ASSERT(notif.rfind(nixl_s_prefix, 0) == 0);
    if (notif.rfind(nixl_scts_prefix, 0) == 0) {
        ctsCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_srts_prefix, 0) == 0) {
        rtsCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_sposted_prefix, 0) == 0) {
        postedCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_srslot_prefix, 0) == 0) {
        rslotCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_sdelete_prefix, 0) == 0) {
        deleteCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_srreq_prefix, 0) == 0) {
        rreqCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_srposted_prefix, 0) == 0) {
        rpostedCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_srrslot_prefix, 0) == 0) {
        rrslotCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_srabort_prefix, 0) == 0) {
        rabortCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_srabortack_prefix, 0) == 0) {
        rabortAckCallback(sender_agent, notif);
    } else if (notif.rfind(nixl_srnak_prefix, 0) == 0) {
        rnakCallback(sender_agent, notif);
    } else {
        // Unknown _NIXLS_ subtype: either a future/newer message kind this build does not
        // understand yet, or a misbehaving peer. Drop it rather than asserting, so an older
        // service build never crashes on a message it simply doesn't recognize.
        NIXL_WARN << "nixlServiceAgentData: ignoring unknown service message subtype from '"
                  << sender_agent << "'";
    }
}

void
nixlServiceAgentData::rtsCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<rtsNotifPayload>(
        nixl_srts_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::ctsCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<ctsNotifPayload>(
        nixl_scts_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::postedCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<postedNotifPayload>(
        nixl_sposted_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::rslotCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<rslotNotifPayload>(
        nixl_srslot_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::deleteCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<deleteNotifPayload>(
        nixl_sdelete_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::rreqCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<rReqPayload>(nixl_srreq_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::rpostedCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<rPostedPayload>(
        nixl_srposted_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::rrslotCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<rrSlotPayload>(
        nixl_srrslot_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::rabortCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<rAbortPayload>(
        nixl_srabort_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::rabortAckCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<rAbortAckPayload>(
        nixl_srabortack_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::rnakCallback(const std::string &sender_agent, const nixl_blob_t &notif) {
    notifCallbackTemplate<rNakPayload>(nixl_srnak_prefix, sender_agent, notif, serviceNotifQueue_);
}

void
nixlServiceAgentData::handleRTS(const std::string &sender_agent,
                                const rtsNotifPayload &rts_payload) {
    auto &inbound_reqs = inboundXferReqs_.try_emplace(sender_agent).first->second;
    // TODO-Eyal: update check, instead of checking only the last pool.
    if (localStagingPools_.size() == 0 ||
        localStagingPools_.back().getSlotSize() != rts_payload.slotSize) {
        // TODO-Eyal: send back a bad request notif.
        NIXL_ASSERT(false);
        return;
    }
    auto slot_group_opt = allocateSlotGroup();
    if (!slot_group_opt.has_value()) {
        // TODO-Eyal: send back error notif.
        NIXL_ASSERT(false);
        return;
    }
    auto slot_group = std::move(slot_group_opt).value();
    // TODO-Eyal: fix once slots are RAII.
    auto cleanup = makeSlotGroupCleanup(slot_group);
    auto slot_work_items = slotGroupToWorkItems(slot_group);
    auto gen_cts_ret = genCTS(sender_agent, rts_payload.xferId, slot_work_items);
    if (gen_cts_ret != NIXL_SUCCESS) {
        // TODO-Eyal: send back error notif.
        NIXL_ASSERT(false);
        return;
    }
    std::move(cleanup).Cancel();
    nixlSerDes serdes;
    auto ret = serdes.importStr(rts_payload.serializedDstList);
    if (ret != NIXL_SUCCESS) {
        // TODO-Eyal: handle error.
        NIXL_ASSERT(false);
        return;
    }

    inbound_reqs[rts_payload.xferId] = std::make_unique<inboundXferReqH>(sender_agent,
                                                                         nixl_xfer_dlist_t(&serdes),
                                                                         slot_work_items,
                                                                         rts_payload.xferId,
                                                                         rts_payload.deltaOptArgs);
    // TODO-Eyal: check return value.
}

void
nixlServiceAgentData::handleCTS(const std::string &sender_agent,
                                const ctsNotifPayload &cts_payload) {
    // TODO-Eyal: handle edge cases where xfer was released while waiting for CTS.
    // (this assertion will fail)
    if (outboundXferReqs_.count(cts_payload.xferId) == 0) {
        // This is a mid-transfer release.
        return;
    }
    NIXL_ASSERT(outboundXferReqs_[cts_payload.xferId]->nonDirectData.has_value());
    auto &req_data = outboundXferReqs_[cts_payload.xferId]->nonDirectData.value();
    NIXL_ASSERT(req_data.state == nixl_service_xfer_state_t::WAIT_CTS);
    req_data.state = nixl_service_xfer_state_t::IN_PROGRESS;
    req_data.remoteSlotDescriptors = cts_payload.receiverSlotDescriptors;
    req_data.remoteSlotMemTypes = cts_payload.receiverMemSpaces;
    req_data.remoteSlotStates = std::array{remote_slot_state_t::FREE, remote_slot_state_t::FREE};
    for (auto i = 0u; i < slots_per_xfer; ++i) {
        activeSlotQueue_.push(activeSlotWorkItem{std::ref(*outboundXferReqs_[cts_payload.xferId]),
                                                 std::ref(req_data.localSlots[i])});
    }
}

void
nixlServiceAgentData::handlePosted(const std::string &sender_agent,
                                   const postedNotifPayload &posted_payload) {
    NIXL_ASSERT(inboundXferReqs_.count(sender_agent) == 1);
    if (inboundXferReqs_[sender_agent].count(posted_payload.xferId) == 0) {
        // This is a mid-transfer release.
        return;
    }
    auto &inbound_req = inboundXferReqs_[sender_agent][posted_payload.xferId];
    NIXL_ASSERT(inbound_req->state == nixl_service_xfer_state_t::IN_PROGRESS);
    auto &slot = inbound_req->localSlots[posted_payload.slotIndex];
    NIXL_ASSERT(slot.state == local_slot_state_t::FREE);
    slotBuffers buffers;
    buffers.src = slot.slot.toRuntimeBuffer();
    // TODO-Eyal: unify buffer structs, we have runtimeBuffer and nixlBasicDesc and slotT.
    buffers.dst.data =
        reinterpret_cast<std::byte *>(inbound_req->dstList[posted_payload.descIndex].addr) +
        posted_payload.chunkIndex * slot.slot.chunkSize;
    // TODO-Eyal: handle chunk size.
    buffers.dst.size = posted_payload.originalSize;
    buffers.dst.space = memSpaceFromNixlMem(inbound_req->dstList.getType());
    auto process_slot_options = slot.slot.getProcessSlotInputOptions();
    if (posted_payload.postedSegments->size() > 1) {
        buffers.src.size = nixlMarshal::marshal_derived_size;
        process_slot_options[nixlMarshal::option_t::CHUNK_DIVISION] =
            nixlMarshal::ChunkDivision::processSlotInput{posted_payload.postedSegments};
    } else {
        // Single segment case, for marshals which don't support chunk division.
        buffers.src.size = posted_payload.postedSegments->back().size;
    }
    if (inbound_req->receiverDeltaRef.has_value()) {
        const auto &delta_ref = *inbound_req->receiverDeltaRef;
        addReferenceOption(process_slot_options,
                           delta_ref.ref,
                           posted_payload.chunkIndex * slot.slot.chunkSize,
                           posted_payload.originalSize,
                           delta_ref.memType,
                           delta_ref.elementSize);
    }
    inbound_req->asyncHandles[posted_payload.slotIndex] =
        backend_->inboundProcessSlot(buffers, posted_payload.metadata, process_slot_options);
    slot.state = local_slot_state_t::BUSY_MARSHAL;
    activeSlotQueue_.push(activeSlotWorkItem{std::ref(*inbound_req), std::ref(slot)});
}

void
nixlServiceAgentData::handleRSlot(const std::string &source_agent,
                                  const rslotNotifPayload &rslot_payload) {
    if (outboundXferReqs_.count(rslot_payload.xferId) == 0) {
        // This is a mid-transfer release.
        return;
    }
    auto &outbound_req = outboundXferReqs_[rslot_payload.xferId];
    NIXL_ASSERT(outbound_req->nonDirectData.has_value());
    auto &req_data = outbound_req->nonDirectData.value();
    NIXL_ASSERT(req_data.state >= nixl_service_xfer_state_t::IN_PROGRESS);
    if (req_data.state == nixl_service_xfer_state_t::DONE) {
        return;
    }
    NIXL_ASSERT(req_data.remoteAgent == source_agent);
    NIXL_ASSERT(req_data.remoteSlotStates[rslot_payload.slotIndex] == remote_slot_state_t::BUSY);
    req_data.rslotsReceived++;
    if (req_data.rslotsReceived == req_data.chunkIterator.getTotalChunks()) {
        req_data.state = nixl_service_xfer_state_t::DONE;
        if (req_data.compressionStatsHandle && req_data.compressionStatsHandle->originalSize > 0) {
            const auto &stats = *req_data.compressionStatsHandle;
            const auto avg_ratio =
                static_cast<double>(stats.compressedSize) / static_cast<double>(stats.originalSize);
            const auto variance =
                std::max(0.0,
                         stats.weightedSumSquaredRatio / static_cast<double>(stats.originalSize) -
                             avg_ratio * avg_ratio);
            NIXL_INFO << "Compression ratio stats for transfer " << rslot_payload.xferId
                      << ": min=" << stats.minRatio << " max=" << stats.maxRatio
                      << " avg=" << avg_ratio << " std=" << std::sqrt(variance);
        }
        auto ret = genDelete(source_agent, rslot_payload.xferId);
        if (ret != NIXL_SUCCESS) {
            // TODO-Eyal: handle error.
            NIXL_ASSERT(false);
            return;
        }
        freeSlotGroup(req_data.localSlots);
        if (!req_data.notifMsg.empty()) {
            auto ret = agent_->genNotif(source_agent, req_data.notifMsg);
            if (ret != NIXL_SUCCESS) {
                // TODO-Eyal: handle error.
                NIXL_ASSERT(false);
                return;
            }
        }
    }
    req_data.remoteSlotStates[rslot_payload.slotIndex] = remote_slot_state_t::FREE;
}

void
nixlServiceAgentData::handleDelete(
    const std::string &sender_agent,
    const deleteNotifPayload &delete_payload,
    std::unordered_map<std::string, std::set<size_t>> &deleted_reqs) {
    NIXL_ASSERT(inboundXferReqs_.count(sender_agent) == 1);
    NIXL_ASSERT(inboundXferReqs_[sender_agent].count(delete_payload.xferId) == 1);
    auto &inbound_req = inboundXferReqs_[sender_agent][delete_payload.xferId];
    inbound_req->markedForDeletion = true;
    deleted_reqs[sender_agent].insert(delete_payload.xferId);
}

void
nixlServiceAgentData::handleRREQ(const std::string &sender_agent, const rReqPayload &rreq_payload) {
    auto send_r_nak = [&](nixl_status_t error_code) {
        rNakPayload rnak_payload(rreq_payload.xferId, static_cast<int32_t>(error_code));
        // TODO: handle a genNotif failure while sending the RNAK itself.
        [[maybe_unused]] auto gen_ret = agent_->genNotif(sender_agent, rnak_payload.serialize());
    };

    if (auto agent_it = readServeReqs_.find(sender_agent);
        agent_it != readServeReqs_.end() && agent_it->second.count(rreq_payload.xferId) != 0) {
        // Duplicate/retried RREQ for a serve already in progress (or done but not yet
        // erased) for this (sender, xferId): silently drop it. Overwriting the existing
        // map entry below would destroy it without returning its slots to the pool first.
        return;
    }

    // TODO: update check, instead of checking only the last pool (same limitation as
    // handleRTS's slotSize check).
    if (localStagingPools_.empty()) {
        send_r_nak(NIXL_ERR_NOT_FOUND);
        return;
    }
    const auto &pool = localStagingPools_.back();
    if (rreq_payload.fingerprint != makeFingerprint(*this, pool)) {
        // slotSize alone (as RTS/CTS relies on) is not a sufficient compatibility check for
        // READ - a mode/algo/chunkedPayloadSize/layout/memType mismatch could still yield the
        // same physical slot size.
        send_r_nak(NIXL_ERR_INVALID_PARAM);
        return;
    }

    nixlSerDes serdes;
    if (serdes.importStr(rreq_payload.serializedSrcList) != NIXL_SUCCESS) {
        send_r_nak(NIXL_ERR_INVALID_PARAM);
        return;
    }
    nixl_xfer_dlist_t src_list(&serdes);

    // Best-effort sanity check on the source region: reject descriptor lists of an
    // unsupported memory type or with a degenerate (zero-length) entry. This is not a real
    // authorization check - there is currently no way to verify that these descriptors
    // actually fall within memory the sender registered, so a served READ is only as safe
    // as the trust relationship between the two agents.
    // TODO: add real registration-based validation once there is a way to query which
    // memory ranges an agent has registered.
    if (src_list.descCount() == 0) {
        send_r_nak(NIXL_ERR_INVALID_PARAM);
        return;
    }
    mem_space_t src_mem_space;
    try {
        src_mem_space = memSpaceFromNixlMem(src_list.getType());
    }
    catch (const std::runtime_error &) {
        send_r_nak(NIXL_ERR_INVALID_PARAM);
        return;
    }
    const auto &supported_mem_spaces = backend_->getSupportedMemSpaces();
    if (std::find(supported_mem_spaces.begin(), supported_mem_spaces.end(), src_mem_space) ==
        supported_mem_spaces.end()) {
        send_r_nak(NIXL_ERR_NOT_SUPPORTED);
        return;
    }
    if (rreq_payload.deltaOptArgs.has_value()) {
        // Delta (including ANS_DELTA) only supports a single descriptor per side, enforced
        // at create time on the initiator (createXferReq). Re-validate here: fillLocalSlot
        // computes each chunk's offset into senderRef as currentChunkLocal * chunkSize,
        // which is only meaningful within a single contiguous reference buffer - a
        // multi-descriptor source would silently alias chunk offsets across descriptors
        // instead of failing loudly.
        if (src_list.descCount() != 1) {
            send_r_nak(NIXL_ERR_NOT_SUPPORTED);
            return;
        }
        // Only these element sizes are supported by the delta XOR kernel; matches the
        // equivalent create-time check on the initiator.
        const auto element_size = rreq_payload.deltaOptArgs->elementSize;
        if (element_size != 1 && element_size != 2 && element_size != 4 && element_size != 8) {
            send_r_nak(NIXL_ERR_INVALID_PARAM);
            return;
        }
    }
    size_t total_chunks = 0;
    for (int i = 0; i < src_list.descCount(); ++i) {
        if (src_list[i].len == 0) {
            send_r_nak(NIXL_ERR_INVALID_PARAM);
            return;
        }
        // src_list.len is initiator-supplied and otherwise unvalidated: reject a length that
        // would overflow the round-up-to-chunkSize computation below. Wrapping could produce
        // a too-small total_chunks (this serve stalls, unnoticed) or a too-large one
        // (chunkIteratorH::operator++ walks currentDesc past descList.descCount(), guarded
        // only by an assert that compiles out in a release build).
        if (src_list[i].len > std::numeric_limits<size_t>::max() - pool.getChunkSize()) {
            send_r_nak(NIXL_ERR_INVALID_PARAM);
            return;
        }
        total_chunks += (src_list[i].len + pool.getChunkSize() - 1) / pool.getChunkSize();
    }

    auto slot_group_opt = allocateSlotGroup();
    if (!slot_group_opt.has_value()) {
        send_r_nak(NIXL_ERR_NOT_FOUND);
        return;
    }
    auto slot_group = std::move(slot_group_opt).value();
    // TODO: fix once slots are RAII.
    auto cleanup = makeSlotGroupCleanup(slot_group);
    auto slot_work_items = slotGroupToWorkItems(slot_group);

    // Build the marshalOptArgs alternative matching this agent's configured mode, carrying
    // the delta sender reference from RREQ when present. RREQ ships the sender's own
    // reference under senderRef; this agent never decodes (it is serving, not receiving),
    // so receiverRef/receiverMemType are left at their defaults.
    const nixl_marshal_opt_args_t serving_opt_args = std::visit(
        overloaded{
            [](const nixlMarshalDirectConfig &) -> nixl_marshal_opt_args_t {
                NIXL_ASSERT(false); // unreachable: a direct-mode agent never has staging pools.
                return nixlMarshalDirectOptArgs{};
            },
            [](const nixlMarshalStagingConfig &) -> nixl_marshal_opt_args_t {
                return nixlMarshalStagingOptArgs{};
            },
            [&](const nixlMarshalDeltaConfig &) -> nixl_marshal_opt_args_t {
                nixlMarshalDeltaOptArgs args;
                if (rreq_payload.deltaOptArgs.has_value()) {
                    args.senderRef = rreq_payload.deltaOptArgs->ref;
                    args.senderMemType = rreq_payload.deltaOptArgs->memType;
                    args.elementSize = rreq_payload.deltaOptArgs->elementSize;
                }
                return args;
            },
            [&](const nixlMarshalCompressConfig &) -> nixl_marshal_opt_args_t {
                nixlMarshalCompressOptArgs args;
                if (rreq_payload.deltaOptArgs.has_value()) {
                    nixlMarshalDeltaOptArgs delta_args;
                    delta_args.senderRef = rreq_payload.deltaOptArgs->ref;
                    delta_args.senderMemType = rreq_payload.deltaOptArgs->memType;
                    delta_args.elementSize = rreq_payload.deltaOptArgs->elementSize;
                    args.delta = delta_args;
                }
                return args;
            },
        },
        mode_);

    auto wrapper = std::make_unique<nixlServiceXferReqH>(
        src_list,
        std::string(), // no RTS is ever sent for a READ-serving request.
        sender_agent,
        slot_work_items,
        rreq_payload.xferId,
        total_chunks,
        serving_opt_args);
    wrapper->op = NIXL_READ;
    // RREQ already carries what RTS+CTS would have conveyed separately for a WRITE, so the
    // serving push starts directly at IN_PROGRESS - there is no CTS wait for READ.
    auto &serving_req_data = wrapper->nonDirectData.value();
    serving_req_data.state = nixl_service_xfer_state_t::IN_PROGRESS;
    serving_req_data.remoteSlotDescriptors = rreq_payload.recvSlotDescriptors;
    serving_req_data.remoteSlotMemTypes = rreq_payload.recvMemSpaces;
    serving_req_data.remoteSlotStates =
        std::array{remote_slot_state_t::FREE, remote_slot_state_t::FREE};
    std::move(cleanup).Cancel();

    auto &stored_reqs = readServeReqs_[sender_agent];
    stored_reqs[rreq_payload.xferId] = std::move(wrapper);
    auto &stored_req = *stored_reqs[rreq_payload.xferId];
    for (auto i = 0u; i < slots_per_xfer; ++i) {
        activeSlotQueue_.push(activeSlotWorkItem{
            std::ref(stored_req), std::ref(stored_req.nonDirectData->localSlots[i])});
    }
}

void
nixlServiceAgentData::handleRPosted(const std::string &sender_agent,
                                    const rPostedPayload &rposted_payload) {
    if (readReceiveReqs_.count(rposted_payload.xferId) == 0) {
        // Mid-transfer release, or a stale message for an already-finished/aborted READ.
        return;
    }
    auto &inbound_req = *readReceiveReqs_[rposted_payload.xferId];
    if (sender_agent != inbound_req.remoteAgent) {
        // readReceiveReqs_ is keyed only by this agent's own xfer id (unlike
        // inboundXferReqs_, which nests by sender), so this check is the only thing
        // preventing an unrelated agent from injecting data into this READ under a
        // guessed/colliding xfer id.
        NIXL_WARN << "nixlServiceAgentData: ignoring RPOSTED for xfer " << rposted_payload.xferId
                  << " from unexpected sender '" << sender_agent << "' (expected '"
                  << inbound_req.remoteAgent << "')";
        return;
    }
    if (rposted_payload.slotIndex >= slots_per_xfer) {
        NIXL_WARN << "nixlServiceAgentData: ignoring RPOSTED for xfer " << rposted_payload.xferId
                  << " with out-of-range slot index " << rposted_payload.slotIndex;
        return;
    }
    if (inbound_req.state != nixl_service_xfer_state_t::IN_PROGRESS) {
        // The request already reached a terminal state (a codec-integrity failure on a
        // sibling slot, or a cancellation) - this slot's async decode may still be
        // legitimately in flight, but no further decodes should be started for it.
        return;
    }
    auto &slot = inbound_req.localSlots[rposted_payload.slotIndex];
    if (slot.state != local_slot_state_t::FREE) {
        // Under normal operation the peer never sends a second RPOSTED for this slot
        // before this agent acks the first with RRSLOT (see trySend's remoteSlotStates
        // gating) - reaching this with a busy slot means a duplicate/replayed RPOSTED for
        // a fill already in flight. Drop it rather than starting a second decode into (or
        // clobbering the in-flight handle of) the same slot.
        NIXL_WARN << "nixlServiceAgentData: ignoring RPOSTED for xfer " << rposted_payload.xferId
                  << " - slot " << rposted_payload.slotIndex << " is not free";
        return;
    }

    const auto dst_desc = resolveAbsoluteOffset(
        inbound_req.dstList, rposted_payload.dstByteOffset, rposted_payload.originalSize);
    if (!dst_desc.has_value()) {
        NIXL_WARN << "nixlServiceAgentData: ignoring RPOSTED for xfer " << rposted_payload.xferId
                  << " with out-of-bounds destination offset " << rposted_payload.dstByteOffset
                  << " (size " << rposted_payload.originalSize << ")";
        return;
    }

    slotBuffers buffers;
    buffers.src = slot.slot.toRuntimeBuffer();
    buffers.dst.data = reinterpret_cast<std::byte *>(dst_desc->addr);
    buffers.dst.size = rposted_payload.originalSize;
    buffers.dst.space = memSpaceFromNixlMem(inbound_req.dstList.getType());
    auto process_slot_options = slot.slot.getProcessSlotInputOptions();
    if (rposted_payload.postedSegments->empty()) {
        NIXL_WARN << "nixlServiceAgentData: ignoring RPOSTED for xfer " << rposted_payload.xferId
                  << " with no posted segments";
        return;
    }
    if (rposted_payload.postedSegments->size() > 1) {
        buffers.src.size = nixlMarshal::marshal_derived_size;
        process_slot_options[nixlMarshal::option_t::CHUNK_DIVISION] =
            nixlMarshal::ChunkDivision::processSlotInput{rposted_payload.postedSegments};
    } else {
        // Single segment case, for marshals which don't support chunk division.
        buffers.src.size = rposted_payload.postedSegments->back().size;
    }
    if (inbound_req.receiverDeltaRef.has_value()) {
        const auto &delta_ref = *inbound_req.receiverDeltaRef;
        addReferenceOption(process_slot_options,
                           delta_ref.ref,
                           rposted_payload.dstByteOffset,
                           rposted_payload.originalSize,
                           delta_ref.memType,
                           delta_ref.elementSize);
    }
    inbound_req.slotExpectedSizes[rposted_payload.slotIndex] = rposted_payload.originalSize;
    // Bump this slot's generation to mark it as (re)filled - the matching RRSLOT sent once
    // this decode completes carries this same value, letting the peer's trySend() (see
    // remoteSlotGenerations) tell this fill's ack apart from a stale/duplicate one for an
    // earlier fill of the same slot.
    ++inbound_req.slotGenerations[rposted_payload.slotIndex];
    inbound_req.asyncHandles[rposted_payload.slotIndex] =
        backend_->inboundProcessSlot(buffers, rposted_payload.metadata, process_slot_options);
    slot.state = local_slot_state_t::BUSY_MARSHAL;
    activeSlotQueue_.push(activeSlotWorkItem{std::ref(inbound_req), std::ref(slot)});
}

void
nixlServiceAgentData::handleRRSlot(const std::string &source_agent,
                                   const rrSlotPayload &rrslot_payload) {
    auto agent_it = readServeReqs_.find(source_agent);
    if (agent_it == readServeReqs_.end() || agent_it->second.count(rrslot_payload.xferId) == 0) {
        // Mid-transfer release, e.g. this agent already aborted/cleaned up the serving req.
        return;
    }
    auto &serving_req = *agent_it->second[rrslot_payload.xferId];
    NIXL_ASSERT(serving_req.nonDirectData.has_value());
    auto &req_data = serving_req.nonDirectData.value();
    NIXL_ASSERT(req_data.state >= nixl_service_xfer_state_t::IN_PROGRESS);
    if (req_data.state == nixl_service_xfer_state_t::DONE ||
        req_data.state == nixl_service_xfer_state_t::CANCELLING) {
        // Already finished, or being drained after an RABORT: rslotsReceived stops
        // mattering once cancelling - the drain-completion check in progressService() only
        // looks at local slot state, so a stale/duplicate RRSLOT here is simply ignored
        // rather than risking a race with that check.
        return;
    }
    NIXL_ASSERT(req_data.remoteAgent == source_agent);
    if (rrslot_payload.slotIndex >= slots_per_xfer) {
        NIXL_WARN << "nixlServiceAgentData: ignoring RRSLOT for xfer " << rrslot_payload.xferId
                  << " with out-of-range slot index " << rrslot_payload.slotIndex;
        return;
    }
    if (req_data.remoteSlotStates[rrslot_payload.slotIndex] != remote_slot_state_t::BUSY ||
        rrslot_payload.slotGeneration != req_data.remoteSlotGenerations[rrslot_payload.slotIndex]) {
        // Stale/duplicate ack: either this slot is not currently awaiting one (already
        // freed by an earlier, legitimate ack for it), or it echoes an older generation
        // than the fill currently in flight (superseded by a newer RPOSTED already sent
        // for this slot). Either way, drop it silently rather than corrupting
        // remoteSlotStates/rslotsReceived.
        return;
    }
    req_data.rslotsReceived++;
    if (req_data.rslotsReceived == req_data.chunkIterator.getTotalChunks()) {
        req_data.state = nixl_service_xfer_state_t::DONE;
        // Unlike handleRSlot (WRITE), there is no genDelete/user notification here: the
        // initiator holds the user handle and is responsible for delivering it once its own
        // receive side also completes (see tryCompleteReadReceive). Freeing this request's
        // slots must wait for every one of them to actually go idle - progressService()'s
        // end-of-tick scan of readServeReqs_ handles that once this state change is visible.
    } else {
        req_data.remoteSlotStates[rrslot_payload.slotIndex] = remote_slot_state_t::FREE;
    }
}

void
nixlServiceAgentData::handleRAbort(const std::string &sender_agent,
                                   const rAbortPayload &rabort_payload) {
    auto agent_it = readServeReqs_.find(sender_agent);
    if (agent_it == readServeReqs_.end() || agent_it->second.count(rabort_payload.xferId) == 0) {
        // This agent is not (or no longer) serving this READ - either the RREQ was itself
        // rejected via RNAK, or the serve already finished and was erased. Either way, it
        // is not touching the initiator's slots, so it is safe - and necessary, to unblock
        // the initiator's own drain - to acknowledge immediately.
        [[maybe_unused]] auto ret = genRAbortAck(sender_agent, rabort_payload.xferId);
        return;
    }
    auto &serving_req = *agent_it->second[rabort_payload.xferId];
    NIXL_ASSERT(serving_req.nonDirectData.has_value());
    auto &req_data = serving_req.nonDirectData.value();
    if (req_data.state == nixl_service_xfer_state_t::CANCELLING) {
        // Duplicate/retried RABORT while already draining; the end-of-tick pass in
        // progressService() will acknowledge once fully drained.
        return;
    }
    if (req_data.state == nixl_service_xfer_state_t::DONE) {
        // Already fully served (rslotsReceived == totalChunks) and pending a fully_drained
        // erase by progressService()'s end-of-tick scan; not touching the initiator's slots
        // anymore either.
        [[maybe_unused]] auto ret = genRAbortAck(sender_agent, rabort_payload.xferId);
        return;
    }
    NIXL_ASSERT(req_data.state == nixl_service_xfer_state_t::IN_PROGRESS);
    req_data.state = nixl_service_xfer_state_t::CANCELLING;
}

void
nixlServiceAgentData::handleRAbortAck(const std::string &sender_agent,
                                      const rAbortAckPayload &raback_payload) {
    if (readReceiveReqs_.count(raback_payload.xferId) == 0) {
        // Already finalized, or a stale/duplicate ack; nothing to do.
        return;
    }
    auto &receive_req = *readReceiveReqs_[raback_payload.xferId];
    if (sender_agent != receive_req.remoteAgent) {
        // readReceiveReqs_ is keyed only by this agent's own xfer id (see handleRPosted for
        // the same concern), so this check is the only thing preventing an unrelated agent
        // from acknowledging (and so prematurely unblocking the drain of) a READ under a
        // guessed/colliding xfer id.
        NIXL_WARN << "nixlServiceAgentData: ignoring RABORT_ACK for xfer " << raback_payload.xferId
                  << " from unexpected sender '" << sender_agent << "' (expected '"
                  << receive_req.remoteAgent << "')";
        return;
    }
    if (receive_req.state != nixl_service_xfer_state_t::CANCELLING) {
        // Not currently draining (e.g. a stale/duplicate ack after finalization, or one
        // that arrived before this agent ever sent an RABORT); nothing to do.
        return;
    }
    // The peer has confirmed it is no longer touching this agent's receive slots. Any of
    // this agent's own in-flight decodes still need to drain first, though (see
    // pollInboundSlotCompletion); the end-of-tick scan in progressService() notices once
    // every local slot has also gone idle and finalizes.
    receive_req.remoteQuiesced = true;
}

void
nixlServiceAgentData::handleRNak(const std::string &sender_agent, const rNakPayload &rnak_payload) {
    if (readReceiveReqs_.count(rnak_payload.xferId) == 0) {
        // Already finalized/released, or a stale/duplicate NAK; nothing to do.
        return;
    }
    auto &receive_req = *readReceiveReqs_[rnak_payload.xferId];
    if (sender_agent != receive_req.remoteAgent) {
        // readReceiveReqs_ is keyed only by this agent's own xfer id (see handleRPosted for
        // the same concern), so this check is the only thing preventing an unrelated agent
        // from failing a READ under a guessed/colliding xfer id.
        NIXL_WARN << "nixlServiceAgentData: ignoring RNAK for xfer " << rnak_payload.xferId
                  << " from unexpected sender '" << sender_agent << "' (expected '"
                  << receive_req.remoteAgent << "')";
        return;
    }
    if (receive_req.state != nixl_service_xfer_state_t::IN_PROGRESS) {
        // Not awaiting an RREQ response anymore (a stale/duplicate NAK, or one racing with
        // a mid-transfer release that already moved this request to CANCELLING); nothing
        // to do.
        return;
    }
    // The peer never admitted this READ, so it holds no serving state for it (no
    // readServeReqs_ entry was ever created) - there is nothing to abort or drain on its
    // side. This request's marshal sub-part will never happen; fail it outright.
    receive_req.state = nixl_service_xfer_state_t::FAILED;
    // Only trust the specific codes a conforming peer's send_r_nak() can emit; anything else
    // (e.g. a malformed or adversarial NIXL_SUCCESS) would otherwise make a failed READ
    // report success via getXferStatus.
    switch (rnak_payload.errorCode) {
    case NIXL_ERR_INVALID_PARAM:
    case NIXL_ERR_NOT_FOUND:
    case NIXL_ERR_NOT_SUPPORTED:
        receive_req.terminalStatus = static_cast<nixl_status_t>(rnak_payload.errorCode);
        break;
    default:
        NIXL_WARN << "nixlServiceAgentData: RNAK for xfer " << rnak_payload.xferId
                  << " carried unexpected errorCode " << rnak_payload.errorCode;
        receive_req.terminalStatus = NIXL_ERR_BACKEND;
        break;
    }
}

nixl_status_t
nixlServiceAgentData::genDelete(const std::string &sender_agent, size_t xfer_id) {
    deleteNotifPayload delete_payload(xfer_id);
    return agent_->genNotif(sender_agent, delete_payload.serialize());
}

nixl_status_t
nixlServiceAgentData::genRTS(nixlServiceXferReqH &xfer_req) {
    NIXL_ASSERT(xfer_req.nonDirectData.has_value());
    NIXL_ASSERT(outboundXferReqs_.count(xfer_req.nonDirectData->xferId) == 1);
    NIXL_ASSERT(xfer_req.nonDirectData->localSlots[0].slot.slotSize ==
                xfer_req.nonDirectData->localSlots[1].slot.slotSize);
    rtsNotifPayload rts_payload(xfer_req.nonDirectData->xferId,
                                xfer_req.nonDirectData->serializedDstDescList,
                                xfer_req.nonDirectData->localSlots[0].slot.slotSize);
    if (const auto *delta_opt_args = getDeltaOptArgs(xfer_req.marshalOptArgs)) {
        rts_payload.deltaOptArgs = nixlMarshalDeltaReceiverRefArgs{delta_opt_args->receiverRef,
                                                                   delta_opt_args->receiverMemType,
                                                                   delta_opt_args->elementSize};
    }
    return agent_->genNotif(xfer_req.nonDirectData->remoteAgent, rts_payload.serialize());
}

nixl_status_t
nixlServiceAgentData::genRREQ(nixlServiceXferReqH &xfer_req) {
    NIXL_ASSERT(xfer_req.readReceiveXferId.has_value());
    const auto xfer_id = *xfer_req.readReceiveXferId;
    NIXL_ASSERT(readReceiveReqs_.count(xfer_id) == 1);
    auto &receive_req = *readReceiveReqs_[xfer_id];
    NIXL_ASSERT(receive_req.localSlots[0].slot.slotSize == receive_req.localSlots[1].slot.slotSize);
    NIXL_ASSERT(receive_req.localSlots[0].slot.pool != nullptr);

    rReqPayload rreq_payload(
        xfer_id,
        receive_req.serializedSrcList,
        // TODO: make this invariant to slots_per_xfer > 2 (same TODO as genCTS).
        std::array{receive_req.localSlots[0].slot.toDesc(receive_req.localSlots[0].slot.slotSize),
                   receive_req.localSlots[1].slot.toDesc(receive_req.localSlots[1].slot.slotSize)},
        std::array{memSpaceFromNixlMem(receive_req.localSlots[0].slot.type),
                   memSpaceFromNixlMem(receive_req.localSlots[1].slot.type)},
        makeFingerprint(*this, *receive_req.localSlots[0].slot.pool));
    if (const auto *delta_opt_args = getDeltaOptArgs(xfer_req.marshalOptArgs)) {
        // RREQ ships this agent's own reference (it is the one encoding), not the peer's
        // receiverRef - the sender and receiver roles are swapped relative to WRITE's RTS.
        rreq_payload.deltaOptArgs = nixlMarshalDeltaSenderRefArgs{
            delta_opt_args->senderRef, delta_opt_args->senderMemType, delta_opt_args->elementSize};
    }
    return agent_->genNotif(receive_req.remoteAgent, rreq_payload.serialize());
}

nixl_status_t
nixlServiceAgentData::genCTS(const std::string &remote_agent,
                             size_t xfer_id,
                             const std::array<slotWorkItem, slots_per_xfer> &slots) {
    ctsNotifPayload cts_payload(xfer_id,
                                // TODO-Eyal: make this invariant to slots_per_xfer > 2.
                                std::array{slots[0].slot.toDesc(slots[0].slot.slotSize),
                                           slots[1].slot.toDesc(slots[1].slot.slotSize)},
                                std::array{memSpaceFromNixlMem(slots[0].slot.type),
                                           memSpaceFromNixlMem(slots[1].slot.type)});
    return agent_->genNotif(remote_agent, cts_payload.serialize());
}

nixl_status_t
nixlServiceAgentData::genRSlot(const std::string &remote_agent, size_t xfer_id, size_t slot_index) {
    // TODO: implement Ready Slot notification format and slot_index payload handling.
    rslotNotifPayload rslot_payload(xfer_id, slot_index);
    return agent_->genNotif(remote_agent, rslot_payload.serialize());
}

nixl_status_t
nixlServiceAgentData::genRRSlot(const std::string &remote_agent,
                                size_t xfer_id,
                                size_t slot_index,
                                uint64_t slot_generation) {
    rrSlotPayload rrslot_payload(xfer_id, slot_index, slot_generation);
    return agent_->genNotif(remote_agent, rrslot_payload.serialize());
}

nixl_status_t
nixlServiceAgentData::genRAbort(const std::string &remote_agent, size_t xfer_id) {
    rAbortPayload rabort_payload(xfer_id);
    return agent_->genNotif(remote_agent, rabort_payload.serialize());
}

nixl_status_t
nixlServiceAgentData::genRAbortAck(const std::string &remote_agent, size_t xfer_id) {
    rAbortAckPayload raback_payload(xfer_id);
    return agent_->genNotif(remote_agent, raback_payload.serialize());
}

namespace nixlService {

size_t
recommendServiceMemSize(const nixl_marshal_config_t &mode, uint32_t max_concurrent_transfers) {
    if (std::holds_alternative<nixlMarshalDirectConfig>(mode)) {
        throw std::invalid_argument("Direct mode does not use service memory");
    }
    if (max_concurrent_transfers == 0) {
        throw std::invalid_argument("maxConcurrentTransfers must be at least 1");
    }
    if (std::holds_alternative<nixlMarshalStagingConfig>(mode)) {
        return stagingBackend::recommendServiceMemSize(default_chunked_payload_size,
                                                       max_concurrent_transfers);
    }
    if (std::holds_alternative<nixlMarshalDeltaConfig>(mode)) {
        return deltaBackend::recommendServiceMemSize(default_chunked_payload_size,
                                                     max_concurrent_transfers);
    }
    if (std::holds_alternative<nixlMarshalCompressConfig>(mode)) {
#ifdef NIXL_HAVE_NVCOMP
        return compressionBackend::recommendServiceMemSize(
            default_chunked_payload_size,
            max_concurrent_transfers,
            std::get<nixlMarshalCompressConfig>(mode).algo);
#else
        throw std::invalid_argument("Compression marshal backend requires nvCOMP support");
#endif
    }
    throw std::invalid_argument("Unknown mode");
}

} // namespace nixlService

std::pair<nixlServiceAgentConfig, std::shared_ptr<nixlServiceAgentData>>
nixlServiceAgent::prepare(nixlServiceAgentConfig cfg) {
    return prepare(std::move(cfg), default_chunked_payload_size);
}

std::pair<nixlServiceAgentConfig, std::shared_ptr<nixlServiceAgentData>>
nixlServiceAgent::prepare(nixlServiceAgentConfig cfg, size_t chunked_payload_size) {
    auto data = std::make_shared<nixlServiceAgentData>(cfg.mode, chunked_payload_size);

    cfg.notifCallbacks_[nixl_s_prefix] =
        [data_weak = std::weak_ptr<nixlServiceAgentData>(data)](nixlNotifCallbackArgs &&args) {
            auto d = data_weak.lock();
            if (!d) {
                return;
            }
            d->serviceNotifCallback(args.remote_agent, args.raw_notif);
        };

    return {std::move(cfg), std::move(data)};
}

nixlServiceAgent::nixlServiceAgent(
    const std::string &name,
    std::pair<nixlServiceAgentConfig, std::shared_ptr<nixlServiceAgentData>> &&tag)
    : nixlAgent(name, tag.first),
      data_(std::move(tag.second)) {
    data_->agent_ = this;
    data_->localAgentName_ = name;
}

nixlServiceAgent::nixlServiceAgent(const std::string &name, nixlServiceAgentConfig cfg)
    : nixlServiceAgent(name, prepare(std::move(cfg))) {}

nixlServiceAgent::~nixlServiceAgent() = default;

nixl_status_t
nixlServiceAgent::registerServiceMem(const nixl_reg_dlist_t &descs,
                                     const nixl_opt_args_t * /*extra_params*/) {
    return std::visit(
        overloaded{
            [this, &descs](const nixlMarshalDirectConfig &) -> nixl_status_t {
                return NIXL_ERR_NOT_SUPPORTED;
            },
            [this, &descs](const auto & /*non_direct_mode*/) -> nixl_status_t {
                auto mem_space = memSpaceFromNixlMem(descs.getType());
                auto supported_mem_spaces = data_->backend_->getSupportedMemSpaces();
                if (std::find(supported_mem_spaces.begin(),
                              supported_mem_spaces.end(),
                              mem_space) == supported_mem_spaces.end()) {
                    return NIXL_ERR_NOT_SUPPORTED;
                }

                auto const ret = nixlAgent::registerMem(descs);
                if (ret != NIXL_SUCCESS) {
                    return ret;
                }

                const auto chunked_payload_size = data_->chunkedPayloadSize_;
                size_t slot_workspace_size = 0;
                auto it_workspace = data_->marshalSlotMemoryRequirements_.opts.find(
                    option_t::WRITEABLE_WORKSPACE_MEMORY);
                if (it_workspace != data_->marshalSlotMemoryRequirements_.opts.end()) {
                    if (auto *ws_req = std::get_if<WriteableWorkspaceMemory::memoryRequirements>(
                            &it_workspace->second)) {
                        slot_workspace_size = ws_req->slotWorkspaceSize;
                    }
                }

                size_t slot_overhead_size = 0;
                auto it_overhead =
                    data_->marshalSlotMemoryRequirements_.opts.find(option_t::SLOT_OVERHEAD);
                if (it_overhead != data_->marshalSlotMemoryRequirements_.opts.end()) {
                    if (auto *oh_req =
                            std::get_if<SlotOverhead::memoryRequirements>(&it_overhead->second)) {
                        slot_overhead_size = oh_req->slotOverheadSize;
                    }
                }
                const auto marshal_overhead = slot_overhead_size + slot_workspace_size;

                for (const auto &desc : descs) {
                    /* `chunkedPayloadSize` is the *payload* commitment: the service agent
                     *  guarantees it will never feed more than that many bytes of source
                     *  into a single slot. The marshal layer can append up to
                     *  `marshalOverhead` bytes on top of the payload, so the physical
                     *  staging slot must reserve `chunkedPayloadSize + marshalOverhead`
                     *  bytes.
                     *
                     *  We round numSlots DOWN, because a partial slot has no usable
                     *  destination memory.
                     *
                     *  Example: desc.len = 1 GB, chunkedPayloadSize = 128 MB, marshalOverhead = 5
                     * MB physicalSlotSize    = 128 + 5           = 133 MB numSlots            =
                     * 1024 / 133        = 7 slots payload per slot    = 128 MB            (enforced
                     * at fill time) total source moved  = 7 * 128           = 896 MB total staging
                     * used  = 7 * 133           = 931 MB
                     */
                    const auto raw_physical_slot_size = chunked_payload_size + marshal_overhead;
                    const auto slot_stride = alignUp(raw_physical_slot_size,
                                                     MarshalBackendSizing::slot_stride_alignment);
                    const auto aligned_base =
                        alignUp(desc.addr, MarshalBackendSizing::slot_stride_alignment);
                    if (aligned_base >= desc.addr && aligned_base - desc.addr > desc.len) {
                        return NIXL_ERR_INVALID_PARAM;
                    }
                    const auto leading_slop = aligned_base - desc.addr;
                    const auto usable_bytes = desc.len - leading_slop;
                    const auto num_slots = usable_bytes / slot_stride;
                    if (num_slots < slots_per_xfer) {
                        return NIXL_ERR_INVALID_PARAM;
                    }
                    data_->localStagingPools_.emplace_back(aligned_base,
                                                           slot_stride,
                                                           num_slots,
                                                           chunked_payload_size,
                                                           descs.getType(),
                                                           slot_workspace_size);
                }

                return NIXL_SUCCESS;
            },
        },
        data_->mode_);
}

nixl_status_t
nixlServiceAgent::deregisterServiceMem(const nixl_reg_dlist_t &descs,
                                       const nixl_opt_args_t * /*extra_params*/) {
    return std::visit(overloaded{
                          [this, &descs](const nixlMarshalDirectConfig &) -> nixl_status_t {
                              return NIXL_ERR_NOT_SUPPORTED;
                          },
                          [this, &descs](const auto &non_direct_mode) -> nixl_status_t {
                              auto const ret = nixlAgent::deregisterMem(descs);
                              if (ret != NIXL_SUCCESS) {
                                  return ret;
                              }

                              // TOOD-Eyal: improve if we want to support mid-request
                              // deregistration.
                              for (const auto &desc : descs) {
                                  data_->localStagingPools_.remove_if(
                                      [desc](const slotPool &pool) -> bool {
                                          return pool.getBaseAddr() == desc.addr;
                                      });
                              }

                              return NIXL_SUCCESS;
                          },
                      },
                      data_->mode_);
}

nixl_status_t
nixlServiceAgent::createXferReq(const nixl_xfer_op_t &operation,
                                const nixl_xfer_dlist_t &local_descs,
                                const nixl_xfer_dlist_t &remote_descs,
                                const std::string &remote_agent,
                                nixlServiceXferReqH *&req_hndl,
                                const nixl_service_opt_args_t *extra_params) {
    // nixl_xfer_op_t only has two named values (NIXL_READ, NIXL_WRITE), but as a plain enum
    // it accepts any integer via a static_cast, e.g. from uninitialized or garbage input.
    // The non-direct path below assumes operation is one of the two (see e.g. the
    // NIXL_ASSERT(operation == NIXL_READ) further down), so reject anything else here up
    // front, making that assert an invariant rather than something a caller could trip.
    // This also covers postXferReq/getXferStatus/releaseXferReq, since they all act on the
    // handle this validates.
    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto marshal_opt_args = getValidMarshalOptArgs(extra_params, data_->mode_);
    if (!marshal_opt_args.has_value()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto create_non_direct_xfer =
        [&](const nixl_marshal_opt_args_t &opt_args,
            const nixl_xfer_dlist_t &marshal_local_descs,
            const nixl_xfer_dlist_t &marshal_remote_descs) -> nixl_status_t {
        if (data_->localStagingPools_.empty()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        auto total_chunks =
            countChunksAndVerifyMatch(marshal_local_descs,
                                      marshal_remote_descs,
                                      data_->localStagingPools_.begin()->getChunkSize());
        if (total_chunks == -1) {
            return NIXL_ERR_INVALID_PARAM;
        }
        auto slot_group_opt = data_->allocateSlotGroup();
        if (!slot_group_opt.has_value()) {
            return NIXL_ERR_NOT_FOUND;
        }
        auto slot_group = std::move(slot_group_opt).value();
        // TODO-Eyal: fix once slots are RAII.
        auto cleanup = makeSlotGroupCleanup(slot_group);
        nixlSerDes serdes;
        const nixl_status_t serialize_ret = marshal_remote_descs.serialize(&serdes);
        if (serialize_ret != NIXL_SUCCESS) {
            return serialize_ret;
        }
        auto wrapper = std::make_unique<nixlServiceXferReqH>(marshal_local_descs,
                                                             serdes.exportStr(),
                                                             remote_agent,
                                                             slotGroupToWorkItems(slot_group),
                                                             data_->nextOutboundXferId_++,
                                                             total_chunks,
                                                             opt_args);
        wrapper->op = operation;
        std::move(cleanup).Cancel();
        req_hndl = wrapper.get();
        data_->outboundXferReqs_[req_hndl->nonDirectData->xferId] = std::move(wrapper);
        return NIXL_SUCCESS;
    };

    // READ counterpart of createNonDirectXfer(): builds a receive context in
    // readReceiveReqs_ (this agent is the sink/initiator) instead of an outbound-sender
    // context in outboundXferReqs_. Unlike the WRITE handle, the returned
    // nixlServiceXferReqH is not itself map-owned: it is released directly to the caller
    // (mirroring the direct-only case below), and only carries the id needed to look up the
    // service-owned receive context.
    auto create_read_receive_xfer =
        [&](const nixl_marshal_opt_args_t &opt_args,
            const nixl_xfer_dlist_t &marshal_dst_descs,
            const nixl_xfer_dlist_t &marshal_src_descs,
            nixlXferReqH *direct_child,
            std::optional<nixlMarshalDeltaReceiverRefArgs> receiver_delta_ref) -> nixl_status_t {
        if (data_->localStagingPools_.empty()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        auto total_chunks =
            countChunksAndVerifyMatch(marshal_dst_descs,
                                      marshal_src_descs,
                                      data_->localStagingPools_.begin()->getChunkSize());
        if (total_chunks == -1) {
            return NIXL_ERR_INVALID_PARAM;
        }
        auto slot_group_opt = data_->allocateSlotGroup();
        if (!slot_group_opt.has_value()) {
            return NIXL_ERR_NOT_FOUND;
        }
        auto slot_group = std::move(slot_group_opt).value();
        // TODO: fix once slots are RAII.
        auto cleanup = makeSlotGroupCleanup(slot_group);
        // The SOURCE list (the peer's address space) is what genRREQ() must later send to
        // the peer; the DESTINATION list (marshal_dst_descs, this agent's own memory) is
        // kept locally and never serialized.
        nixlSerDes serdes;
        const nixl_status_t serialize_ret = marshal_src_descs.serialize(&serdes);
        if (serialize_ret != NIXL_SUCCESS) {
            return serialize_ret;
        }
        const auto xfer_id = data_->nextOutboundXferId_++;
        auto receive_req =
            std::make_unique<inboundXferReqH>(remote_agent,
                                              nixl_xfer_dlist_t(marshal_dst_descs),
                                              slotGroupToWorkItems(slot_group),
                                              xfer_id,
                                              static_cast<size_t>(total_chunks),
                                              serdes.exportStr(),
                                              direct_child,
                                              resolveReadNotif(std::nullopt, extra_params),
                                              receiver_delta_ref);
        std::move(cleanup).Cancel();
        data_->readReceiveReqs_[xfer_id] = std::move(receive_req);

        auto wrapper = std::make_unique<nixlServiceXferReqH>();
        wrapper->marshalOptArgs = opt_args;
        wrapper->op = NIXL_READ;
        wrapper->readReceiveXferId = xfer_id;
        req_hndl = wrapper.release();
        return NIXL_SUCCESS;
    };

    return std::visit(
        overloaded{
            [&](const nixlMarshalDirectOptArgs &) -> nixl_status_t {
                auto wrapper = std::make_unique<nixlServiceXferReqH>();
                wrapper->op = operation;
                const nixl_status_t ret = nixlAgent::createXferReq(operation,
                                                                   local_descs,
                                                                   remote_descs,
                                                                   remote_agent,
                                                                   wrapper->xferReq,
                                                                   extra_params);
                if (ret != NIXL_SUCCESS) {
                    return ret;
                }

                req_hndl = wrapper.release();
                return ret;
            },
            [&](const auto &non_direct_mode_opt_args) -> nixl_status_t {
                const nixl_marshal_opt_args_t marshal_opt_args_variant = non_direct_mode_opt_args;
                if (const auto *delta_opt_args = getDeltaOptArgs(marshal_opt_args_variant)) {
                    if (delta_opt_args->senderRef == nullptr ||
                        delta_opt_args->receiverRef == nullptr) {
                        return NIXL_ERR_INVALID_PARAM;
                    }
                    if (local_descs.descCount() != 1 || remote_descs.descCount() != 1) {
                        return NIXL_ERR_NOT_SUPPORTED;
                    }
                    if (delta_opt_args->elementSize != 1 && delta_opt_args->elementSize != 2 &&
                        delta_opt_args->elementSize != 4 && delta_opt_args->elementSize != 8) {
                        // Only these element sizes are supported by the delta XOR kernel;
                        // validate here rather than failing later at GPU submit time.
                        return NIXL_ERR_INVALID_PARAM;
                    }
                }
                auto split = splitSmallDescPairs(local_descs, remote_descs);
                if (!split.valid) {
                    return NIXL_ERR_INVALID_PARAM;
                }
                if (operation == NIXL_READ && !split.marshalLocal.isEmpty() &&
                    remote_agent == data_->localAgentName_) {
                    // Loopback marshalled READ (the same agent both encoding and decoding
                    // into itself) is untested; gate it explicitly rather than risk silent
                    // corruption. Direct-only READs are unaffected - the base nixlAgent path
                    // below already handles loopback fine.
                    return NIXL_ERR_NOT_SUPPORTED;
                }
                NIXL_ASSERT(!split.marshalLocal.isEmpty() || !split.directLocal.isEmpty());
                nixlXferReqH *xfer_req = nullptr;
                if (!split.directLocal.isEmpty()) {
                    const auto ret = nixlAgent::createXferReq(operation,
                                                              split.directLocal,
                                                              split.directRemote,
                                                              remote_agent,
                                                              xfer_req,
                                                              extra_params);
                    if (ret != NIXL_SUCCESS) {
                        return ret;
                    }
                }
                if (split.marshalLocal.isEmpty()) {
                    // Direct-only non-direct requests do not use service slots and are not
                    // tracked in outboundXferReqs_/readReceiveReqs_. This is the only
                    // marshalled READ sub-case supported so far when the whole request fits
                    // under direct_desc_threshold.
                    auto wrapper = std::make_unique<nixlServiceXferReqH>();
                    wrapper->marshalOptArgs = non_direct_mode_opt_args;
                    wrapper->op = operation;
                    wrapper->xferReq = xfer_req;
                    req_hndl = wrapper.release();
                    return NIXL_SUCCESS;
                }
                if (operation == NIXL_WRITE) {
                    const nixl_status_t non_direct_ret = create_non_direct_xfer(
                        non_direct_mode_opt_args, split.marshalLocal, split.marshalRemote);
                    if (non_direct_ret != NIXL_SUCCESS) {
                        if (xfer_req != nullptr) {
                            nixlAgent::releaseXferReq(xfer_req);
                        }
                        return non_direct_ret;
                    }
                    req_hndl->xferReq = xfer_req;
                    return NIXL_SUCCESS;
                }

                NIXL_ASSERT(operation == NIXL_READ);
                std::optional<nixlMarshalDeltaReceiverRefArgs> receiver_delta_ref;
                if (const auto *delta_opt_args = getDeltaOptArgs(marshal_opt_args_variant)) {
                    receiver_delta_ref =
                        nixlMarshalDeltaReceiverRefArgs{delta_opt_args->receiverRef,
                                                        delta_opt_args->receiverMemType,
                                                        delta_opt_args->elementSize};
                }
                const nixl_status_t read_ret = create_read_receive_xfer(non_direct_mode_opt_args,
                                                                        split.marshalLocal,
                                                                        split.marshalRemote,
                                                                        xfer_req,
                                                                        receiver_delta_ref);
                if (read_ret != NIXL_SUCCESS) {
                    if (xfer_req != nullptr) {
                        nixlAgent::releaseXferReq(xfer_req);
                    }
                    return read_ret;
                }
                // req_hndl->xferReq is left null: the direct child (xfer_req above) was just
                // handed to create_read_receive_xfer, which attached it to the receive
                // context's own directChild instead (see inboundXferReqH), so
                // progressService() can reach it - req_hndl stays a thin token carrying
                // only readReceiveXferId.
                return NIXL_SUCCESS;
            },
        },
        *marshal_opt_args);
}

nixl_status_t
nixlServiceAgent::makeXferReq(const nixl_xfer_op_t &operation,
                              const nixlDlistH *local_side,
                              const std::vector<int> &local_indices,
                              const nixlDlistH *remote_side,
                              const std::vector<int> &remote_indices,
                              nixlServiceXferReqH *&req_hndl,
                              const nixl_service_opt_args_t *extra_params) {
    const auto marshal_opt_args = getValidMarshalOptArgs(extra_params, data_->mode_);
    if (!marshal_opt_args.has_value()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    return std::visit(overloaded{
                          [&](const nixlMarshalDirectOptArgs &) -> nixl_status_t {
                              auto wrapper = std::make_unique<nixlServiceXferReqH>();
                              wrapper->xferReq = nullptr;
                              wrapper->op = operation;

                              const nixl_status_t ret = nixlAgent::makeXferReq(operation,
                                                                               local_side,
                                                                               local_indices,
                                                                               remote_side,
                                                                               remote_indices,
                                                                               wrapper->xferReq,
                                                                               extra_params);
                              if (ret != NIXL_SUCCESS) {
                                  return ret;
                              }

                              req_hndl = wrapper.release();
                              return ret;
                          },
                          [&](const auto &non_direct_mode_opt_args) -> nixl_status_t {
                              // TODO: allocate a pipeline-tracking nixlServiceXferReqH bound to
                              // the pre-prepared dlist handles.
                              // TODO-Eyal: implement Dlist serialization and validation.
                              // This unconditionally rejects the non-direct path regardless of
                              // `operation`, so both NIXL_WRITE and NIXL_READ are covered.
                              return NIXL_ERR_NOT_SUPPORTED;
                          },
                      },
                      *marshal_opt_args);
}

nixl_status_t
nixlServiceAgent::postXferReq(nixlServiceXferReqH *req_hndl,
                              const nixl_service_opt_args_t *extra_params) {
    return std::visit(
        overloaded{
            [&](const nixlMarshalDirectOptArgs &) -> nixl_status_t {
                return nixlAgent::postXferReq(req_hndl->xferReq, extra_params);
            },
            [&](const auto &non_direct_mode_opt_args) -> nixl_status_t {
                // True for either direction's marshal sub-part; at most one of the two is
                // ever set on a given handle (WRITE never sets readReceiveXferId, READ never
                // sets nonDirectData), so this does not need to branch on req_hndl->op.
                const bool has_marshal_part =
                    req_hndl->nonDirectData.has_value() || req_hndl->readReceiveXferId.has_value();

                // For a READ with a marshal part, the direct child lives on the receive
                // context (inboundXferReqH::directChild) rather than on req_hndl itself;
                // look the context up once, up front, so both the repost check and the
                // direct-child post below can use it. Reject a repost before touching the
                // direct child: its own repost check (inside nixlAgent::postXferReq below)
                // only rejects while its own sub-transfer is still NIXL_IN_PROG, so if it
                // happened to finish first it would otherwise get silently reposted
                // (re-triggering another RDMA read) even though the logical READ as a whole
                // must stay non-repostable for as long as the marshal sub-part is
                // outstanding.
                inboundXferReqH *receive_req = nullptr;
                if (req_hndl->op == NIXL_READ && req_hndl->readReceiveXferId.has_value()) {
                    NIXL_ASSERT(data_->readReceiveReqs_.count(*req_hndl->readReceiveXferId) == 1);
                    receive_req = data_->readReceiveReqs_[*req_hndl->readReceiveXferId].get();
                    if (receive_req->state != nixl_service_xfer_state_t::PRE_START) {
                        return NIXL_ERR_REPOST_ACTIVE;
                    }
                }

                // The direct child to post: the receive context's for a marshal READ (mixed
                // or marshal-only, where it may itself be null), otherwise req_hndl's own
                // (WRITE, or a direct-only READ, which always has one - createXferReq
                // requires at least one of direct/marshal to be non-empty).
                nixlXferReqH *const direct_child =
                    receive_req != nullptr ? receive_req->directChild : req_hndl->xferReq;
                nixl_status_t direct_post_ret = NIXL_SUCCESS;
                if (direct_child != nullptr) {
                    nixl_service_opt_args_t direct_extra_params;
                    const nixl_opt_args_t *direct_extra_params_ptr = nullptr;
                    if (extra_params != nullptr) {
                        direct_extra_params = *extra_params;
                        if (has_marshal_part) {
                            direct_extra_params.notif.reset();
                        }
                        direct_extra_params_ptr = &direct_extra_params;
                    }
                    if (has_marshal_part && req_hndl->op == NIXL_READ) {
                        // The receive context owns the (create- or post-time) notification
                        // and delivers it once both sub-parts finish; the direct child must
                        // never carry one, including a create-time notif baked into the base
                        // handle by createXferReq. Unlike WRITE above, this applies
                        // regardless of extra_params, since for READ (unlike WRITE) the
                        // receive context also captures the create-time notif, so leaving it
                        // on the direct child would fire it twice.
                        direct_extra_params.notif.reset();
                        direct_extra_params.hasNotif = false;
                        direct_extra_params_ptr = &direct_extra_params;
                    }
                    direct_post_ret = nixlAgent::postXferReq(direct_child, direct_extra_params_ptr);
                    if (direct_post_ret < NIXL_SUCCESS) {
                        return direct_post_ret;
                    }
                }

                if (req_hndl->op == NIXL_READ) {
                    if (receive_req == nullptr) {
                        // Direct-only READ: nothing else to post. Return the direct child's
                        // own status rather than hard-coding NIXL_SUCCESS, since it may still
                        // be NIXL_IN_PROG.
                        return direct_post_ret;
                    }
                    NIXL_ASSERT(receive_req->state == nixl_service_xfer_state_t::PRE_START);
                    const auto gen_rreq_ret = data_->genRREQ(*req_hndl);
                    if (gen_rreq_ret != NIXL_SUCCESS) {
                        return gen_rreq_ret;
                    }
                    // Unlike WRITE (PRE_START -> WAIT_CTS -> IN_PROGRESS via CTS), READ has
                    // no CTS: the initiator already advertised its recv slots in RREQ itself.
                    receive_req->state = nixl_service_xfer_state_t::IN_PROGRESS;
                    // Only overwrite the notification captured at create time if extra_params
                    // explicitly supplies one - mirrors the WRITE branch below and the base
                    // nixlAgent::postXferReq convention, so a bare postXferReq(req_hndl)
                    // doesn't silently discard a notification set only at createXferReq()
                    // time.
                    receive_req->notif = resolveReadNotif(receive_req->notif, extra_params);
                    return NIXL_IN_PROG;
                }

                if (!req_hndl->nonDirectData.has_value()) {
                    return NIXL_SUCCESS;
                }
                auto gen_rts_ret = data_->genRTS(*req_hndl);
                if (gen_rts_ret != NIXL_SUCCESS) {
                    return gen_rts_ret;
                }
                req_hndl->nonDirectData->state = nixl_service_xfer_state_t::WAIT_CTS;
                if (extra_params != nullptr && extra_params->notif.has_value()) {
                    // Mixed requests keep the user notification on the marshal path,
                    // which we assume completes last.
                    // TODO: reconsider once notification ordering should cover
                    // both direct and marshal sub-transfers.
                    req_hndl->nonDirectData->notifMsg = *extra_params->notif;
                }
                return NIXL_SUCCESS;
            },
        },
        req_hndl->marshalOptArgs);
}

nixl_status_t
nixlServiceAgent::getXferStatus(nixlServiceXferReqH *req_hndl) {
    if (std::holds_alternative<nixlMarshalDirectOptArgs>(req_hndl->marshalOptArgs)) {
        return nixlAgent::getXferStatus(req_hndl->xferReq);
    }

    data_->progressService();

    if (req_hndl->op == NIXL_READ) {
        if (!req_hndl->readReceiveXferId.has_value()) {
            // Direct-only READ: nothing else to check.
            return nixlAgent::getXferStatus(req_hndl->xferReq);
        }
        NIXL_ASSERT(data_->readReceiveReqs_.count(*req_hndl->readReceiveXferId) == 1);
        auto &receive_req = *data_->readReceiveReqs_[*req_hndl->readReceiveXferId];
        if (receive_req.state == nixl_service_xfer_state_t::FAILED) {
            return receive_req.terminalStatus;
        }
        if (receive_req.state == nixl_service_xfer_state_t::PRE_START) {
            return NIXL_ERR_NOT_POSTED;
        }
        // progressService() above already polled the direct child (if any) and the marshal
        // sub-part, and finalized state via tryCompleteReadReceive once both are done - see
        // progressReadReceive() - so there is nothing left to poll here.
        return (receive_req.state == nixl_service_xfer_state_t::DONE) ? NIXL_SUCCESS : NIXL_IN_PROG;
    }

    if (req_hndl->xferReq != nullptr) {
        const nixl_status_t direct_status = nixlAgent::getXferStatus(req_hndl->xferReq);
        if (direct_status != NIXL_SUCCESS) {
            return direct_status;
        }
    }

    if (!req_hndl->nonDirectData.has_value()) {
        return NIXL_SUCCESS;
    }
    switch (req_hndl->nonDirectData->state) {
    case nixl_service_xfer_state_t::PRE_START:
        return NIXL_ERR_NOT_POSTED;
    case nixl_service_xfer_state_t::WAIT_CTS:
    case nixl_service_xfer_state_t::IN_PROGRESS:
        return NIXL_IN_PROG;
    case nixl_service_xfer_state_t::DONE:
        return NIXL_SUCCESS;
    default:
        NIXL_ASSERT(false);
        return NIXL_ERR_UNKNOWN;
    }
}

nixl_status_t
nixlServiceAgent::releaseXferReq(nixlServiceXferReqH *req_hndl) {
    if (!req_hndl) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return std::visit(
        overloaded{
            [&](const nixlMarshalDirectOptArgs &) -> nixl_status_t {
                const nixl_status_t ret = nixlAgent::releaseXferReq(req_hndl->xferReq);
                delete req_hndl;
                return ret;
            },
            [&](const auto &non_direct_mode_opt_args) -> nixl_status_t {
                NIXL_ASSERT(req_hndl != nullptr);
                nixl_status_t direct_ret = NIXL_SUCCESS;
                if (req_hndl->xferReq != nullptr) {
                    direct_ret = nixlAgent::releaseXferReq(req_hndl->xferReq);
                    req_hndl->xferReq = nullptr;
                }

                if (req_hndl->op == NIXL_READ) {
                    nixl_status_t read_ret = NIXL_SUCCESS;
                    if (req_hndl->readReceiveXferId.has_value()) {
                        const auto xfer_id = *req_hndl->readReceiveXferId;
                        NIXL_ASSERT(data_->readReceiveReqs_.count(xfer_id) == 1);
                        auto &receive_req = *data_->readReceiveReqs_[xfer_id];
                        // For a marshal READ, the direct child (if any - null for
                        // marshal-only) lives here rather than on req_hndl (see
                        // inboundXferReqH::directChild), so it is released here instead of
                        // by the block above. Only clear it once the base release actually
                        // succeeds: releaseXferReq() returns NIXL_ERR_REPOST_ACTIVE without
                        // deleting the handle when the backend fails to cancel an in-flight
                        // transfer, so on that failure directChild is still valid, and
                        // bailing out here - before touching state or slots - leaves the
                        // child, the context, and this token exactly as they were, so the
                        // caller can retry by calling releaseXferReq(req_hndl) again.
                        if (receive_req.directChild != nullptr) {
                            direct_ret = nixlAgent::releaseXferReq(receive_req.directChild);
                            if (direct_ret != NIXL_SUCCESS) {
                                return direct_ret;
                            }
                            receive_req.directChild = nullptr;
                        }
                        switch (receive_req.state) {
                        case nixl_service_xfer_state_t::PRE_START:
                        case nixl_service_xfer_state_t::DONE:
                            // Either the peer never learned about this READ (PRE_START), or
                            // the marshal sub-transfer completed successfully and all of
                            // its slots are idle (see tryCompleteReadReceive) - free and
                            // erase immediately.
                            NIXL_ASSERT(receive_req.directChild == nullptr);
                            data_->freeSlotGroup(receive_req.localSlots);
                            data_->readReceiveReqs_.erase(xfer_id);
                            break;
                        case nixl_service_xfer_state_t::IN_PROGRESS:
                        case nixl_service_xfer_state_t::FAILED:
                            // Mid-transfer release, or a codec-integrity failure on one
                            // slot while a sibling slot's decode may still be legitimately
                            // in flight: either way, slots must not be freed yet. Ask the
                            // peer to quiesce via RABORT and retain the receive context
                            // (and its slots) until handleRAbortAck confirms it has stopped.
                            receive_req.state = nixl_service_xfer_state_t::CANCELLING;
                            read_ret = data_->genRAbort(receive_req.remoteAgent, xfer_id);
                            if (read_ret != NIXL_SUCCESS) {
                                // TODO: handle error.
                                NIXL_ASSERT(false);
                            }
                            break;
                        default:
                            // WAIT_CTS is WRITE-only; CANCELLING means releaseXferReq was
                            // somehow called twice on the same handle.
                            NIXL_ASSERT(false);
                            break;
                        }
                    }
                    delete req_hndl;
                    return (direct_ret != NIXL_SUCCESS) ? direct_ret : read_ret;
                }

                if (!req_hndl->nonDirectData.has_value()) {
                    delete req_hndl;
                    return direct_ret;
                }
                auto &req_data = req_hndl->nonDirectData.value();
                // If the state is done, delete was sent and slots were freed.
                // If the state is pre-start, the receiver doesn't know about the
                // transfer.
                if (req_data.state != nixl_service_xfer_state_t::DONE) {
                    if (req_data.state != nixl_service_xfer_state_t::PRE_START) {
                        // This path is a mid-transfer release.
                        auto ret = data_->genDelete(req_data.remoteAgent, req_data.xferId);
                        if (ret != NIXL_SUCCESS) {
                            // TODO: handle error.
                            NIXL_ASSERT(false);
                            return ret;
                        }
                    }
                    data_->freeSlotGroup(req_data.localSlots);
                }
                data_->outboundXferReqs_.erase(
                    req_data.xferId); // This will delete the req_hndl, since it is
                                      // the raw pointer in the unique_ptr.
                return direct_ret;
            },
        },
        req_hndl->marshalOptArgs);
}

nixl_status_t
nixlServiceAgent::getNotifs(nixl_notifs_t &notifs, const nixl_opt_args_t *extra_params) {
    [[maybe_unused]] auto progress_ret = data_->progressService();
    NIXL_ASSERT(progress_ret == NIXL_SUCCESS || progress_ret == NIXL_IN_PROG);
    auto notifs_ret = nixlAgent::getNotifs(notifs, extra_params);
    return notifs_ret;
}
