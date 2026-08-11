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
#ifndef DELTA_BACKEND_H
#define DELTA_BACKEND_H

#include "marshal_backend.h"
#include "nixl_service_types.h"

#include <limits>
#include <stdexcept>

namespace nixlMarshal {

/**
 * @class deltaBackend
 * @brief Marshal backend that produces the differences between src and ref buffers
 */
class deltaBackend final : public backend {
private:
    const nixlMarshalDeltaConfig cfg_;

    struct passkey {
        explicit passkey() = default;
    };

public:
    static std::shared_ptr<deltaBackend>
    createBackend(const nixlMarshalDeltaConfig &cfg);

    explicit deltaBackend(passkey, const nixlMarshalDeltaConfig &cfg) : backend(), cfg_(cfg) {
        throw std::runtime_error("DeltaBackend: not implemented");
    }

    const std::vector<mem_space_t> &
    getSupportedMemSpaces() const override;

    std::unique_ptr<outbound_async_handle_t>
    outboundProcessSlot(const slotBuffers &buffers,
                        const process_slot_input_options_t &opts = {}) override;

    std::unique_ptr<inbound_async_handle_t>
    inboundProcessSlot(const slotBuffers &buffers,
                       const std::string &metadata,
                       const process_slot_input_options_t &opts = {}) override;

    static constexpr size_t
    recommendServiceMemSize(size_t chunked_payload_size, uint32_t max_concurrent_transfers) {
        constexpr size_t slots = MarshalBackendSizing::slots_per_transfer;
        if (chunked_payload_size > std::numeric_limits<size_t>::max() / slots) {
            throw std::invalid_argument("chunkedPayloadSize too large");
        }
        const size_t per_transfer_size = chunked_payload_size * slots;
        if (max_concurrent_transfers != 0 &&
            per_transfer_size > std::numeric_limits<size_t>::max() / max_concurrent_transfers) {
            throw std::invalid_argument("maxConcurrentTransfers too large");
        }
        return per_transfer_size * max_concurrent_transfers;
    }

    memoryRequirements
    getSlotMemoryRequirements() const noexcept override;
};

} // namespace nixlMarshal

#endif // DELTA_BACKEND_H
