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
#ifndef STAGING_BACKEND_H
#define STAGING_BACKEND_H

#include "marshal_backend.h"
#include "nixl_service_types.h"

namespace nixlMarshal {

class stagingBackend final : public backend {
private:
    const nixlMarshalStagingConfig cfg_;

    struct passkey {
        explicit passkey() = default;
    };

public:
    static std::shared_ptr<stagingBackend>
    createBackend(const nixlMarshalStagingConfig &cfg);

    static constexpr size_t
    recommendServiceMemSize(size_t chunked_payload_size, uint32_t max_concurrent_transfers) {
        return chunked_payload_size * MarshalBackendSizing::slots_per_transfer *
            max_concurrent_transfers;
    }

    explicit stagingBackend(passkey, const nixlMarshalStagingConfig &cfg) : backend(), cfg_(cfg) {}

    ~stagingBackend() override;

    const std::vector<mem_space_t> &
    getSupportedMemSpaces() const override;

    std::unique_ptr<inbound_async_handle_t>
    inboundProcessSlot(const slotBuffers &buffers,
                       const std::string &metadata,
                       const process_slot_input_options_t &opts = {}) override;

    std::unique_ptr<outbound_async_handle_t>
    outboundProcessSlot(const slotBuffers &buffers,
                        const process_slot_input_options_t &opts = {}) override;

    memoryRequirements
    getSlotMemoryRequirements() const noexcept override;
};

} // namespace nixlMarshal

#endif // STAGING_BACKEND_H
