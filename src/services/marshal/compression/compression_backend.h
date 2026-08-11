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
#ifndef COMPRESSION_BACKEND_H
#define COMPRESSION_BACKEND_H

#include "marshal/marshal_backend.h"
#include "nixl_service_types.h"

#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace nixlMarshal {

class compressionBackend final : public backend {
private:
    struct passkey {
        explicit passkey() = default;
    };

    nixlMarshalCompressConfig cfg_;
    memoryRequirements memoryRequirements_;


public:
    static size_t
    recommendServiceMemSize(size_t chunked_payload_size,
                            uint32_t max_concurrent_transfers,
                            nixl_marshal_compress_algo_t algo);

    static std::shared_ptr<compressionBackend>
    createBackend(const nixlMarshalCompressConfig &cfg, size_t chunked_payload_size);

    explicit compressionBackend(passkey,
                                const nixlMarshalCompressConfig &cfg,
                                size_t chunked_payload_size);
    ~compressionBackend() override = default;


    compressionBackend(const compressionBackend &) = delete;
    compressionBackend &
    operator=(const compressionBackend &) = delete;
    compressionBackend(compressionBackend &&) = delete;
    compressionBackend &
    operator=(compressionBackend &&) = delete;

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
#endif // COMPRESSION_BACKEND_H
