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
#include "staging_backend.h"
#include "nixl.h"
#include "nixl_log.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>

namespace nixlMarshal {
namespace {

    const std::vector<mem_space_t> kSupportedMemSpaces = {mem_space_t::DEVICE};

    class cudaEvent {
        cudaEvent_t event_ = nullptr;

    public:
        explicit cudaEvent(cudaStream_t stream) {
            if (auto err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
                err != cudaSuccess) {
                throw std::runtime_error(std::string("staging: cudaEventCreate failed: ") +
                                         cudaGetErrorString(err));
            }
            if (auto err = cudaEventRecord(event_, stream); err != cudaSuccess) {
                cudaEventDestroy(event_);
                event_ = nullptr;
                throw std::runtime_error(std::string("staging: cudaEventRecord failed: ") +
                                         cudaGetErrorString(err));
            }
        }

        cudaEvent(const cudaEvent &) = delete;
        cudaEvent &
        operator=(const cudaEvent &) = delete;
        cudaEvent(cudaEvent &&) = delete;
        cudaEvent &
        operator=(cudaEvent &&) = delete;

        ~cudaEvent() {
            if (event_) {
                cudaEventDestroy(event_);
            }
        }

        bool
        ready() const {
            const auto err = cudaEventQuery(event_);
            if (err == cudaSuccess) {
                return true;
            }
            if (err == cudaErrorNotReady) {
                return false;
            }
            throw std::runtime_error(std::string("staging: cudaEventQuery failed: ") +
                                     cudaGetErrorString(err));
        }
    };

    class stagingInboundHandle final
        : public asyncHandleImpl<stagingInboundHandle, inboundSlotCompletionData> {
        cudaEvent ev_;
        size_t size_;

    public:
        stagingInboundHandle(std::weak_ptr<backend> backend, size_t size, cudaStream_t stream)
            : asyncHandleImpl(std::move(backend)),
              ev_(stream),
              size_(size) {}

        std::optional<inboundSlotCompletionData>
        checkForCompletionImpl() {
            if (!ev_.ready()) {
                return std::nullopt;
            }
            return inboundSlotCompletionData{size_};
        }
    };

    class stagingOutboundHandle final
        : public asyncHandleImpl<stagingOutboundHandle, outboundSlotCompletionData> {
        cudaEvent ev_;
        size_t size_;

    public:
        stagingOutboundHandle(std::weak_ptr<backend> backend, size_t size, cudaStream_t stream)
            : asyncHandleImpl(std::move(backend)),
              ev_(stream),
              size_(size) {}

        std::optional<outboundSlotCompletionData>
        checkForCompletionImpl() {
            if (!ev_.ready()) {
                return std::nullopt;
            }
            // Staging produces no metadata; the wire layer treats empty as "no metadata".
            return outboundSlotCompletionData{size_};
        }
    };

    struct submittedCopy {
        size_t payloadSize;
        cudaStream_t stream;
    };

    submittedCopy
    submitStagingCopy(const slotBuffers &b, const process_slot_input_options_t &opts) {
        if (b.src.space != mem_space_t::DEVICE || b.dst.space != mem_space_t::DEVICE) {
            throw std::runtime_error("staging processSlot: only device memory is supported");
        }

        auto it = opts.find(option_t::USER_CUDA_STREAM);
        if (it == opts.end()) {
            throw std::runtime_error("staging processSlot: VRAM staging requires a CUDA stream");
        }

        auto user_stream_opt = std::get_if<UserCudaStream::processSlotInput>(&it->second);
        if (!user_stream_opt) {
            throw std::runtime_error(
                "staging processSlot: VRAM staging requires a valid CUDA stream option");
        }

        if (b.dst.size < b.src.size) {
            throw std::runtime_error("staging processSlot: destination size smaller than source");
        }

        cudaStream_t stream = user_stream_opt->stream;

        const auto err = cudaMemcpyAsync(reinterpret_cast<void *>(b.dst.data),
                                         reinterpret_cast<const void *>(b.src.data),
                                         b.src.size,
                                         cudaMemcpyDefault,
                                         stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("staging processSlot: cudaMemcpyAsync failed: ") +
                                     cudaGetErrorString(err));
        }

        return {b.src.size, stream};
    }

} // namespace

std::shared_ptr<stagingBackend>
stagingBackend::createBackend(const nixlMarshalStagingConfig &cfg) {
    return std::make_shared<stagingBackend>(passkey{}, cfg);
}

stagingBackend::~stagingBackend() = default;

const std::vector<mem_space_t> &
stagingBackend::getSupportedMemSpaces() const {
    return kSupportedMemSpaces;
}

std::unique_ptr<inbound_async_handle_t>
stagingBackend::inboundProcessSlot(const slotBuffers &buffers,
                                   const std::string & /*metadata*/,
                                   const process_slot_input_options_t &opts) {
    // Staging does not unmarshall, so the inbound metadata is ignored.
    const auto s = submitStagingCopy(buffers, opts);
    return std::make_unique<stagingInboundHandle>(
        std::static_pointer_cast<stagingBackend>(shared_from_this()), s.payloadSize, s.stream);
}

std::unique_ptr<outbound_async_handle_t>
stagingBackend::outboundProcessSlot(const slotBuffers &buffers,
                                    const process_slot_input_options_t &opts) {
    const auto s = submitStagingCopy(buffers, opts);
    return std::make_unique<stagingOutboundHandle>(
        std::static_pointer_cast<stagingBackend>(shared_from_this()), s.payloadSize, s.stream);
}

memoryRequirements
stagingBackend::getSlotMemoryRequirements() const noexcept {
    // The staging backend does not need any per-slot workspace or overhead,
    // so chunkedPayloadSize is intentionally ignored.
    return memoryRequirements{{}};
}

} // namespace nixlMarshal
