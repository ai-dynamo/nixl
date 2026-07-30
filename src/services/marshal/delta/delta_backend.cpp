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
#include "delta_backend.h"
#include "delta_kernel.cuh"
#include "nixl.h"

#include <algorithm>

namespace nixlMarshal {
namespace {

    const std::vector<mem_space_t> kSupportedMemSpaces = {mem_space_t::DEVICE};

    template<typename CompletionT>
    class deltaEventHandle : public asyncHandleImpl<deltaEventHandle<CompletionT>, CompletionT> {
    private:
        cudaEvent_t event_;
        size_t size_;

    public:
        deltaEventHandle(std::weak_ptr<backend> backend, cudaStream_t stream, size_t size)
            : asyncHandleImpl<deltaEventHandle, CompletionT>(std::move(backend)),
              size_(size) {
            auto err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("DeltaEventHandle: cudaEventCreateWithFlags failed: ") +
                    cudaGetErrorString(err));
            }
            err = cudaEventRecord(event_, stream);
            if (err != cudaSuccess) {
                cudaEventDestroy(event_);
                event_ = nullptr;
                throw std::runtime_error(std::string("DeltaEventHandle: cudaEventRecord failed: ") +
                                         cudaGetErrorString(err));
            }
        }

        ~deltaEventHandle() override {
            cudaEventDestroy(event_);
        }

        std::optional<CompletionT>
        checkForCompletionImpl() {
            const auto err = cudaEventQuery(event_);
            if (err == cudaSuccess) {
                if constexpr (std::is_same_v<CompletionT, outboundSlotCompletionData>) {
                    return CompletionT{size_};
                }
                if constexpr (std::is_same_v<CompletionT, inboundSlotCompletionData>) {
                    return CompletionT{size_};
                }
                throw std::runtime_error("DeltaEventHandle: unsupported completion type");
            }
            if (err == cudaErrorNotReady) {
                return std::nullopt;
            }
            throw std::runtime_error(std::string("DeltaEventHandle: cudaEventQuery failed: ") +
                                     cudaGetErrorString(err));
        }
    };

    struct assertedOpsVals {
        ReadOnlyReferenceStructuredMemory::processSlotInput refMemOpts;
        UserCudaStream::processSlotInput streamOpts;
    };

    assertedOpsVals
    assertInputValidation(const slotBuffers &b, const process_slot_input_options_t &opts) {
        const auto supported_end = std::end(kSupportedMemSpaces);
        if (std::find(std::begin(kSupportedMemSpaces), supported_end, b.src.space) ==
            supported_end) {
            throw std::runtime_error("delta processSlot: unsupported source memory space");
        }
        if (std::find(std::begin(kSupportedMemSpaces), supported_end, b.dst.space) ==
            supported_end) {
            throw std::runtime_error("delta processSlot: unsupported destination memory space");
        }

        auto it = opts.find(option_t::READ_ONLY_REFERENCE_STRUCTURED_MEMORY);
        if (it == opts.end()) {
            throw std::runtime_error("delta processSlot: delta backend requires a read only "
                                     "reference structured memory option to be provided");
        }

        auto ref_mem_opts =
            std::get_if<ReadOnlyReferenceStructuredMemory::processSlotInput>(&it->second);
        if (!ref_mem_opts) {
            throw std::runtime_error("delta processSlot: delta backend requires a ProcessSlotInput "
                                     "for the reference buffer");
        }
        if (std::find(std::begin(kSupportedMemSpaces), supported_end, ref_mem_opts->ref.space) ==
            supported_end) {
            throw std::runtime_error("delta processSlot: unsupported reference memory space");
        }

        if (b.src.size != b.dst.size || b.src.size != ref_mem_opts->ref.size) {
            throw std::runtime_error("delta processSlot: buffers' sizes don't match");
        }
        if (b.src.size == 0) {
            throw std::runtime_error("delta processSlot: buffers' sizes must be greater than 0");
        }
        if ((b.src.size % ref_mem_opts->elementSize) != 0) {
            throw std::runtime_error(
                "delta processSlot: buffer size must be a multiple of element size");
        }

        it = opts.find(option_t::USER_CUDA_STREAM);
        if (it == opts.end()) {
            throw std::runtime_error(
                "delta processSlot: delta kernel requires a CUDA stream option");
        }

        auto stream_opt = std::get_if<UserCudaStream::processSlotInput>(&it->second);
        if (!stream_opt) {
            throw std::runtime_error("delta processSlot: delta kernel requires a ProcessSlotInput "
                                     "for the CUDA stream option");
        }

        return {*ref_mem_opts, *stream_opt};
    }

    void
    submitDeltaKernel(const slotBuffers &b, const assertedOpsVals &opt_vals) {
        const auto stream = opt_vals.streamOpts.stream;
        const auto ref = opt_vals.refMemOpts.ref;

        const auto err = [&] {
            switch (opt_vals.refMemOpts.elementSize) {
            case 1:
                return cudaXorKernel<uint8_t>(b.dst.data, b.src.data, ref.data, b.src.size, stream);
            case 2:
                return cudaXorKernel<uint16_t>(
                    b.dst.data, b.src.data, ref.data, b.src.size, stream);
            case 4:
                return cudaXorKernel<uint32_t>(
                    b.dst.data, b.src.data, ref.data, b.src.size, stream);
            case 8:
                return cudaXorKernel<uint64_t>(
                    b.dst.data, b.src.data, ref.data, b.src.size, stream);
            default:
                throw std::runtime_error("delta submitDeltaKernel: unsupported element size");
            }
        }();

        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("delta submitDeltaKernel: cudaXorKernel failed: ") +
                cudaGetErrorString(err));
        }
    }

} // namespace

std::shared_ptr<deltaBackend>
deltaBackend::createBackend(const nixlMarshalDeltaConfig &cfg) {
    return std::make_shared<deltaBackend>(passkey{}, cfg);
}

const std::vector<mem_space_t> &
deltaBackend::getSupportedMemSpaces() const {
    return kSupportedMemSpaces;
}

std::unique_ptr<outbound_async_handle_t>
deltaBackend::outboundProcessSlot(const slotBuffers &buffers,
                                  const process_slot_input_options_t &opts) {
    const auto opt_vals = assertInputValidation(buffers, opts);
    submitDeltaKernel(buffers, opt_vals);
    return std::make_unique<deltaEventHandle<outboundSlotCompletionData>>(
        shared_from_this(), opt_vals.streamOpts.stream, buffers.src.size);
}

std::unique_ptr<inbound_async_handle_t>
deltaBackend::inboundProcessSlot(const slotBuffers &buffers,
                                 const std::string & /*metadata*/,
                                 const process_slot_input_options_t &opts) {
    const auto opt_vals = assertInputValidation(buffers, opts);
    submitDeltaKernel(buffers, opt_vals);
    return std::make_unique<deltaEventHandle<inboundSlotCompletionData>>(
        shared_from_this(), opt_vals.streamOpts.stream, buffers.src.size);
}

memoryRequirements
deltaBackend::getSlotMemoryRequirements() const noexcept {
    return memoryRequirements{{}};
}

} // namespace nixlMarshal
