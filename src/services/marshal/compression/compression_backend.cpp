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
#include "compression_backend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <optional>
#include <numeric>
#include <nvcomp/ans.h>
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "delta_kernel.cuh"

namespace nixlMarshal {
namespace {

    const std::vector<mem_space_t> kSupportedMemSpaces = {mem_space_t::DEVICE};

    struct marshalOverhead {
        size_t slotOverheadSize;
        size_t workspaceSize;
    };

    constexpr size_t min_payload = 1 << 18; // 256 KB
    // TODO: tune this value based on delta_ans / ans
    constexpr double overhead_multiplier = 2.6; // Base 1.0 + 1.6 Overhead

    constexpr size_t nvcomp_chunk_size = 1 << 18; // 256 KB
    constexpr size_t nvcomp_chunk_default_alignment = 8;
    constexpr size_t max_sub_chunk_count = 8;

    const nvcompBatchedANSCompressOpts_t kANSCompressOpts = {
        nvcomp_rANS,
        NVCOMP_TYPE_FLOAT16,
        max_sub_chunk_count,
        {0}}; // TODO: add options in config, test NVCOMP_TYPE_uint8

    const nvcompBatchedANSDecompressOpts_t kANSDecompressOpts =
        nvcompBatchedANSDecompressDefaultOpts;


    constexpr size_t workspace_size_per_chunk =
        sizeof(void *) * 6; // input ptrs, output ptrs, input sizes, output
                            // sizes, decompress sizes, statuses


    using algo_t = nixl_marshal_compress_algo_t;

    inline void
    throwIfCuda(cudaError_t e, const char *what) {
        if (e != cudaSuccess) {
            throw std::runtime_error(absl::StrCat(what, ": ", cudaGetErrorString(e)));
        }
    }

    inline void
    throwIfNvcomp(nvcompStatus_t s, const char *what) {
        if (s != nvcompSuccess) {
            throw std::runtime_error(absl::StrCat(what, ": nvcomp status=", static_cast<int>(s)));
        }
    }

    // alignment is a power of two
    template<typename T>
    constexpr T
    alignUp(T value, size_t alignment) noexcept {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    template<typename T>
    constexpr bool
    isAlignedTo(T value, size_t alignment) noexcept {
        return (value & (alignment - 1)) == 0;
    }

    inline size_t
    getNvcompChunkStride() {
        size_t max_output_compressed_size;
        throwIfNvcomp(nvcompBatchedANSCompressGetMaxOutputChunkSize(
                          nvcomp_chunk_size, kANSCompressOpts, &max_output_compressed_size),
                      "getNvcompChunkStride: get max output chunk size");
        max_output_compressed_size =
            alignUp(max_output_compressed_size, nvcomp_chunk_default_alignment);
        return std::max(max_output_compressed_size, nvcomp_chunk_size);
    }

    // Extra scratch space for a preprocessing stage, carved out of the per-slot workspace.
    size_t
    algoWorkspaceOverhead(algo_t algo, size_t chunked_payload_size) {
        switch (algo) {
        // ANS_DELTA stages a full-payload copy for the delta kernel
        case algo_t::ANS_DELTA:
            return chunked_payload_size;
        case algo_t::ANS:
            return 0;
        case algo_t::BITCOMP:
            break;
        }
        throw std::runtime_error("CompressionBackend: unsupported compression algo");
    }

    marshalOverhead
    computeMarshalOverhead(size_t chunked_payload_size, algo_t algo) {
        nvcompAlignmentRequirements_t alignment_requirements_compress;
        throwIfNvcomp(nvcompBatchedANSCompressGetRequiredAlignments(
                          kANSCompressOpts, &alignment_requirements_compress),
                      "CompressionBackend: get required alignments");

        nvcompAlignmentRequirements_t alignment_requirements_decompress;
        throwIfNvcomp(nvcompBatchedANSDecompressGetRequiredAlignments(
                          kANSDecompressOpts, &alignment_requirements_decompress),
                      "CompressionBackend: get required alignments");

        size_t nvcomp_num_chunks =
            (chunked_payload_size + nvcomp_chunk_size - 1) / nvcomp_chunk_size;

        size_t temp_compressed_size;
        throwIfNvcomp(nvcompBatchedANSCompressGetTempSizeAsync(nvcomp_num_chunks,
                                                               nvcomp_chunk_size,
                                                               kANSCompressOpts,
                                                               &temp_compressed_size,
                                                               chunked_payload_size),
                      "CompressionBackend: get compressed temp size");

        size_t temp_decompressed_size;
        throwIfNvcomp(nvcompBatchedANSDecompressGetTempSizeAsync(nvcomp_num_chunks,
                                                                 nvcomp_chunk_size,
                                                                 kANSDecompressOpts,
                                                                 &temp_decompressed_size,
                                                                 chunked_payload_size),
                      "CompressionBackend: get uncompressed temp size");


        size_t workspace_size = workspace_size_per_chunk * nvcomp_num_chunks +
            std::max(temp_compressed_size + alignment_requirements_compress.temp,
                     temp_decompressed_size + alignment_requirements_decompress.temp) +
            algoWorkspaceOverhead(algo, chunked_payload_size);
        workspace_size = alignUp(workspace_size, MarshalBackendSizing::slot_stride_alignment);

        size_t slot_overhead_size =
            getNvcompChunkStride() * nvcomp_num_chunks - chunked_payload_size;

        return {slot_overhead_size, workspace_size};
    }

    size_t
    alignedPhysicalSlotStrideSize(size_t chunked_payload_size, algo_t algo) {
        const auto overhead = computeMarshalOverhead(chunked_payload_size, algo);
        const size_t raw_physical_slot_size =
            chunked_payload_size + overhead.slotOverheadSize + overhead.workspaceSize;
        return alignUp(raw_physical_slot_size, MarshalBackendSizing::slot_stride_alignment);
    }

    size_t
    recommendAnsServiceMemSize(size_t chunked_payload_size,
                               uint32_t max_concurrent_transfers,
                               algo_t algo) {
        const size_t actual = std::max(chunked_payload_size, min_payload);
        const size_t slot_stride = alignedPhysicalSlotStrideSize(actual, algo);
        const size_t min_pool_bytes =
            slot_stride * MarshalBackendSizing::slots_per_transfer * max_concurrent_transfers;
        // TODO: I think this is redundant
        const double raw_total = actual * overhead_multiplier *
            MarshalBackendSizing::slots_per_transfer * max_concurrent_transfers;
        size_t total_buffer_size = static_cast<size_t>(std::ceil(raw_total));
        if (total_buffer_size < min_pool_bytes) {
            total_buffer_size = min_pool_bytes;
        }
        const auto remainder = total_buffer_size % slot_stride;
        if (remainder != 0) {
            total_buffer_size += slot_stride - remainder;
        }
        return total_buffer_size;
    }

    class cudaEvent {
        cudaEvent_t event_ = nullptr;

    public:
        explicit cudaEvent(cudaStream_t stream) {
            throwIfCuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
                        "compression: cudaEventCreateWithFlags");
            if (auto err = cudaEventRecord(event_, stream); err != cudaSuccess) {
                cudaEventDestroy(event_);
                event_ = nullptr;
                throwIfCuda(err, "compression: cudaEventRecord failed");
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
            throwIfCuda(err, "compression: cudaEventQuery failed");
            return false;
        }
    };

    class ansWorkspaceLayout {
    public:
        ansWorkspaceLayout(absl::Span<std::byte> workspace,
                           size_t temp_align,
                           size_t nvcomp_num_chunks)
            : nvcompNumChunks_(nvcomp_num_chunks) {
            std::byte *p = workspace.data();

            inputPtrs_ = place<void *>(p);
            outputPtrs_ = place<void *>(p);
            inputSizes_ = place<size_t>(p);
            outputSizes_ = place<size_t>(p);
            decompressSizes_ = place<size_t>(p);
            statuses_ = place<nvcompStatus_t>(p);

            p = reinterpret_cast<std::byte *>(alignUp(reinterpret_cast<uintptr_t>(p), temp_align));
            if (p > workspace.end()) {
                throw std::runtime_error("AnsWorkspaceLayout: workspace is too small");
            }
            tempPtr_ = reinterpret_cast<void *>(p);
            workspaceActualSize_ = static_cast<size_t>(p - workspace.data());
        }

        [[nodiscard]] void **
        getInputPtrsPlace() const noexcept {
            return inputPtrs_;
        }

        [[nodiscard]] size_t *
        getInputSizesPlace() const noexcept {
            return inputSizes_;
        }

        [[nodiscard]] void **
        getOutputPtrsPlace() const noexcept {
            return outputPtrs_;
        }

        [[nodiscard]] size_t *
        getOutputSizesPlace() const noexcept {
            return outputSizes_;
        }

        [[nodiscard]] size_t *
        getDecompressSizesPlace() const noexcept {
            return decompressSizes_;
        }

        [[nodiscard]] nvcompStatus_t *
        getStatusesPlace() const noexcept {
            return statuses_;
        }

        [[nodiscard]] void *
        getTempPtr() const noexcept {
            return tempPtr_;
        }

        [[nodiscard]] size_t
        getWorkspaceActualSize() const noexcept {
            return workspaceActualSize_;
        }

    private:
        template<typename T>
        [[nodiscard]] T *
        place(std::byte *&p) noexcept {
            constexpr std::size_t align = alignof(T);

            auto raw = reinterpret_cast<uintptr_t>(p);
            auto aligned = alignUp(raw, align);

            p = reinterpret_cast<std::byte *>(aligned);

            T *out = reinterpret_cast<T *>(p);

            p += sizeof(T) * nvcompNumChunks_;

            return out;
        }

        size_t nvcompNumChunks_;
        void **inputPtrs_ = nullptr;
        void **outputPtrs_ = nullptr;
        size_t *inputSizes_ = nullptr;
        size_t *outputSizes_ = nullptr;
        size_t *decompressSizes_ = nullptr;
        nvcompStatus_t *statuses_ = nullptr;
        void *tempPtr_ = nullptr;
        size_t workspaceActualSize_ = 0;
    };

    size_t *
    launchAnsCompress(const runtimeBuffer &src,
                      const runtimeBuffer &dst,
                      const runtimeBuffer &workspace,
                      cudaStream_t stream) {

        nvcompAlignmentRequirements_t alignment_requirements;
        throwIfNvcomp(nvcompBatchedANSCompressGetRequiredAlignments(kANSCompressOpts,
                                                                    &alignment_requirements),
                      "launchAnsCompress: get required alignments");
        if (!isAlignedTo(reinterpret_cast<uintptr_t>(src.data), alignment_requirements.input)) {
            throw std::runtime_error(
                "launchAnsCompress: input address is not aligned, the required alignment is " +
                std::to_string(alignment_requirements.input) + " the actual alignment is " +
                std::to_string(reinterpret_cast<uintptr_t>(src.data) &
                               (alignment_requirements.input - 1)));
        }
        if (!isAlignedTo(reinterpret_cast<uintptr_t>(dst.data), alignment_requirements.output)) {
            throw std::runtime_error(
                "launchAnsCompress: output address is not aligned, the required alignment is " +
                std::to_string(alignment_requirements.output) + " the actual alignment is " +
                std::to_string(reinterpret_cast<uintptr_t>(dst.data) &
                               (alignment_requirements.output - 1)));
        }

        // here we populate workspace appropriate pointers
        size_t nvcomp_num_chunks = (src.size + nvcomp_chunk_size - 1) / nvcomp_chunk_size;
        ansWorkspaceLayout workspace_layout(absl::Span<std::byte>(workspace.data, workspace.size),
                                            alignment_requirements.temp,
                                            nvcomp_num_chunks);
        auto d_in_ptrs = workspace_layout.getInputPtrsPlace();
        auto d_in_sizes = workspace_layout.getInputSizesPlace();
        auto d_out_ptrs = workspace_layout.getOutputPtrsPlace();

        std::vector<void *> h_in_ptrs(nvcomp_num_chunks);
        std::vector<void *> h_out_ptrs(nvcomp_num_chunks);
        std::vector<size_t> h_in_sizes(nvcomp_num_chunks, nvcomp_chunk_size);
        size_t last_chunk_size = src.size - nvcomp_chunk_size * (nvcomp_num_chunks - 1);
        h_in_sizes[nvcomp_num_chunks - 1] = last_chunk_size;
        size_t chunk_stride = getNvcompChunkStride();

        // TODO: optimize this, maybe cuda kernel directly to device
        for (size_t i = 0; i < nvcomp_num_chunks; i++) {
            h_in_ptrs[i] = reinterpret_cast<void *>(src.data + i * nvcomp_chunk_size);
            h_out_ptrs[i] = reinterpret_cast<void *>(dst.data + i * chunk_stride);
        }

        throwIfCuda(cudaMemcpyAsync(d_in_ptrs,
                                    h_in_ptrs.data(),
                                    sizeof(void *) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsCompress: in_ptrs copy");
        throwIfCuda(cudaMemcpyAsync(d_out_ptrs,
                                    h_out_ptrs.data(),
                                    sizeof(void *) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsCompress: out_ptrs copy");
        throwIfCuda(cudaMemcpyAsync(d_in_sizes,
                                    h_in_sizes.data(),
                                    sizeof(size_t) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsCompress: in_sizes copy");

        auto d_out_sizes = workspace_layout.getOutputSizesPlace();
        auto temp_data = workspace_layout.getTempPtr();

        throwIfNvcomp(
            nvcompBatchedANSCompressAsync(
                d_in_ptrs,
                d_in_sizes,
                nvcomp_chunk_size,
                nvcomp_num_chunks,
                temp_data,
                workspace.size - workspace_layout.getWorkspaceActualSize(),
                d_out_ptrs,
                d_out_sizes,
                kANSCompressOpts,
                nullptr, // TODO: add to workspace, this is per chunk status array on the device
                stream),
            "launchAnsCompress: nvcompBatchedANSCompressAsync");

        return d_out_sizes;
    }

    size_t *
    launchAnsDecompress(const runtimeBuffer &src,
                        const runtimeBuffer &dst,
                        const runtimeBuffer &workspace,
                        const std::vector<size_t> &segments_sizes,
                        cudaStream_t stream) {

        nvcompAlignmentRequirements_t alignment_requirements;
        throwIfNvcomp(nvcompBatchedANSDecompressGetRequiredAlignments(kANSDecompressOpts,
                                                                      &alignment_requirements),
                      "launchAnsDecompress: get required alignments");

        if (!isAlignedTo(reinterpret_cast<uintptr_t>(src.data), alignment_requirements.input)) {
            throw std::runtime_error(
                "launchAnsDecompress: input address is not aligned, the required alignment is " +
                std::to_string(alignment_requirements.input) + " the actual alignment is " +
                std::to_string(reinterpret_cast<uintptr_t>(src.data) &
                               (alignment_requirements.input - 1)));
        }
        if (!isAlignedTo(reinterpret_cast<uintptr_t>(dst.data), alignment_requirements.output)) {
            throw std::runtime_error(
                "launchAnsDecompress: output address is not aligned, the required alignment is " +
                std::to_string(alignment_requirements.output) + " the actual alignment is " +
                std::to_string(reinterpret_cast<uintptr_t>(dst.data) &
                               (alignment_requirements.output - 1)));
        }
        size_t nvcomp_num_chunks = segments_sizes.size();

        // here we populate workspace appropriate pointers
        ansWorkspaceLayout workspace_layout(absl::Span<std::byte>(workspace.data, workspace.size),
                                            alignment_requirements.temp,
                                            nvcomp_num_chunks);
        auto d_in_ptrs = workspace_layout.getInputPtrsPlace();
        auto d_in_sizes = workspace_layout.getInputSizesPlace();
        auto d_out_ptrs = workspace_layout.getOutputPtrsPlace();
        auto d_decompress_sizes = workspace_layout.getDecompressSizesPlace();
        std::vector<void *> h_in_ptrs(nvcomp_num_chunks);
        std::vector<void *> h_out_ptrs(nvcomp_num_chunks);
        std::vector<size_t> h_decompress_sizes(nvcomp_num_chunks, nvcomp_chunk_size);
        h_decompress_sizes[nvcomp_num_chunks - 1] =
            dst.size - nvcomp_chunk_size * (nvcomp_num_chunks - 1); // last chunk size
        size_t chunk_stride = getNvcompChunkStride();

        for (size_t i = 0; i < nvcomp_num_chunks; i++) {
            h_in_ptrs[i] = reinterpret_cast<void *>(src.data + i * chunk_stride);
            h_out_ptrs[i] = reinterpret_cast<void *>(dst.data + i * nvcomp_chunk_size);
        }

        throwIfCuda(cudaMemcpyAsync(d_in_ptrs,
                                    h_in_ptrs.data(),
                                    sizeof(void *) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsDecompress: in_ptrs copy");
        throwIfCuda(cudaMemcpyAsync(d_out_ptrs,
                                    h_out_ptrs.data(),
                                    sizeof(void *) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsDecompress: out_ptrs copy");
        throwIfCuda(cudaMemcpyAsync(d_in_sizes,
                                    segments_sizes.data(),
                                    sizeof(size_t) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsDecompress: in_sizes copy");
        throwIfCuda(cudaMemcpyAsync(d_decompress_sizes,
                                    h_decompress_sizes.data(),
                                    sizeof(size_t) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsDecompress: decompress_sizes copy");


        auto d_out_sizes = workspace_layout.getOutputSizesPlace();
        auto d_statuses = workspace_layout.getStatusesPlace();
        auto temp_data = workspace_layout.getTempPtr();

        throwIfNvcomp(nvcompBatchedANSDecompressAsync(d_in_ptrs,
                                                      d_in_sizes,
                                                      d_decompress_sizes,
                                                      d_out_sizes,
                                                      nvcomp_num_chunks,
                                                      temp_data,
                                                      workspace.size -
                                                          workspace_layout.getWorkspaceActualSize(),
                                                      d_out_ptrs,
                                                      kANSDecompressOpts,
                                                      d_statuses,
                                                      stream),
                      "launchAnsDecompress: nvcompBatchedANSDecompressAsync");

        return d_out_sizes;
    }

    void
    submitDeltaKernel(const runtimeBuffer &src,
                      const runtimeBuffer &dst,
                      const runtimeBuffer &ref,
                      size_t element_size,
                      cudaStream_t stream) {
        switch (element_size) {
        case 1: {
            throwIfCuda(cudaXorKernel<uint8_t>(dst.data, src.data, ref.data, src.size, stream),
                        "delta submitDeltaKernel: cudaXorKernel failed");
            break;
        }
        case 2: {
            throwIfCuda(cudaXorKernel<uint16_t>(dst.data, src.data, ref.data, src.size, stream),
                        "delta submitDeltaKernel: cudaXorKernel failed");
            break;
        }
        case 4: {
            throwIfCuda(cudaXorKernel<uint32_t>(dst.data, src.data, ref.data, src.size, stream),
                        "delta submitDeltaKernel: cudaXorKernel failed");
            break;
        }
        case 8: {
            throwIfCuda(cudaXorKernel<uint64_t>(dst.data, src.data, ref.data, src.size, stream),
                        "delta submitDeltaKernel: cudaXorKernel failed");
            break;
        }
        default:
            throw std::invalid_argument("delta submitDeltaKernel: unsupported element size");
        }
    }

    void
    validateProcessSlotArgs(const slotBuffers &buffers,
                            const process_slot_input_options_t &opts,
                            algo_t algo) {
        if (buffers.src.size == 0 || buffers.dst.size == 0 ||
            buffers.src.space != mem_space_t::DEVICE || buffers.dst.space != mem_space_t::DEVICE) {
            throw std::runtime_error("validateProcessSlotArgs: invalid arguments");
        }
        auto ws_it = opts.find(option_t::WRITEABLE_WORKSPACE_MEMORY);
        if (ws_it == opts.end()) {
            throw std::runtime_error(
                "validateProcessSlotArgs: writeable workspace memory is required");
        }
        auto workspace_opt =
            std::get_if<WriteableWorkspaceMemory::processSlotInput>(&ws_it->second);
        if (!workspace_opt) {
            throw std::runtime_error(
                "validateProcessSlotArgs: writeable workspace memory is required");
        }
        if (workspace_opt->workspace.space != mem_space_t::DEVICE) {
            throw std::runtime_error(
                "validateProcessSlotArgs: writeable workspace memory is required");
        }


        auto stream_it = opts.find(option_t::USER_CUDA_STREAM);
        if (stream_it == opts.end()) {
            throw std::runtime_error("validateProcessSlotArgs: user cuda stream is required");
        }

        auto user_stream_opt = std::get_if<UserCudaStream::processSlotInput>(&stream_it->second);
        if (!user_stream_opt || user_stream_opt->stream == nullptr) {
            throw std::runtime_error("validateProcessSlotArgs: user cuda stream is required");
        }

        if (algo == algo_t::ANS_DELTA) {
            auto ref_it = opts.find(option_t::READ_ONLY_REFERENCE_STRUCTURED_MEMORY);
            if (ref_it == opts.end()) {
                throw std::runtime_error(
                    "validateProcessSlotArgs: read only reference structured memory is required");
            }
            auto ref_opt =
                std::get_if<ReadOnlyReferenceStructuredMemory::processSlotInput>(&ref_it->second);
            if (!ref_opt || ref_opt->ref.space != mem_space_t::DEVICE) {
                throw std::runtime_error(
                    "validateProcessSlotArgs: read only reference structured memory is required");
            }
        }
    }


} // namespace

class compressionInboundHandle final
    : public asyncHandleImpl<compressionInboundHandle, inboundSlotCompletionData> {
public:
    explicit compressionInboundHandle(std::weak_ptr<backend> backend,
                                      size_t nvcomp_num_chunks,
                                      size_t *device_output_sizes,
                                      cudaStream_t stream)
        : asyncHandleImpl(std::move(backend)),
          doneEvent_(stream),
          nvcompNumChunks_(nvcomp_num_chunks),
          deviceOutputSizes_(device_output_sizes),
          finalSizes_(nvcomp_num_chunks) {}

    std::optional<inboundSlotCompletionData>
    checkForCompletionImpl() {
        if (!doneEvent_.ready()) {
            return std::nullopt;
        }
        throwIfCuda(cudaMemcpy(finalSizes_.data(),
                               deviceOutputSizes_,
                               sizeof(size_t) * nvcompNumChunks_,
                               cudaMemcpyDeviceToHost),
                    "CompressionInboundHandle: finalSizes copy");
        size_t total_size = std::reduce(finalSizes_.begin(), finalSizes_.end(), std::size_t{0});

        return inboundSlotCompletionData{total_size};
    }

private:
    cudaEvent doneEvent_;
    size_t nvcompNumChunks_;
    size_t *deviceOutputSizes_;
    std::vector<size_t> finalSizes_;
};

class compressionOutboundHandle final
    : public asyncHandleImpl<compressionOutboundHandle, outboundSlotCompletionData> {
public:
    explicit compressionOutboundHandle(std::weak_ptr<backend> backend,
                                       size_t nvcomp_num_chunks,
                                       size_t *device_output_sizes,
                                       cudaStream_t stream,
                                       size_t original_payload_size)
        : asyncHandleImpl(std::move(backend)),
          doneEvent_(stream),
          nvcompNumChunks_(nvcomp_num_chunks),
          deviceOutputSizes_(device_output_sizes),
          originalPayloadSize_(original_payload_size),
          finalSizes_(nvcomp_num_chunks) {}

    std::optional<outboundSlotCompletionData>
    checkForCompletionImpl() {
        if (!doneEvent_.ready()) {
            return std::nullopt;
        }
        throwIfCuda(cudaMemcpy(finalSizes_.data(),
                               deviceOutputSizes_,
                               sizeof(size_t) * nvcompNumChunks_,
                               cudaMemcpyDeviceToHost),
                    "CompressionOutboundHandle: finalSizes copy");
        const size_t chunk_stride = getNvcompChunkStride();

        auto segments = std::make_shared<std::vector<ChunkDivision::segment>>(nvcompNumChunks_);
        for (size_t i = 0; i < nvcompNumChunks_; ++i) {
            (*segments)[i] = {i * chunk_stride, finalSizes_[i]};
        }
        outboundSlotCompletionData completion_data;
        completion_data.size = marshal_derived_size;
        completion_data.options.insert(ChunkDivision::processSlotOutput{std::move(segments)});
        completion_data.metadata = marshalMetadata_;
        return completion_data;
    }

private:
    cudaEvent doneEvent_;
    size_t nvcompNumChunks_;
    size_t *deviceOutputSizes_;
    size_t originalPayloadSize_;
    std::vector<size_t> finalSizes_;
    std::string marshalMetadata_ = "";
};

size_t
compressionBackend::recommendServiceMemSize(size_t chunked_payload_size,
                                            uint32_t max_concurrent_transfers,
                                            algo_t algo) {
    // TODO: tune per algo
    return recommendAnsServiceMemSize(chunked_payload_size, max_concurrent_transfers, algo);
}

std::shared_ptr<compressionBackend>
compressionBackend::createBackend(const nixlMarshalCompressConfig &cfg,
                                  size_t chunked_payload_size) {
    return std::make_shared<compressionBackend>(passkey{}, cfg, chunked_payload_size);
}

compressionBackend::compressionBackend(passkey,
                                       const nixlMarshalCompressConfig &cfg,
                                       size_t chunked_payload_size)
    : backend(),
      cfg_(cfg),
      memoryRequirements_() {
    switch (cfg_.algo) {
    case algo_t::ANS_DELTA:
        throw std::runtime_error("CompressionBackend: ans_delta not implemented");
    case algo_t::ANS: {

        auto [slotOverheadSize, workspaceSize] =
            computeMarshalOverhead(chunked_payload_size, cfg_.algo);

        memoryRequirements_.opts[option_t::WRITEABLE_WORKSPACE_MEMORY] =
            WriteableWorkspaceMemory::memoryRequirements{workspaceSize};
        memoryRequirements_.opts[option_t::SLOT_OVERHEAD] =
            SlotOverhead::memoryRequirements{slotOverheadSize};
        break;
    }
    case algo_t::BITCOMP:
        throw std::runtime_error("CompressionBackend: bitcomp not supported");
    default:
        throw std::runtime_error("CompressionBackend: unsupported compression algo");
    }
}

const std::vector<mem_space_t> &
compressionBackend::getSupportedMemSpaces() const {
    return kSupportedMemSpaces;
}

std::unique_ptr<inbound_async_handle_t>
compressionBackend::inboundProcessSlot(const slotBuffers &buffers,
                                       const std::string & /*metadata*/,
                                       const process_slot_input_options_t &opts) {
    validateProcessSlotArgs(buffers, opts, cfg_.algo);
    // TODO: move it to args validation
    const auto chunk_division_it = opts.find(option_t::CHUNK_DIVISION);
    std::vector<size_t> chunk_sizes;
    if (chunk_division_it == opts.end()) {
        chunk_sizes.push_back(buffers.src.size);
    } else {
        const auto *chunk_division_value =
            std::get_if<ChunkDivision::processSlotInput>(&chunk_division_it->second);
        if (!chunk_division_value || !chunk_division_value->segments ||
            chunk_division_value->segments->empty()) {
            throw std::runtime_error(
                "CompressionBackend inboundProcessSlot: invalid chunk division");
        }
        const auto &chunk_segments = *chunk_division_value->segments;
        chunk_sizes.reserve(chunk_segments.size());
        for (const auto &seg : chunk_segments) {
            chunk_sizes.push_back(seg.size);
        }
    }
    // Todo: switch cases based on Metadata


    auto comp_stream =
        std::get<UserCudaStream::processSlotInput>(opts.find(option_t::USER_CUDA_STREAM)->second)
            .stream;

    size_t *device_output_sizes;

    auto workspace = std::get<WriteableWorkspaceMemory::processSlotInput>(
                         opts.find(option_t::WRITEABLE_WORKSPACE_MEMORY)->second)
                         .workspace;

    switch (cfg_.algo) {
    case algo_t::ANS_DELTA: {
        runtimeBuffer delta_staging_buffer(absl::Span<std::byte>(workspace.data, buffers.dst.size),
                                           mem_space_t::DEVICE);
        workspace.data += delta_staging_buffer.size;
        workspace.size -= delta_staging_buffer.size;
        auto ref_mem_opt = std::get<ReadOnlyReferenceStructuredMemory::processSlotInput>(
            opts.find(option_t::READ_ONLY_REFERENCE_STRUCTURED_MEMORY)->second);
        device_output_sizes = launchAnsDecompress(
            buffers.src, delta_staging_buffer, workspace, chunk_sizes, comp_stream);
        submitDeltaKernel(delta_staging_buffer,
                          buffers.dst,
                          ref_mem_opt.ref,
                          ref_mem_opt.elementSize,
                          comp_stream);
        break;
    }
    case algo_t::ANS: {
        device_output_sizes =
            launchAnsDecompress(buffers.src, buffers.dst, workspace, chunk_sizes, comp_stream);
        break;
    }
    case algo_t::BITCOMP:
        throw std::runtime_error("CompressionBackend: bitcomp not supported");
    default:
        throw std::runtime_error("CompressionBackend: unsupported compression algo");
    }
    return std::make_unique<compressionInboundHandle>(
        shared_from_this(), chunk_sizes.size(), device_output_sizes, comp_stream);
}

std::unique_ptr<outbound_async_handle_t>
compressionBackend::outboundProcessSlot(const slotBuffers &buffers,
                                        const process_slot_input_options_t &opts) {
    validateProcessSlotArgs(buffers, opts, cfg_.algo);
    const size_t nvcomp_num_chunks = (buffers.src.size + nvcomp_chunk_size - 1) / nvcomp_chunk_size;
    size_t *device_output_sizes;
    auto comp_stream =
        std::get<UserCudaStream::processSlotInput>(opts.find(option_t::USER_CUDA_STREAM)->second)
            .stream;
    auto workspace = std::get<WriteableWorkspaceMemory::processSlotInput>(
                         opts.find(option_t::WRITEABLE_WORKSPACE_MEMORY)->second)
                         .workspace;
    switch (cfg_.algo) {
    case algo_t::ANS_DELTA: {
        runtimeBuffer delta_staging_buffer(absl::Span<std::byte>(workspace.data, buffers.src.size),
                                           mem_space_t::DEVICE);
        workspace.data += delta_staging_buffer.size;
        workspace.size -= delta_staging_buffer.size;
        auto ref_mem_opt = std::get<ReadOnlyReferenceStructuredMemory::processSlotInput>(
            opts.find(option_t::READ_ONLY_REFERENCE_STRUCTURED_MEMORY)->second);
        submitDeltaKernel(buffers.src,
                          delta_staging_buffer,
                          ref_mem_opt.ref,
                          ref_mem_opt.elementSize,
                          comp_stream);
        device_output_sizes =
            launchAnsCompress(delta_staging_buffer, buffers.dst, workspace, comp_stream);
        break;
    }
    case algo_t::ANS: {
        device_output_sizes = launchAnsCompress(buffers.src, buffers.dst, workspace, comp_stream);
        break;
    }
    case algo_t::BITCOMP:
        throw std::runtime_error("CompressionBackend: bitcomp not supported");
    default:
        throw std::runtime_error("CompressionBackend: unsupported compression algo");
    }
    return std::make_unique<compressionOutboundHandle>(
        shared_from_this(), nvcomp_num_chunks, device_output_sizes, comp_stream, buffers.src.size);
}

memoryRequirements
compressionBackend::getSlotMemoryRequirements() const noexcept {
    return memoryRequirements_;
}
} // namespace nixlMarshal
