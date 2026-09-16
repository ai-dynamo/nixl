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
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <numeric>
#include <nvcomp/ans.h>
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "delta_kernel.cuh"
#include "coalesce_kernels.cuh"

namespace nixlMarshal {

struct ansDataTypeLayout {
    nvcompAlignmentRequirements_t compressionAlignmentRequirements;
    size_t compressionTempSize;
    size_t chunkStride;
};

struct ansLayoutCache {
    std::array<ansDataTypeLayout,
               static_cast<size_t>(nixl_marshal_compress_data_type_t::NUM_COMPRESS_DATA_TYPES)>
        dataTypeLayouts;
    nvcompAlignmentRequirements_t decompressionAlignmentRequirements;
    size_t decompressionTempSize;
    size_t numChunks;
};

namespace {

    const std::vector<mem_space_t> kSupportedMemSpaces = {mem_space_t::DEVICE};

    struct marshalOverhead {
        size_t slotOverheadSize;
        size_t workspaceSize;
    };

    constexpr size_t min_payload = 1 << 18; // 256 KB
    // TODO: tune this value based on delta_ans / ans
    constexpr double overhead_multiplier = 2.7; // heuristic for maximum compressed output and
                                                // workspace size (total physical slot size)

    constexpr size_t min_alignment = 1;
    constexpr size_t nvcomp_chunk_default_alignment = 8;
    constexpr size_t max_sub_chunk_count = 8;
    constexpr std::array<nixl_marshal_compress_data_type_t,
                         static_cast<size_t>(
                             nixl_marshal_compress_data_type_t::NUM_COMPRESS_DATA_TYPES)>
        supported_ans_data_types = {
            nixl_marshal_compress_data_type_t::CHAR,
            nixl_marshal_compress_data_type_t::UCHAR,
            nixl_marshal_compress_data_type_t::FLOAT16,
            nixl_marshal_compress_data_type_t::FLOAT8_E4M3,
    };

    const nvcompBatchedANSDecompressOpts_t kANSDecompressOpts =
        nvcompBatchedANSDecompressDefaultOpts;


    constexpr size_t workspace_size_per_chunk =
        sizeof(void *) * 7; // input ptrs, output ptrs, input sizes, output
                            // sizes, decompress sizes, statuses, offsets


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

    nvcompType_t
    toNvcompDataType(nixl_marshal_compress_data_type_t data_type) {
        switch (data_type) {
        case nixl_marshal_compress_data_type_t::CHAR:
            return NVCOMP_TYPE_CHAR;
        case nixl_marshal_compress_data_type_t::UCHAR:
            return NVCOMP_TYPE_UCHAR;
        case nixl_marshal_compress_data_type_t::FLOAT16:
            return NVCOMP_TYPE_FLOAT16;
        case nixl_marshal_compress_data_type_t::FLOAT8_E4M3:
            return NVCOMP_TYPE_FLOAT8_E4M3;
        case nixl_marshal_compress_data_type_t::NUM_COMPRESS_DATA_TYPES:
            break;
        }
        throw std::invalid_argument("compressionBackend: unsupported ANS data type");
    }

    nvcompBatchedANSCompressOpts_t
    makeANSCompressOpts(nixl_marshal_compress_data_type_t data_type) {
        return {nvcomp_rANS, toNvcompDataType(data_type), max_sub_chunk_count, {0}};
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

    size_t
    getNvcompChunkStride(
        size_t nvcomp_chunk_size,
        size_t max_output_compressed_size,
        const nvcompAlignmentRequirements_t &compression_alignment_requirements,
        const nvcompAlignmentRequirements_t &decompression_alignment_requirements) {
        const auto alignment = std::max({nvcomp_chunk_default_alignment,
                                         compression_alignment_requirements.output,
                                         decompression_alignment_requirements.input});
        return alignUp(std::max(max_output_compressed_size, nvcomp_chunk_size), alignment);
    }

    std::unique_ptr<const ansLayoutCache>
    makeANSLayoutCache(size_t chunked_payload_size, size_t nvcomp_chunk_size) {
        auto cache = std::make_unique<ansLayoutCache>();
        throwIfNvcomp(nvcompBatchedANSDecompressGetRequiredAlignments(
                          kANSDecompressOpts, &cache->decompressionAlignmentRequirements),
                      "compressionBackend: get required alignments");

        cache->numChunks = (chunked_payload_size + nvcomp_chunk_size - 1) / nvcomp_chunk_size;
        for (size_t i = 0; i < supported_ans_data_types.size(); ++i) {
            const auto compress_opts = makeANSCompressOpts(supported_ans_data_types[i]);
            auto &layout = cache->dataTypeLayouts[i];
            throwIfNvcomp(nvcompBatchedANSCompressGetRequiredAlignments(
                              compress_opts, &layout.compressionAlignmentRequirements),
                          "compressionBackend: get required alignments");
            throwIfNvcomp(nvcompBatchedANSCompressGetTempSizeAsync(cache->numChunks,
                                                                   nvcomp_chunk_size,
                                                                   compress_opts,
                                                                   &layout.compressionTempSize,
                                                                   chunked_payload_size),
                          "compressionBackend: get compressed temp size");

            size_t max_output_compressed_size;
            throwIfNvcomp(nvcompBatchedANSCompressGetMaxOutputChunkSize(
                              nvcomp_chunk_size, compress_opts, &max_output_compressed_size),
                          "compressionBackend: get max output chunk size");
            layout.chunkStride = getNvcompChunkStride(nvcomp_chunk_size,
                                                      max_output_compressed_size,
                                                      layout.compressionAlignmentRequirements,
                                                      cache->decompressionAlignmentRequirements);
        }

        throwIfNvcomp(nvcompBatchedANSDecompressGetTempSizeAsync(cache->numChunks,
                                                                 nvcomp_chunk_size,
                                                                 kANSDecompressOpts,
                                                                 &cache->decompressionTempSize,
                                                                 chunked_payload_size),
                      "compressionBackend: get uncompressed temp size");
        return cache;
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
        throw std::runtime_error("compressionBackend: unsupported compression algo");
    }

    marshalOverhead
    computeMarshalOverhead(size_t chunked_payload_size, algo_t algo, const ansLayoutCache &cache) {
        const size_t nvcomp_num_chunks = cache.numChunks;
        size_t max_chunk_stride = 0;
        size_t max_temp_compressed_size = 0;
        size_t max_compress_temp_alignment = min_alignment;
        size_t max_compress_output_alignment = min_alignment;
        for (const auto &layout : cache.dataTypeLayouts) {
            max_chunk_stride = std::max(max_chunk_stride, layout.chunkStride);
            max_temp_compressed_size =
                std::max(max_temp_compressed_size, layout.compressionTempSize);
            max_compress_temp_alignment =
                std::max(max_compress_temp_alignment, layout.compressionAlignmentRequirements.temp);
            max_compress_output_alignment = std::max(
                max_compress_output_alignment, layout.compressionAlignmentRequirements.output);
        }

        // offsets[] and sizes[] are NIXL's packed-format header; nvCOMP's per-chunk maximum
        // accounts only for the compressed chunk data that follows it.
        const size_t wire_data_capacity =
            2 * nvcomp_num_chunks * sizeof(size_t) + max_chunk_stride * nvcomp_num_chunks;
        const size_t slot_overhead_size = wire_data_capacity - chunked_payload_size;

        size_t workspace_size = workspace_size_per_chunk * nvcomp_num_chunks +
            max_compress_output_alignment + wire_data_capacity +
            std::max(max_temp_compressed_size + max_compress_temp_alignment,
                     cache.decompressionTempSize + cache.decompressionAlignmentRequirements.temp) +
            algoWorkspaceOverhead(algo,
                                  chunked_payload_size); // we add slot size to the workspace size
                                                         // because of coalescing
        workspace_size = alignUp(workspace_size, MarshalBackendSizing::slot_stride_alignment);


        return {slot_overhead_size, workspace_size};
    }

    size_t
    alignedPhysicalSlotStrideSize(size_t chunked_payload_size,
                                  algo_t algo,
                                  const ansLayoutCache &cache) {
        const auto overhead = computeMarshalOverhead(chunked_payload_size, algo, cache);
        const size_t raw_physical_slot_size =
            chunked_payload_size + overhead.slotOverheadSize + overhead.workspaceSize;
        return alignUp(raw_physical_slot_size, MarshalBackendSizing::slot_stride_alignment);
    }

    size_t
    recommendAnsServiceMemSize(size_t chunked_payload_size,
                               uint32_t max_concurrent_transfers,
                               algo_t algo,
                               size_t nvcomp_chunk_size) {
        const size_t actual = std::max(chunked_payload_size, min_payload);
        const auto cache = makeANSLayoutCache(actual, nvcomp_chunk_size);
        const size_t slot_stride = alignedPhysicalSlotStrideSize(actual, algo, *cache);
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
                           size_t slot_align,
                           size_t temp_align,
                           size_t slot_size,
                           size_t nvcomp_num_chunks)
            : nvcompNumChunks_(nvcomp_num_chunks) {
            std::byte *p = workspace.data();

            inputPtrs_ = place<void *>(p);
            outputPtrs_ = place<void *>(p);
            inputSizes_ = place<size_t>(p);
            outputSizes_ = place<size_t>(p);
            decompressSizes_ = place<size_t>(p);
            statuses_ = place<nvcompStatus_t>(p);
            offsets_ = place<size_t>(p);
            p = reinterpret_cast<std::byte *>(alignUp(reinterpret_cast<uintptr_t>(p), slot_align));
            slotPtr_ = p;
            p += slot_size;
            p = reinterpret_cast<std::byte *>(alignUp(reinterpret_cast<uintptr_t>(p), temp_align));
            if (p > workspace.end()) {
                throw std::runtime_error("ansWorkspaceLayout: workspace is too small");
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

        [[nodiscard]] size_t *
        getOffsetsPlace() const noexcept {
            return offsets_;
        }

        [[nodiscard]] void *
        getSlotPtr() const noexcept {
            return slotPtr_;
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
        size_t *offsets_ = nullptr;
        void *slotPtr_ = nullptr;
        void *tempPtr_ = nullptr;
        size_t workspaceActualSize_ = 0;
    };

    struct ansCompressResult {
        size_t *sizes;
        size_t *offsets;
    };

    ansCompressResult
    launchAnsCompress(const runtimeBuffer &src,
                      const runtimeBuffer &dst,
                      const runtimeBuffer &workspace,
                      cudaStream_t stream,
                      size_t nvcomp_chunk_size,
                      nixl_marshal_compress_data_type_t data_type,
                      const ansLayoutCache &cache) {

        const auto compress_opts = makeANSCompressOpts(data_type);
        const auto index = static_cast<size_t>(data_type);
        const auto &layout = cache.dataTypeLayouts[index];
        const auto &alignment_requirements = layout.compressionAlignmentRequirements;
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
        const size_t chunk_stride = layout.chunkStride;
        const size_t packed_header_size = 2 * sizeof(size_t) * nvcomp_num_chunks;
        if (packed_header_size > dst.size ||
            chunk_stride > (dst.size - packed_header_size) / nvcomp_num_chunks) {
            throw std::runtime_error("launchAnsCompress: destination buffer is too small");
        }
        ansWorkspaceLayout workspace_layout(absl::Span<std::byte>(workspace.data, workspace.size),
                                            alignment_requirements.output,
                                            alignment_requirements.temp,
                                            dst.size,
                                            nvcomp_num_chunks);
        auto d_in_ptrs = workspace_layout.getInputPtrsPlace();
        auto d_in_sizes = workspace_layout.getInputSizesPlace();
        auto d_out_ptrs = workspace_layout.getOutputPtrsPlace();

        std::vector<size_t> h_in_sizes(nvcomp_num_chunks, nvcomp_chunk_size);
        std::vector<void *> h_ptrs(nvcomp_num_chunks * 2);

        if (src.size % nvcomp_chunk_size != 0) {
            h_in_sizes[nvcomp_num_chunks - 1] = src.size % nvcomp_chunk_size;
        }

        auto *slot_ptr = reinterpret_cast<std::byte *>(workspace_layout.getSlotPtr());
        // we copy both the input and output pointers to one array to save one cudaMemcpyAsync call,
        // as they are already in the same memory space
        for (size_t i = 0; i < nvcomp_num_chunks; i++) {
            h_ptrs[i] = reinterpret_cast<void *>(src.data + i * nvcomp_chunk_size);
            h_ptrs[i + nvcomp_num_chunks] = reinterpret_cast<void *>(slot_ptr + i * chunk_stride);
        }
        throwIfCuda(cudaMemcpyAsync(d_in_ptrs,
                                    h_ptrs.data(),
                                    sizeof(void *) * nvcomp_num_chunks * 2,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsCompress: in_ptrs and out_ptrs copy");

        throwIfCuda(cudaMemcpyAsync(d_in_sizes,
                                    h_in_sizes.data(),
                                    sizeof(size_t) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsCompress: in_sizes copy");

        auto d_out_sizes = workspace_layout.getOutputSizesPlace();
        auto temp_data = workspace_layout.getTempPtr();

        throwIfNvcomp(nvcompBatchedANSCompressAsync(d_in_ptrs,
                                                    d_in_sizes,
                                                    nvcomp_chunk_size,
                                                    nvcomp_num_chunks,
                                                    temp_data,
                                                    workspace.size -
                                                        workspace_layout.getWorkspaceActualSize(),
                                                    d_out_ptrs,
                                                    d_out_sizes,
                                                    compress_opts,
                                                    nullptr,
                                                    stream),
                      "launchAnsCompress: nvcompBatchedANSCompressAsync");

        auto d_offsets = workspace_layout.getOffsetsPlace();

        throwIfCuda(cudaExclusivePrefixSum(d_out_sizes, d_offsets, nvcomp_num_chunks, stream),
                    "launchAnsCompress: exclusive prefix sum of chunk sizes");
        throwIfCuda(cudaMemcpyAsync(dst.data,
                                    d_offsets,
                                    sizeof(size_t) * nvcomp_num_chunks,
                                    cudaMemcpyDeviceToDevice,
                                    stream),
                    "launchAnsCompress: offsets copy");
        std::byte *d_out_ptr = dst.data + sizeof(size_t) * nvcomp_num_chunks;
        throwIfCuda(cudaMemcpyAsync(d_out_ptr,
                                    d_out_sizes,
                                    sizeof(size_t) * nvcomp_num_chunks,
                                    cudaMemcpyDeviceToDevice,
                                    stream),
                    "launchAnsCompress: sizes copy");
        d_out_ptr += sizeof(size_t) * nvcomp_num_chunks;
        throwIfCuda(cudaCoalesceKernel(
                        d_out_ptr, d_out_ptrs, d_out_sizes, d_offsets, nvcomp_num_chunks, stream),
                    "launchAnsCompress: coalesce kernel");

        // now, the dst buffer is offsets | sizes | chunk 0 | chunk 1 | ...

        return {d_out_sizes, d_offsets};
    }

    // nvcomp_num_chunks is derived from the uncompressed size, so it must be supplied by the
    // caller: src here is the packed compressed buffer and its size carries no chunk division.
    size_t *
    launchAnsDecompress(const runtimeBuffer &src,
                        const runtimeBuffer &dst,
                        const runtimeBuffer &workspace,
                        size_t nvcomp_num_chunks,
                        size_t nvcomp_chunk_size,
                        cudaStream_t stream,
                        const ansLayoutCache &cache) {

        const size_t packed_header_size = 2 * sizeof(size_t) * nvcomp_num_chunks;
        if (src.size < packed_header_size) {
            throw std::runtime_error(
                "launchAnsDecompress: compressed buffer is smaller than packed header");
        }

        const auto &alignment_requirements = cache.decompressionAlignmentRequirements;

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

        // here we populate workspace appropriate pointers
        ansWorkspaceLayout workspace_layout(absl::Span<std::byte>(workspace.data, workspace.size),
                                            min_alignment,
                                            alignment_requirements.temp,
                                            dst.size,
                                            nvcomp_num_chunks);
        auto d_in_ptrs = workspace_layout.getInputPtrsPlace();
        auto d_out_ptrs = workspace_layout.getOutputPtrsPlace();
        auto d_decompress_sizes = workspace_layout.getDecompressSizesPlace();
        std::vector<void *> h_out_ptrs(nvcomp_num_chunks);
        std::vector<size_t> h_decompress_sizes(nvcomp_num_chunks, nvcomp_chunk_size);

        if (dst.size % nvcomp_chunk_size != 0) {
            h_decompress_sizes[nvcomp_num_chunks - 1] = dst.size % nvcomp_chunk_size;
        }

        for (size_t i = 0; i < nvcomp_num_chunks; i++) {
            h_out_ptrs[i] = reinterpret_cast<void *>(dst.data + i * nvcomp_chunk_size);
        }


        throwIfCuda(cudaMemcpyAsync(d_out_ptrs,
                                    h_out_ptrs.data(),
                                    sizeof(void *) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsDecompress: out_ptrs copy");
        throwIfCuda(cudaMemcpyAsync(d_decompress_sizes,
                                    h_decompress_sizes.data(),
                                    sizeof(size_t) * nvcomp_num_chunks,
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "launchAnsDecompress: decompress_sizes copy");

        // The packed layout is offsets | sizes | chunk 0 | chunk 1 | ...
        const auto *d_packed_offsets = reinterpret_cast<const size_t *>(src.data);
        const auto *d_packed_sizes =
            reinterpret_cast<const size_t *>(src.data + sizeof(size_t) * nvcomp_num_chunks);
        void *packed_base = src.data + packed_header_size;

        // The packed offsets are relative to packed_base, so we materialize absolute chunk
        // pointers into the workspace: src is read-only and its header must stay intact.
        throwIfCuda(cudaChunkPtrsFromOffsets(
                        d_in_ptrs, d_packed_offsets, packed_base, nvcomp_num_chunks, stream),
                    "launchAnsDecompress: convert offsets to input chunk pointers");


        auto d_out_sizes = workspace_layout.getOutputSizesPlace();
        auto d_statuses = workspace_layout.getStatusesPlace();
        auto temp_data = workspace_layout.getTempPtr();

        throwIfNvcomp(nvcompBatchedANSDecompressAsync(d_in_ptrs,
                                                      d_packed_sizes,
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

    nixl_marshal_compress_data_type_t
    getANSDataType(const process_slot_input_options_t &opts) {
        const auto data_type_it = opts.find(option_t::ANS_DATA_TYPE);
        if (data_type_it == opts.end()) {
            return nixl_marshal_compress_data_type_t::FLOAT16;
        }
        const auto *data_type_opt =
            std::get_if<AnsDataType::processSlotInput>(&data_type_it->second);
        if (data_type_opt == nullptr) {
            throw std::runtime_error("compressionBackend: invalid ANS data type option");
        }
        return data_type_opt->dataType;
    }

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

        slot_completion_result_t<inboundSlotCompletionData>
        checkForCompletionImpl() {
            if (!doneEvent_.ready()) {
                return NIXL_IN_PROG;
            }
            throwIfCuda(cudaMemcpy(finalSizes_.data(),
                                   deviceOutputSizes_,
                                   sizeof(size_t) * nvcompNumChunks_,
                                   cudaMemcpyDeviceToHost),
                        "compressionInboundHandle: finalSizes copy");
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
                                           ansCompressResult compress_result,
                                           cudaStream_t stream)
            : asyncHandleImpl(std::move(backend)),
              doneEvent_(stream),
              nvcompNumChunks_(nvcomp_num_chunks),
              compressResult_(compress_result) {}

        slot_completion_result_t<outboundSlotCompletionData>
        checkForCompletionImpl() {
            if (!doneEvent_.ready()) {
                return NIXL_IN_PROG;
            }
            size_t last_offset, last_size;
            throwIfCuda(cudaMemcpy(&last_offset,
                                   compressResult_.offsets + (nvcompNumChunks_ - 1),
                                   sizeof(size_t),
                                   cudaMemcpyDeviceToHost),
                        "compressionOutboundHandle: lastOffset copy");
            throwIfCuda(cudaMemcpy(&last_size,
                                   compressResult_.sizes + (nvcompNumChunks_ - 1),
                                   sizeof(size_t),
                                   cudaMemcpyDeviceToHost),
                        "compressionOutboundHandle: lastSize copy");
            size_t coalesced_bytes = last_offset + last_size +
                2 * sizeof(size_t) *
                    nvcompNumChunks_; // 2 * sizeof(size_t) * nvcompNumChunks_ for offsets and sizes

            outboundSlotCompletionData completion_data;
            completion_data.size = coalesced_bytes;
            completion_data.metadata = marshalMetadata_;
            return completion_data;
        }

    private:
        cudaEvent doneEvent_;
        size_t nvcompNumChunks_;
        ansCompressResult compressResult_;
        std::string marshalMetadata_ = "";
    };

} // namespace

size_t
compressionBackend::nvcompChunkSizeForPayload(size_t payload) {
    constexpr size_t ki_b = size_t{1} << 10;
    constexpr size_t mi_b = size_t{1} << 20;
    constexpr size_t gi_b = size_t{1} << 30;

    if (payload <= 2 * mi_b) {
        return 16 * ki_b;
    }
    if (payload <= 16 * mi_b) {
        return 32 * ki_b;
    }
    if (payload <= 64 * mi_b) {
        return 64 * ki_b;
    }
    if (payload <= 600 * mi_b) {
        return 128 * ki_b;
    }
    if (payload <= 1 * gi_b) {
        return 256 * ki_b;
    }
    if (payload <= 4 * gi_b) {
        return 512 * ki_b;
    }
    return 512 * ki_b;
}

size_t
compressionBackend::recommendServiceMemSize(size_t chunked_payload_size,
                                            uint32_t num_slot_groups,
                                            algo_t algo) {
    // TODO: tune per algo
    return recommendAnsServiceMemSize(chunked_payload_size,
                                      num_slot_groups,
                                      algo,
                                      nvcompChunkSizeForPayload(chunked_payload_size));
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
      nvcompChunkSize_(nvcompChunkSizeForPayload(chunked_payload_size)),
      ansLayoutCache_(cfg.algo == algo_t::ANS || cfg.algo == algo_t::ANS_DELTA ?
                          makeANSLayoutCache(chunked_payload_size, nvcompChunkSize_) :
                          nullptr),
      memoryRequirements_() {
    switch (cfg_.algo) {
    case algo_t::ANS_DELTA:
    case algo_t::ANS: {

        auto [slotOverheadSize, workspaceSize] =
            computeMarshalOverhead(chunked_payload_size, cfg_.algo, *ansLayoutCache_);

        memoryRequirements_.opts[option_t::WRITEABLE_WORKSPACE_MEMORY] =
            WriteableWorkspaceMemory::memoryRequirements{workspaceSize};
        memoryRequirements_.opts[option_t::SLOT_OVERHEAD] =
            SlotOverhead::memoryRequirements{slotOverheadSize};
        break;
    }
    case algo_t::BITCOMP:
        throw std::runtime_error("compressionBackend: bitcomp not supported");
    default:
        throw std::runtime_error("compressionBackend: unsupported compression algo");
    }
}

compressionBackend::~compressionBackend() = default;

const std::vector<mem_space_t> &
compressionBackend::getSupportedMemSpaces() const {
    return kSupportedMemSpaces;
}

std::unique_ptr<inbound_async_handle_t>
compressionBackend::inboundProcessSlot(const slotBuffers &buffers,
                                       const std::string & /*metadata*/,
                                       const process_slot_input_options_t &opts) {
    validateProcessSlotArgs(buffers, opts, cfg_.algo);

    size_t nvcomp_num_chunks = (buffers.dst.size + nvcompChunkSize_ - 1) / nvcompChunkSize_;
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
        device_output_sizes = launchAnsDecompress(buffers.src,
                                                  delta_staging_buffer,
                                                  workspace,
                                                  nvcomp_num_chunks,
                                                  nvcompChunkSize_,
                                                  comp_stream,
                                                  *ansLayoutCache_);
        submitDeltaKernel(delta_staging_buffer,
                          buffers.dst,
                          ref_mem_opt.ref,
                          ref_mem_opt.elementSize,
                          comp_stream);
        break;
    }
    case algo_t::ANS: {
        device_output_sizes = launchAnsDecompress(buffers.src,
                                                  buffers.dst,
                                                  workspace,
                                                  nvcomp_num_chunks,
                                                  nvcompChunkSize_,
                                                  comp_stream,
                                                  *ansLayoutCache_);
        break;
    }
    default:
        throw std::runtime_error("compressionBackend: unsupported compression algo");
    }
    return std::make_unique<compressionInboundHandle>(
        shared_from_this(), nvcomp_num_chunks, device_output_sizes, comp_stream);
}

std::unique_ptr<outbound_async_handle_t>
compressionBackend::outboundProcessSlot(const slotBuffers &buffers,
                                        const process_slot_input_options_t &opts) {
    validateProcessSlotArgs(buffers, opts, cfg_.algo);
    const auto ans_data_type = getANSDataType(opts);
    size_t nvcomp_num_chunks = (buffers.src.size + nvcompChunkSize_ - 1) / nvcompChunkSize_;
    ansCompressResult compress_result;
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
        compress_result = launchAnsCompress(delta_staging_buffer,
                                            buffers.dst,
                                            workspace,
                                            comp_stream,
                                            nvcompChunkSize_,
                                            ans_data_type,
                                            *ansLayoutCache_);
        break;
    }
    case algo_t::ANS: {
        compress_result = launchAnsCompress(buffers.src,
                                            buffers.dst,
                                            workspace,
                                            comp_stream,
                                            nvcompChunkSize_,
                                            ans_data_type,
                                            *ansLayoutCache_);
        break;
    }
    case algo_t::BITCOMP:
        throw std::runtime_error("compressionBackend: bitcomp not supported");
    default:
        throw std::runtime_error("compressionBackend: unsupported compression algo");
    }
    return std::make_unique<compressionOutboundHandle>(
        shared_from_this(), nvcomp_num_chunks, compress_result, comp_stream);
}

memoryRequirements
compressionBackend::getSlotMemoryRequirements() const noexcept {
    return memoryRequirements_;
}
} // namespace nixlMarshal
