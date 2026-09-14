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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <cuda_runtime_api.h>

#include "compression_backend.h"

namespace gtest {
namespace services {
    namespace marshals {

        namespace {

            using nixlMarshal::compressionBackend;
            using nixlMarshal::mem_space_t;
            using nixlMarshal::option_t;
            using nixlMarshal::process_slot_input_options_t;
            using nixlMarshal::runtimeBuffer;
            using nixlMarshal::slotBuffers;
            namespace AnsDataType = nixlMarshal::AnsDataType;
            namespace ReadOnlyReferenceStructuredMemory =
                nixlMarshal::ReadOnlyReferenceStructuredMemory;
            namespace SlotOverhead = nixlMarshal::SlotOverhead;
            namespace UserCudaStream = nixlMarshal::UserCudaStream;
            namespace WriteableWorkspaceMemory = nixlMarshal::WriteableWorkspaceMemory;

            template<typename CompletionT>
            [[nodiscard]] const CompletionT *
            tryGetCompletedSlotData(
                const nixlMarshal::slot_completion_result_t<CompletionT> &result) noexcept {
                return std::get_if<CompletionT>(&result);
            }

            constexpr size_t payload_bytes = 45 * 1024 * 1024 + 12340;
            constexpr size_t ref_element_size = sizeof(uint8_t);
            constexpr size_t destination_canary_bytes = 64;
            constexpr uint8_t destination_canary_value = 0xA5;
            constexpr std::array<nixl_marshal_compress_data_type_t,
                                 static_cast<size_t>(
                                     nixl_marshal_compress_data_type_t::NUM_COMPRESS_DATA_TYPES)>
                ans_data_types = {
                    nixl_marshal_compress_data_type_t::CHAR,
                    nixl_marshal_compress_data_type_t::UCHAR,
                    nixl_marshal_compress_data_type_t::FLOAT16,
                    nixl_marshal_compress_data_type_t::FLOAT8_E4M3,
            };


            constexpr size_t ref_perturb_stride = 128;
            constexpr uint8_t ref_perturb_mask = 0xFF;
            constexpr auto completion_timeout = std::chrono::seconds(5);

            absl::Span<std::byte>
            asByteSpan(void *p, size_t n) {
                return absl::Span<std::byte>(reinterpret_cast<std::byte *>(p), n);
            }

            constexpr size_t nvcomp_chunk_size_bytes = 1U << 16; // 64 KB, mirrors backend constant

            size_t
            numChunksFor(size_t payload_bytes) {
                return std::max<size_t>(
                    1, (payload_bytes + nvcomp_chunk_size_bytes - 1) / nvcomp_chunk_size_bytes);
            }

            // Reconstructs the total payload size the outbound handle is expected to report by
            // reading the packed "[offsets[N]][sizes[N]]" header that launchAnsCompress writes to
            // the front of the compressed buffer. The coalesced payload ends at the last chunk's
            // offset plus its (unpadded) size, and the header occupies 2 * N * sizeof(size_t).
            size_t
            expectedPackedPayloadSize(const void *packed_dst, size_t num_chunks) {
                std::vector<size_t> offsets(num_chunks);
                std::vector<size_t> sizes(num_chunks);
                const auto *base = reinterpret_cast<const std::byte *>(packed_dst);
                EXPECT_EQ(
                    cudaMemcpy(
                        offsets.data(), base, sizeof(size_t) * num_chunks, cudaMemcpyDeviceToHost),
                    cudaSuccess);
                EXPECT_EQ(cudaMemcpy(sizes.data(),
                                     base + sizeof(size_t) * num_chunks,
                                     sizeof(size_t) * num_chunks,
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
                const size_t header_bytes = 2 * sizeof(size_t) * num_chunks;
                return header_bytes + offsets.back() + sizes.back();
            }

            const char *
            algoName(nixl_marshal_compress_algo_t algo) {
                switch (algo) {
                case nixl_marshal_compress_algo_t::ANS:
                    return "ANS";
                case nixl_marshal_compress_algo_t::ANS_DELTA:
                    return "ANS_DELTA";
                case nixl_marshal_compress_algo_t::BITCOMP:
                    return "BITCOMP";
                }
                return "UNKNOWN";
            }

        } // namespace

        class compressionBackendTest
            : public ::testing::TestWithParam<nixl_marshal_compress_algo_t> {
        protected:
            std::shared_ptr<compressionBackend> backend_;
            size_t slotBytes_ = 0;
            size_t workspaceBytes_ = 0;

            void *gpuSrc_ = nullptr;
            void *gpuDst_ = nullptr;
            void *gpuRoundtrip_ = nullptr;
            void *gpuWorkspace_ = nullptr;
            void *gpuWorkspaceInbound_ = nullptr;
            void *gpuRef_ = nullptr;
            cudaStream_t stream_ = nullptr;

            std::vector<uint8_t> hostSrc_;
            std::vector<uint8_t> hostRef_;

            [[nodiscard]] bool
            isDelta() const {
                return GetParam() == nixl_marshal_compress_algo_t::ANS_DELTA;
            }

            void
            SetUp() override {
                if (isDelta()) {
                    GTEST_SKIP() << "CompressionBackend: ans_delta not implemented (delta mode "
                                    "disabled)";
                }
                int device_count = 0;
                ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
                if (device_count <= 0) {
                    GTEST_SKIP() << "No CUDA device available";
                }
                ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

                nixlMarshalCompressConfig cfg{};
                cfg.algo = GetParam();
                backend_ = compressionBackend::createBackend(cfg, payload_bytes);
                ASSERT_NE(backend_, nullptr);

                const auto mem_reqs = backend_->getSlotMemoryRequirements();
                const auto overhead_it = mem_reqs.opts.find(option_t::SLOT_OVERHEAD);
                ASSERT_NE(overhead_it, mem_reqs.opts.end());
                const auto *slot_overhead =
                    std::get_if<SlotOverhead::memoryRequirements>(&overhead_it->second);
                ASSERT_NE(slot_overhead, nullptr);

                const auto ws_req_it = mem_reqs.opts.find(option_t::WRITEABLE_WORKSPACE_MEMORY);
                ASSERT_NE(ws_req_it, mem_reqs.opts.end());
                const auto *ws_req =
                    std::get_if<WriteableWorkspaceMemory::memoryRequirements>(&ws_req_it->second);
                ASSERT_NE(ws_req, nullptr);

                slotBytes_ = payload_bytes + slot_overhead->slotOverheadSize;
                workspaceBytes_ = ws_req->slotWorkspaceSize;

                ASSERT_EQ(cudaMalloc(&gpuSrc_, payload_bytes), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&gpuDst_, slotBytes_ + destination_canary_bytes), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&gpuRoundtrip_, payload_bytes), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&gpuWorkspace_, workspaceBytes_), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&gpuWorkspaceInbound_, workspaceBytes_), cudaSuccess);
                ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);

                hostSrc_.assign(payload_bytes, 0);
                for (size_t i = 0; i < payload_bytes; ++i) {
                    hostSrc_[i] = (i * 4) % 256;
                }
                ASSERT_EQ(
                    cudaMemcpy(gpuSrc_, hostSrc_.data(), hostSrc_.size(), cudaMemcpyHostToDevice),
                    cudaSuccess);
                ASSERT_EQ(cudaMemset(gpuDst_, 0, slotBytes_), cudaSuccess);
                ASSERT_EQ(cudaMemset(gpuRoundtrip_, 0, payload_bytes), cudaSuccess);
                ASSERT_EQ(cudaMemset(gpuWorkspaceInbound_, 0xCD, workspaceBytes_), cudaSuccess);

                if (isDelta()) {
                    ASSERT_EQ(cudaMalloc(&gpuRef_, payload_bytes), cudaSuccess);
                    hostRef_ = hostSrc_;
                    for (size_t i = 0; i < hostRef_.size(); i += ref_perturb_stride) {
                        hostRef_[i] ^= ref_perturb_mask;
                    }
                    ASSERT_EQ(
                        cudaMemcpy(
                            gpuRef_, hostRef_.data(), hostRef_.size(), cudaMemcpyHostToDevice),
                        cudaSuccess);
                }
            }

            void
            TearDown() override {
                if (stream_) {
                    cudaStreamDestroy(stream_);
                }
                if (gpuWorkspaceInbound_) {
                    cudaFree(gpuWorkspaceInbound_);
                }
                if (gpuWorkspace_) {
                    cudaFree(gpuWorkspace_);
                }
                if (gpuRoundtrip_) {
                    cudaFree(gpuRoundtrip_);
                }
                if (gpuDst_) {
                    cudaFree(gpuDst_);
                }
                if (gpuSrc_) {
                    cudaFree(gpuSrc_);
                }
                if (gpuRef_) {
                    cudaFree(gpuRef_);
                }
            }

            process_slot_input_options_t
            makeOptions(
                const runtimeBuffer &workspace,
                std::optional<nixl_marshal_compress_data_type_t> data_type = std::nullopt) const {
                process_slot_input_options_t opts{
                    {option_t::WRITEABLE_WORKSPACE_MEMORY,
                     WriteableWorkspaceMemory::processSlotInput{workspace}},
                    {option_t::USER_CUDA_STREAM, UserCudaStream::processSlotInput{stream_}},
                };
                if (data_type.has_value()) {
                    opts.emplace(option_t::ANS_DATA_TYPE,
                                 AnsDataType::processSlotInput{*data_type});
                }
                if (isDelta()) {
                    runtimeBuffer ref(asByteSpan(gpuRef_, payload_bytes), mem_space_t::DEVICE);
                    opts.emplace(
                        option_t::READ_ONLY_REFERENCE_STRUCTURED_MEMORY,
                        ReadOnlyReferenceStructuredMemory::processSlotInput{ref, ref_element_size});
                }
                return opts;
            }

            template<typename Handle>
            static auto
            waitForCompletion(Handle &handle) {
                auto future = std::async(std::launch::async, [&] {
                    auto result = handle->checkForCompletion();
                    while (std::holds_alternative<nixl_status_t>(result) &&
                           std::get<nixl_status_t>(result) == NIXL_IN_PROG) {
                        result = handle->checkForCompletion();
                    }
                    return result;
                });
                EXPECT_EQ(future.wait_for(completion_timeout), std::future_status::ready)
                    << "operation did not complete within timeout";
                return future.get();
            }
        };

        TEST_P(compressionBackendTest, CreateBackendValid) {
            EXPECT_NE(backend_, nullptr);
        }

        TEST_P(compressionBackendTest, GetSlotMemoryRequirementsHasOverheadAndWorkspace) {
            EXPECT_GT(slotBytes_, payload_bytes);
            EXPECT_GT(workspaceBytes_, 0U);
        }

        TEST_P(compressionBackendTest, InboundRejectsBufferSmallerThanPackedHeader) {
            const size_t packed_header_size = 2 * sizeof(size_t) * numChunksFor(payload_bytes);
            ASSERT_GT(packed_header_size, 1U);

            runtimeBuffer src(asByteSpan(gpuDst_, packed_header_size - 1), mem_space_t::DEVICE);
            runtimeBuffer dst(asByteSpan(gpuRoundtrip_, payload_bytes), mem_space_t::DEVICE);
            runtimeBuffer workspace(asByteSpan(gpuWorkspaceInbound_, workspaceBytes_),
                                    mem_space_t::DEVICE);

            EXPECT_THROW(
                backend_->inboundProcessSlot(slotBuffers{src, dst}, "", makeOptions(workspace)),
                std::runtime_error);
        }

        TEST_P(compressionBackendTest, OutboundCompressionProducesNonEmptyOutput) {
            runtimeBuffer src(asByteSpan(gpuSrc_, payload_bytes), mem_space_t::DEVICE);
            runtimeBuffer dst(asByteSpan(gpuDst_, slotBytes_), mem_space_t::DEVICE);
            runtimeBuffer workspace(asByteSpan(gpuWorkspace_, workspaceBytes_),
                                    mem_space_t::DEVICE);

            auto handle =
                backend_->outboundProcessSlot(slotBuffers{src, dst}, makeOptions(workspace));
            ASSERT_NE(handle, nullptr);

            auto completion_result = waitForCompletion(handle);
            const auto *completion = tryGetCompletedSlotData(completion_result);
            ASSERT_NE(completion, nullptr);
            const size_t num_chunks = numChunksFor(payload_bytes);
            const size_t header_bytes = 2 * sizeof(size_t) * num_chunks;
            EXPECT_GT(completion->size, header_bytes)
                << "payload must include more than just the offsets/sizes header";
            EXPECT_LE(completion->size, slotBytes_);
            EXPECT_EQ(completion->size, expectedPackedPayloadSize(gpuDst_, num_chunks))
                << "reported size must match header + last chunk offset + last chunk size";
            EXPECT_LT(completion->size, payload_bytes)
                << "compressible input should shrink below the original payload";

            std::array<uint8_t, 64> dst_prefix{};
            ASSERT_EQ(
                cudaMemcpy(dst_prefix.data(), gpuDst_, dst_prefix.size(), cudaMemcpyDeviceToHost),
                cudaSuccess);
            const bool any_non_zero =
                std::any_of(dst_prefix.begin(), dst_prefix.end(), [](uint8_t v) { return v != 0; });
            EXPECT_TRUE(any_non_zero) << "compressed output prefix was all-zero";
        }

        TEST_P(compressionBackendTest, OutboundInboundRoundTripPreservesData) {
            runtimeBuffer src(asByteSpan(gpuSrc_, payload_bytes), mem_space_t::DEVICE);
            runtimeBuffer dst(asByteSpan(gpuDst_, slotBytes_), mem_space_t::DEVICE);
            runtimeBuffer outbound_workspace(asByteSpan(gpuWorkspace_, workspaceBytes_),
                                             mem_space_t::DEVICE);
            runtimeBuffer inbound_workspace(asByteSpan(gpuWorkspaceInbound_, workspaceBytes_),
                                            mem_space_t::DEVICE);
            const auto outbound_opts = makeOptions(outbound_workspace);

            auto compress_handle =
                backend_->outboundProcessSlot(slotBuffers{src, dst}, outbound_opts);
            ASSERT_NE(compress_handle, nullptr);
            auto compress_result = waitForCompletion(compress_handle);
            const auto *compress_completion = tryGetCompletedSlotData(compress_result);
            ASSERT_NE(compress_completion, nullptr);
            const size_t num_chunks = numChunksFor(payload_bytes);
            ASSERT_EQ(compress_completion->size, expectedPackedPayloadSize(gpuDst_, num_chunks));
            ASSERT_LE(compress_completion->size, slotBytes_);

            runtimeBuffer roundtrip_dst(asByteSpan(gpuRoundtrip_, payload_bytes),
                                        mem_space_t::DEVICE);

            // The compressed buffer is self-describing: the inbound path reads the packed
            // offsets/sizes header directly, so no chunk-division option is supplied. Only the
            // reported payload size is handed to the receiver, exercising that it is sufficient
            // to fully reconstruct the original data.
            runtimeBuffer compressed(asByteSpan(gpuDst_, compress_completion->size),
                                     mem_space_t::DEVICE);

            auto decompress_opts = makeOptions(inbound_workspace);

            auto decompress_handle =
                backend_->inboundProcessSlot(slotBuffers{compressed, roundtrip_dst},
                                             compress_completion->metadata,
                                             decompress_opts);
            ASSERT_NE(decompress_handle, nullptr);

            auto decompress_result = waitForCompletion(decompress_handle);
            const auto *decompress_completion = tryGetCompletedSlotData(decompress_result);
            ASSERT_NE(decompress_completion, nullptr);
            EXPECT_EQ(decompress_completion->size, payload_bytes);

            std::vector<uint8_t> host_roundtrip(payload_bytes);
            ASSERT_EQ(cudaMemcpy(host_roundtrip.data(),
                                 gpuRoundtrip_,
                                 host_roundtrip.size(),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(host_roundtrip, hostSrc_);
        }

        TEST_P(compressionBackendTest, OneWorstCaseSizedBackendRoundTripsAllAnsDataTypes) {
            runtimeBuffer src(asByteSpan(gpuSrc_, payload_bytes), mem_space_t::DEVICE);
            runtimeBuffer dst(asByteSpan(gpuDst_, slotBytes_), mem_space_t::DEVICE);
            runtimeBuffer outbound_workspace(asByteSpan(gpuWorkspace_, workspaceBytes_),
                                             mem_space_t::DEVICE);
            runtimeBuffer inbound_workspace(asByteSpan(gpuWorkspaceInbound_, workspaceBytes_),
                                            mem_space_t::DEVICE);
            runtimeBuffer roundtrip_dst(asByteSpan(gpuRoundtrip_, payload_bytes),
                                        mem_space_t::DEVICE);

            for (const auto data_type : ans_data_types) {
                SCOPED_TRACE(::testing::Message() << "data_type=" << static_cast<int>(data_type));
                ASSERT_EQ(cudaMemset(gpuDst_, 0, slotBytes_), cudaSuccess);
                ASSERT_EQ(cudaMemset(gpuRoundtrip_, 0, payload_bytes), cudaSuccess);

                auto compress_handle = backend_->outboundProcessSlot(
                    slotBuffers{src, dst}, makeOptions(outbound_workspace, data_type));
                ASSERT_NE(compress_handle, nullptr);
                auto compress_result = waitForCompletion(compress_handle);
                const auto *compress_completion = tryGetCompletedSlotData(compress_result);
                ASSERT_NE(compress_completion, nullptr);
                ASSERT_LE(compress_completion->size, slotBytes_);

                runtimeBuffer compressed(asByteSpan(gpuDst_, compress_completion->size),
                                         mem_space_t::DEVICE);
                auto decompress_handle =
                    backend_->inboundProcessSlot(slotBuffers{compressed, roundtrip_dst},
                                                 compress_completion->metadata,
                                                 makeOptions(inbound_workspace));
                ASSERT_NE(decompress_handle, nullptr);
                auto decompress_result = waitForCompletion(decompress_handle);
                const auto *decompress_completion = tryGetCompletedSlotData(decompress_result);
                ASSERT_NE(decompress_completion, nullptr);
                EXPECT_EQ(decompress_completion->size, payload_bytes);

                std::vector<uint8_t> host_roundtrip(payload_bytes);
                ASSERT_EQ(cudaMemcpy(host_roundtrip.data(),
                                     gpuRoundtrip_,
                                     host_roundtrip.size(),
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
                EXPECT_EQ(host_roundtrip, hostSrc_);
            }
        }

        TEST_P(compressionBackendTest, HighEntropyPackedOutputStaysWithinAdvertisedCapacity) {
            uint32_t random_state = 0xC001CAFE;
            for (auto &value : hostSrc_) {
                random_state = random_state * 1664525U + 1013904223U;
                value = static_cast<uint8_t>(random_state >> 24U);
            }
            ASSERT_EQ(cudaMemcpy(gpuSrc_, hostSrc_.data(), hostSrc_.size(), cudaMemcpyHostToDevice),
                      cudaSuccess);

            runtimeBuffer src(asByteSpan(gpuSrc_, payload_bytes), mem_space_t::DEVICE);
            runtimeBuffer dst(asByteSpan(gpuDst_, slotBytes_), mem_space_t::DEVICE);
            runtimeBuffer workspace(asByteSpan(gpuWorkspace_, workspaceBytes_),
                                    mem_space_t::DEVICE);
            auto *canary_begin = reinterpret_cast<std::byte *>(gpuDst_) + slotBytes_;

            for (const auto data_type : ans_data_types) {
                SCOPED_TRACE(::testing::Message() << "data_type=" << static_cast<int>(data_type));
                ASSERT_EQ(cudaMemset(gpuDst_, 0, slotBytes_), cudaSuccess);
                ASSERT_EQ(
                    cudaMemset(canary_begin, destination_canary_value, destination_canary_bytes),
                    cudaSuccess);

                auto compress_handle = backend_->outboundProcessSlot(
                    slotBuffers{src, dst}, makeOptions(workspace, data_type));
                ASSERT_NE(compress_handle, nullptr);
                auto completion_result = waitForCompletion(compress_handle);
                const auto *completion = tryGetCompletedSlotData(completion_result);
                ASSERT_NE(completion, nullptr);
                EXPECT_LE(completion->size, slotBytes_);
                EXPECT_EQ(completion->size,
                          expectedPackedPayloadSize(gpuDst_, numChunksFor(payload_bytes)));

                std::array<uint8_t, destination_canary_bytes> canary{};
                ASSERT_EQ(
                    cudaMemcpy(canary.data(), canary_begin, canary.size(), cudaMemcpyDeviceToHost),
                    cudaSuccess);
                EXPECT_TRUE(std::all_of(canary.begin(), canary.end(), [](uint8_t value) {
                    return value == destination_canary_value;
                }));
            }
        }

        TEST_P(compressionBackendTest, OutboundInboundRoundTripWithSmallerRuntimePayload) {
            // The configured payload is an upper bound. A final service slot can be smaller and
            // must still fit the workspace provisioned from that upper bound. The 1152-byte
            // shortfall mirrors one KV token in the failing SGLang workload.
            constexpr size_t runtime_payload_shortfall_bytes = 1152;
            constexpr size_t runtime_payload_bytes =
                payload_bytes - runtime_payload_shortfall_bytes;
            const size_t expected_num_chunks = numChunksFor(runtime_payload_bytes);

            runtimeBuffer src(asByteSpan(gpuSrc_, runtime_payload_bytes), mem_space_t::DEVICE);
            runtimeBuffer dst(asByteSpan(gpuDst_, slotBytes_), mem_space_t::DEVICE);
            runtimeBuffer outbound_workspace(asByteSpan(gpuWorkspace_, workspaceBytes_),
                                             mem_space_t::DEVICE);
            runtimeBuffer inbound_workspace(asByteSpan(gpuWorkspaceInbound_, workspaceBytes_),
                                            mem_space_t::DEVICE);

            auto compress_handle = backend_->outboundProcessSlot(slotBuffers{src, dst},
                                                                 makeOptions(outbound_workspace));
            ASSERT_NE(compress_handle, nullptr);
            auto compress_result = waitForCompletion(compress_handle);
            const auto *compress_completion = tryGetCompletedSlotData(compress_result);
            ASSERT_NE(compress_completion, nullptr);

            const size_t header_bytes = 2 * sizeof(size_t) * expected_num_chunks;
            EXPECT_GT(compress_completion->size, header_bytes);
            EXPECT_LE(compress_completion->size, slotBytes_);
            EXPECT_EQ(compress_completion->size,
                      expectedPackedPayloadSize(gpuDst_, expected_num_chunks));

            runtimeBuffer roundtrip_dst(asByteSpan(gpuRoundtrip_, runtime_payload_bytes),
                                        mem_space_t::DEVICE);
            runtimeBuffer compressed(asByteSpan(gpuDst_, compress_completion->size),
                                     mem_space_t::DEVICE);

            auto decompress_opts = makeOptions(inbound_workspace);
            auto decompress_handle =
                backend_->inboundProcessSlot(slotBuffers{compressed, roundtrip_dst},
                                             compress_completion->metadata,
                                             decompress_opts);
            ASSERT_NE(decompress_handle, nullptr);

            auto decompress_result = waitForCompletion(decompress_handle);
            const auto *decompress_completion = tryGetCompletedSlotData(decompress_result);
            ASSERT_NE(decompress_completion, nullptr);
            EXPECT_EQ(decompress_completion->size, runtime_payload_bytes);

            std::vector<uint8_t> host_roundtrip(runtime_payload_bytes);
            ASSERT_EQ(cudaMemcpy(host_roundtrip.data(),
                                 gpuRoundtrip_,
                                 host_roundtrip.size(),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_TRUE(std::equal(host_roundtrip.begin(), host_roundtrip.end(), hostSrc_.begin()));
        }

        TEST_P(compressionBackendTest, OutboundInboundRoundTripWithNonAlignedPayload) {
            constexpr size_t k_remainder_bytes =
                12340; // even (FP16-safe), non-zero, non-power-of-2
            constexpr size_t nvcomp_small_chunk_size_bytes = 1U << 14;
            constexpr size_t k_non_aligned_payload_bytes =
                3 * nvcomp_small_chunk_size_bytes + k_remainder_bytes;
            constexpr size_t expected_num_chunks = 4;
            static_assert(k_non_aligned_payload_bytes % nvcomp_small_chunk_size_bytes != 0,
                          "payload must not divide evenly into the nvcomp chunk size");

            nixlMarshalCompressConfig cfg{};
            cfg.algo = GetParam();
            auto backend = compressionBackend::createBackend(cfg, k_non_aligned_payload_bytes);
            ASSERT_NE(backend, nullptr);

            const auto mem_reqs = backend->getSlotMemoryRequirements();
            const auto overhead_it = mem_reqs.opts.find(option_t::SLOT_OVERHEAD);
            ASSERT_NE(overhead_it, mem_reqs.opts.end());
            const auto *slot_overhead =
                std::get_if<SlotOverhead::memoryRequirements>(&overhead_it->second);
            ASSERT_NE(slot_overhead, nullptr);

            const auto ws_req_it = mem_reqs.opts.find(option_t::WRITEABLE_WORKSPACE_MEMORY);
            ASSERT_NE(ws_req_it, mem_reqs.opts.end());
            const auto *ws_req =
                std::get_if<WriteableWorkspaceMemory::memoryRequirements>(&ws_req_it->second);
            ASSERT_NE(ws_req, nullptr);

            const size_t slot_bytes = k_non_aligned_payload_bytes + slot_overhead->slotOverheadSize;
            const size_t ws_bytes = ws_req->slotWorkspaceSize;

            struct cudaScopedAlloc {
                void *ptr = nullptr;

                ~cudaScopedAlloc() {
                    if (ptr) {
                        cudaFree(ptr);
                    }
                }
            };

            cudaScopedAlloc gpu_src, gpu_dst, gpu_roundtrip, gpu_outbound_ws, gpu_inbound_ws,
                gpu_local_ref;
            ASSERT_EQ(cudaMalloc(&gpu_src.ptr, k_non_aligned_payload_bytes), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&gpu_dst.ptr, slot_bytes), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&gpu_roundtrip.ptr, k_non_aligned_payload_bytes), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&gpu_outbound_ws.ptr, ws_bytes), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&gpu_inbound_ws.ptr, ws_bytes), cudaSuccess);

            std::vector<uint8_t> host_payload(k_non_aligned_payload_bytes);
            for (size_t i = 0; i < k_non_aligned_payload_bytes; ++i) {
                host_payload[i] = (i * 4) % 256;
            }
            ASSERT_EQ(
                cudaMemcpy(
                    gpu_src.ptr, host_payload.data(), host_payload.size(), cudaMemcpyHostToDevice),
                cudaSuccess);
            ASSERT_EQ(cudaMemset(gpu_dst.ptr, 0, slot_bytes), cudaSuccess);
            ASSERT_EQ(cudaMemset(gpu_roundtrip.ptr, 0, k_non_aligned_payload_bytes), cudaSuccess);
            ASSERT_EQ(cudaMemset(gpu_inbound_ws.ptr, 0xCD, ws_bytes), cudaSuccess);

            std::vector<uint8_t> host_local_ref;
            if (isDelta()) {
                ASSERT_EQ(cudaMalloc(&gpu_local_ref.ptr, k_non_aligned_payload_bytes), cudaSuccess);
                host_local_ref = host_payload;
                for (size_t i = 0; i < host_local_ref.size(); i += ref_perturb_stride) {
                    host_local_ref[i] ^= ref_perturb_mask;
                }
                ASSERT_EQ(cudaMemcpy(gpu_local_ref.ptr,
                                     host_local_ref.data(),
                                     host_local_ref.size(),
                                     cudaMemcpyHostToDevice),
                          cudaSuccess);
            }

            auto build_opts = [&](void *workspace) {
                process_slot_input_options_t opts{
                    {option_t::WRITEABLE_WORKSPACE_MEMORY,
                     WriteableWorkspaceMemory::processSlotInput{
                         runtimeBuffer(asByteSpan(workspace, ws_bytes), mem_space_t::DEVICE)}},
                    {option_t::USER_CUDA_STREAM, UserCudaStream::processSlotInput{stream_}},
                };
                if (isDelta()) {
                    opts.emplace(option_t::READ_ONLY_REFERENCE_STRUCTURED_MEMORY,
                                 ReadOnlyReferenceStructuredMemory::processSlotInput{
                                     runtimeBuffer(
                                         asByteSpan(gpu_local_ref.ptr, k_non_aligned_payload_bytes),
                                         mem_space_t::DEVICE),
                                     ref_element_size});
                }
                return opts;
            };

            runtimeBuffer src(asByteSpan(gpu_src.ptr, k_non_aligned_payload_bytes),
                              mem_space_t::DEVICE);
            runtimeBuffer dst(asByteSpan(gpu_dst.ptr, slot_bytes), mem_space_t::DEVICE);

            auto compress_handle = backend->outboundProcessSlot(slotBuffers{src, dst},
                                                                build_opts(gpu_outbound_ws.ptr));
            ASSERT_NE(compress_handle, nullptr);
            auto compress_result = waitForCompletion(compress_handle);
            const auto *compress_completion = tryGetCompletedSlotData(compress_result);
            ASSERT_NE(compress_completion, nullptr);

            const size_t header_bytes = 2 * sizeof(size_t) * expected_num_chunks;
            EXPECT_GT(compress_completion->size, header_bytes);
            EXPECT_LE(compress_completion->size, slot_bytes);
            EXPECT_EQ(compress_completion->size,
                      expectedPackedPayloadSize(gpu_dst.ptr, expected_num_chunks));

            runtimeBuffer roundtrip_dst(asByteSpan(gpu_roundtrip.ptr, k_non_aligned_payload_bytes),
                                        mem_space_t::DEVICE);

            runtimeBuffer compressed(asByteSpan(gpu_dst.ptr, compress_completion->size),
                                     mem_space_t::DEVICE);
            auto decompress_opts = build_opts(gpu_inbound_ws.ptr);
            auto decompress_handle =
                backend->inboundProcessSlot(slotBuffers{compressed, roundtrip_dst},
                                            compress_completion->metadata,
                                            decompress_opts);
            ASSERT_NE(decompress_handle, nullptr);
            auto decompress_result = waitForCompletion(decompress_handle);
            const auto *decompress_completion = tryGetCompletedSlotData(decompress_result);
            ASSERT_NE(decompress_completion, nullptr);
            EXPECT_EQ(decompress_completion->size, k_non_aligned_payload_bytes);

            std::vector<uint8_t> host_roundtrip(k_non_aligned_payload_bytes);
            ASSERT_EQ(cudaMemcpy(host_roundtrip.data(),
                                 gpu_roundtrip.ptr,
                                 host_roundtrip.size(),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(host_roundtrip, host_payload);
        }

        TEST_P(compressionBackendTest, OutboundFailsWhenWorkspaceTooSmall) {
            runtimeBuffer src(asByteSpan(gpuSrc_, payload_bytes), mem_space_t::DEVICE);
            runtimeBuffer dst(asByteSpan(gpuDst_, slotBytes_), mem_space_t::DEVICE);

            // The ANS_DELTA path first carves a full-payload staging buffer out of the
            // workspace, so leave room for that (backed by the real allocation) while still
            // starving the nvcomp bookkeeping/temp region that follows it. For plain ANS a
            // handful of bytes is already too small for the per-chunk pointer/size arrays.
            // The bookkeeping region is only a few KB, so starve it with a tiny amount rather
            // than a value that would exceed the (surprisingly small) real workspace.
            constexpr size_t starved_bookkeeping_bytes = 256;
            const size_t insufficient_workspace_bytes =
                (isDelta() ? payload_bytes : 0) + starved_bookkeeping_bytes;
            ASSERT_LT(insufficient_workspace_bytes, workspaceBytes_);
            runtimeBuffer tiny_workspace(asByteSpan(gpuWorkspace_, insufficient_workspace_bytes),
                                         mem_space_t::DEVICE);

            EXPECT_THROW(
                backend_->outboundProcessSlot(slotBuffers{src, dst}, makeOptions(tiny_workspace)),
                std::runtime_error);
        }

        INSTANTIATE_TEST_SUITE_P(
            Algos,
            compressionBackendTest,
            ::testing::Values(nixl_marshal_compress_algo_t::ANS,
                              nixl_marshal_compress_algo_t::ANS_DELTA),
            [](const ::testing::TestParamInfo<nixl_marshal_compress_algo_t> &info) {
                return algoName(info.param);
            });

    } // namespace marshals
} // namespace services
} // namespace gtest
