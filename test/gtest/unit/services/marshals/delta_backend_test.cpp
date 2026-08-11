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
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "delta_backend.h"

namespace gtest::services::marshals {

constexpr size_t buffer_size = 1024;
constexpr size_t chunk_size = 64;
constexpr std::byte host_fill_pattern_a = std::byte{0xA5};
constexpr std::byte host_fill_pattern_b = std::byte{0x3C};
constexpr std::byte zero_pattern = std::byte{0x00};
constexpr std::chrono::seconds timeout{2};

template<typename CompletionT>
CompletionT
waitForCompletion(nixlMarshal::asyncHandle<CompletionT> &handle) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto result = handle.checkForCompletion(); result.has_value()) {
            return *result;
        }
        std::this_thread::yield();
    }
    throw std::runtime_error("Timed out waiting for delta async handle completion");
}

class deltaBackendTest : public ::testing::Test {
protected:
    std::shared_ptr<nixlMarshal::deltaBackend> backend_;
    cudaStream_t stream_ = nullptr;
    void *gpuSrc_ = nullptr;
    void *gpuDst_ = nullptr;
    void *gpuRef_ = nullptr;

    void
    SetUp() override {
        GTEST_SKIP() << "DeltaBackend not implemented (delta mode disabled)";
        int device_count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }

        ASSERT_EQ(cudaMalloc(&gpuSrc_, buffer_size), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&gpuDst_, buffer_size), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&gpuRef_, buffer_size), cudaSuccess);
        ASSERT_EQ(cudaMemset(gpuSrc_, 0, buffer_size), cudaSuccess);
        ASSERT_EQ(cudaMemset(gpuDst_, 0, buffer_size), cudaSuccess);
        ASSERT_EQ(cudaMemset(gpuRef_, 0, buffer_size), cudaSuccess);

        backend_ = nixlMarshal::deltaBackend::createBackend(nixlMarshalDeltaConfig{});
        ASSERT_NE(backend_, nullptr);
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    void
    TearDown() override {
        if (stream_) {
            ASSERT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
        }
        if (gpuRef_) {
            ASSERT_EQ(cudaFree(gpuRef_), cudaSuccess);
        }
        if (gpuDst_) {
            ASSERT_EQ(cudaFree(gpuDst_), cudaSuccess);
        }
        if (gpuSrc_) {
            ASSERT_EQ(cudaFree(gpuSrc_), cudaSuccess);
        }
    }

    nixlMarshal::slotBuffers
    makeSlotBuffs(size_t size,
                  void *src_data = nullptr,
                  void *dst_data = nullptr,
                  size_t src_offset = 0,
                  size_t dst_size = 0) const {
        void *effective_src = src_data == nullptr ? gpuSrc_ : src_data;
        void *effective_dst = dst_data == nullptr ? gpuDst_ : dst_data;
        const size_t effective_dst_size = dst_size == 0 ? size : dst_size;
        nixlMarshal::runtimeBuffer src(
            absl::Span<std::byte>(static_cast<std::byte *>(effective_src) + src_offset, size),
            nixlMarshal::mem_space_t::DEVICE);
        nixlMarshal::runtimeBuffer dst(
            absl::Span<std::byte>(static_cast<std::byte *>(effective_dst), effective_dst_size),
            nixlMarshal::mem_space_t::DEVICE);
        return nixlMarshal::slotBuffers{src, dst};
    }

    nixlMarshal::process_slot_input_options_t
    makeOpts(size_t element_size,
             void *ref_data,
             size_t ref_size,
             bool include_ref = true,
             bool include_stream = true) const {
        nixlMarshal::process_slot_input_options_t opts;
        if (include_ref) {
            nixlMarshal::runtimeBuffer ref_buf(
                absl::Span<std::byte>(static_cast<std::byte *>(ref_data), ref_size),
                nixlMarshal::mem_space_t::DEVICE);
            opts.emplace(nixlMarshal::option_t::READ_ONLY_REFERENCE_STRUCTURED_MEMORY,
                         nixlMarshal::ReadOnlyReferenceStructuredMemory::processSlotInput{
                             ref_buf, element_size});
        }
        if (include_stream) {
            opts.emplace(nixlMarshal::option_t::USER_CUDA_STREAM,
                         nixlMarshal::UserCudaStream::processSlotInput{stream_});
        }
        return opts;
    }
};

TEST_F(deltaBackendTest, OutboundThenInbound_SameRef_Roundtrip) {
    void *gpu_src_a = nullptr;
    void *gpu_dst_a = nullptr;
    void *gpu_dst_b = nullptr;
    void *gpu_ref = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_src_a, chunk_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gpu_dst_a, chunk_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gpu_dst_b, chunk_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gpu_ref, chunk_size), cudaSuccess);

    ASSERT_EQ(cudaMemset(gpu_src_a, std::to_integer<int>(host_fill_pattern_a), chunk_size),
              cudaSuccess);
    ASSERT_EQ(cudaMemset(gpu_ref, std::to_integer<int>(host_fill_pattern_b), chunk_size),
              cudaSuccess);
    ASSERT_EQ(cudaMemset(gpu_dst_a, 0, chunk_size), cudaSuccess);
    ASSERT_EQ(cudaMemset(gpu_dst_b, 0, chunk_size), cudaSuccess);

    // Transaction A (outbound): srcA XOR ref -> dstA
    auto handle_a = backend_->outboundProcessSlot(makeSlotBuffs(chunk_size, gpu_src_a, gpu_dst_a),
                                                  makeOpts(4, gpu_ref, chunk_size));
    ASSERT_NE(handle_a, nullptr);
    const auto completion_a = waitForCompletion(*handle_a);

    EXPECT_EQ(completion_a.size, chunk_size);
    EXPECT_TRUE(completion_a.options.empty());

    // Transaction B (inbound): dstA (now srcB) XOR ref -> dstB; should equal hostSrcA.
    auto handle_b = backend_->inboundProcessSlot(makeSlotBuffs(chunk_size, gpu_dst_a, gpu_dst_b),
                                                 "unused-metadata",
                                                 makeOpts(4, gpu_ref, chunk_size));
    ASSERT_NE(handle_b, nullptr);
    const auto completion_b = waitForCompletion(*handle_b);
    EXPECT_EQ(completion_b.size, chunk_size);

    std::vector<std::byte> host_dst_b(chunk_size, zero_pattern);
    ASSERT_EQ(cudaMemcpy(host_dst_b.data(), gpu_dst_b, chunk_size, cudaMemcpyDeviceToHost),
              cudaSuccess);
    std::vector<std::byte> expected(chunk_size, host_fill_pattern_a);
    EXPECT_EQ(host_dst_b, expected);

    ASSERT_EQ(cudaFree(gpu_ref), cudaSuccess);
    ASSERT_EQ(cudaFree(gpu_dst_b), cudaSuccess);
    ASSERT_EQ(cudaFree(gpu_dst_a), cudaSuccess);
    ASSERT_EQ(cudaFree(gpu_src_a), cudaSuccess);
}

} // namespace gtest::services::marshals
