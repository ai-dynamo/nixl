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

#include <future>
#include <vector>

#include "staging_backend.h"

namespace gtest {
namespace services {
    namespace marshals {

        using runtime_buffer_t = nixlMarshal::runtimeBuffer;

        constexpr size_t chunk_size = 1024;
        constexpr uintptr_t fake_src_addr = 0x1000;
        constexpr uintptr_t fake_dst_addr = 0x2000;
        constexpr size_t small_xfer_size = 64;
        constexpr int fake_device_id = 0;

        class stagingBackendTest : public ::testing::Test {
        protected:
            std::shared_ptr<nixlMarshal::stagingBackend> backend_;

            void
            SetUp() override {
                backend_ = nixlMarshal::stagingBackend::createBackend(nixlMarshalStagingConfig{});
            }
        };

        class stagingBackendGpuTest : public ::testing::Test {
        protected:
            std::shared_ptr<nixlMarshal::stagingBackend> backend_;
            cudaStream_t stream_ = nullptr;
            void *gpuSrc_;
            void *gpuDst_;
            runtime_buffer_t src_;
            runtime_buffer_t dst_;

            void
            SetUp() override {
                int device_count = 0;
                cudaGetDeviceCount(&device_count);
                if (device_count == 0) {
                    GTEST_SKIP() << "No CUDA device available";
                }

                ASSERT_EQ(cudaMalloc(&gpuSrc_, chunk_size), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&gpuDst_, chunk_size), cudaSuccess);
                ASSERT_EQ(cudaMemset(gpuSrc_, 0, chunk_size), cudaSuccess);
                ASSERT_EQ(cudaMemset(gpuDst_, 0, chunk_size), cudaSuccess);

                backend_ = nixlMarshal::stagingBackend::createBackend(nixlMarshalStagingConfig{});
                ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);

                int dev_id = 0;
                ASSERT_EQ(cudaGetDevice(&dev_id), cudaSuccess);

                src_ = runtime_buffer_t(
                    absl::Span<std::byte>(reinterpret_cast<std::byte *>(gpuSrc_), chunk_size),
                    nixlMarshal::mem_space_t::DEVICE);
                dst_ = runtime_buffer_t(
                    absl::Span<std::byte>(reinterpret_cast<std::byte *>(gpuDst_), chunk_size),
                    nixlMarshal::mem_space_t::DEVICE);
            }

            void
            TearDown() override {
                if (gpuDst_) {
                    cudaFree(gpuDst_);
                }
                if (gpuSrc_) {
                    cudaFree(gpuSrc_);
                }
                if (stream_) {
                    cudaStreamDestroy(stream_);
                }
            }
        };

        TEST_F(stagingBackendTest, CreateBackendValid) {
            ASSERT_NE(backend_, nullptr);
        }

        TEST_F(stagingBackendTest, GetSupportedMemSpacesValid) {
            auto mems = backend_->getSupportedMemSpaces();
            ASSERT_FALSE(mems.empty());
            EXPECT_EQ(mems.front(), nixlMarshal::mem_space_t::DEVICE);
        }

        TEST_F(stagingBackendTest, GetSlotMemoryRequirementsValid) {
            auto requirements = backend_->getSlotMemoryRequirements();
            EXPECT_TRUE(requirements.opts.empty());
        }

        TEST_F(stagingBackendTest, ProcessSlotWithoutStreamThrows) {
            runtime_buffer_t src(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_src_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::DEVICE);
            runtime_buffer_t dst(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_dst_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::DEVICE);
            EXPECT_THROW(
                backend_->inboundProcessSlot(nixlMarshal::slotBuffers{src, dst}, /*metadata=*/{}),
                std::runtime_error);
        }

        TEST_F(stagingBackendTest, MultipleBackendsCoexistIndependently_Throws) {
            auto backend01 = nixlMarshal::stagingBackend::createBackend(nixlMarshalStagingConfig{});
            auto backend02 = nixlMarshal::stagingBackend::createBackend(nixlMarshalStagingConfig{});
            ASSERT_NE(backend01, nullptr);
            ASSERT_NE(backend02, nullptr);

            runtime_buffer_t src(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_src_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::DEVICE);
            runtime_buffer_t dst(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_dst_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::DEVICE);

            for (const auto &b : {backend_, backend01, backend02}) {
                EXPECT_THROW(
                    b->inboundProcessSlot(nixlMarshal::slotBuffers{src, dst}, /*metadata=*/{}),
                    std::runtime_error);
            }
        }

        TEST_F(stagingBackendTest, ProcessSlotUnsupportedSrcTypeThrows) {
            runtime_buffer_t src(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_src_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::HOST);
            runtime_buffer_t dst(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_dst_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::DEVICE);
            EXPECT_THROW(backend_->outboundProcessSlot(
                             nixlMarshal::slotBuffers{src, dst},
                             {{nixlMarshal::option_t::USER_CUDA_STREAM,
                               nixlMarshal::UserCudaStream::processSlotInput{cudaStream_t{0}}}}),
                         std::runtime_error);
        }

        TEST_F(stagingBackendTest, ProcessSlotUnsupportedDstTypeThrows) {
            runtime_buffer_t src(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_src_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::DEVICE);
            runtime_buffer_t dst(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_dst_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::HOST);
            EXPECT_THROW(backend_->outboundProcessSlot(
                             nixlMarshal::slotBuffers{src, dst},
                             {{nixlMarshal::option_t::USER_CUDA_STREAM,
                               nixlMarshal::UserCudaStream::processSlotInput{cudaStream_t{0}}}}),
                         std::runtime_error);
        }

        TEST_F(stagingBackendTest, ProcessSlotBothTypesUnsupportedThrows) {
            runtime_buffer_t src(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_src_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::HOST);
            runtime_buffer_t dst(absl::Span<std::byte>(reinterpret_cast<std::byte *>(fake_dst_addr),
                                                       small_xfer_size),
                                 nixlMarshal::mem_space_t::HOST);
            EXPECT_THROW(backend_->outboundProcessSlot(
                             nixlMarshal::slotBuffers{src, dst},
                             {{nixlMarshal::option_t::USER_CUDA_STREAM,
                               nixlMarshal::UserCudaStream::processSlotInput{cudaStream_t{0}}}}),
                         std::runtime_error);
        }

        TEST_F(stagingBackendGpuTest, ProcessSlotInboundVram_DataMatchesSourceValid) {
            const uint8_t pattern = 0xAA;

            std::vector<uint8_t> host_src(chunk_size, pattern);
            ASSERT_EQ(cudaMemcpy(gpuSrc_, host_src.data(), chunk_size, cudaMemcpyHostToDevice),
                      cudaSuccess);

            auto handle = backend_->outboundProcessSlot(
                nixlMarshal::slotBuffers{src_, dst_},
                {{nixlMarshal::option_t::USER_CUDA_STREAM,
                  nixlMarshal::UserCudaStream::processSlotInput{stream_}}});
            ASSERT_NE(handle, nullptr);

            auto future = std::async(std::launch::async, [&] {
                std::optional<nixlMarshal::outboundSlotCompletionData> result;
                while (!result.has_value()) {
                    result = handle->checkForCompletion();
                }
                return *result;
            });

            ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready)
                << "Copy did not complete in time";

            auto result = future.get();
            EXPECT_EQ(result.size, chunk_size);
            EXPECT_TRUE(result.options.empty());
            EXPECT_TRUE(result.metadata.empty());

            std::vector<uint8_t> host_dst(chunk_size, 0);
            ASSERT_EQ(cudaMemcpy(host_dst.data(), gpuDst_, chunk_size, cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(host_src, host_dst);
        }

    } // namespace marshals
} // namespace services
} // namespace gtest
