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
#ifndef TEST_GTEST_UNIT_SERVICES_MARSHALS_MOCK_MARSHAL_BACKEND_H
#define TEST_GTEST_UNIT_SERVICES_MARSHALS_MOCK_MARSHAL_BACKEND_H

#include <gmock/gmock.h>

#include "marshal/marshal_backend.h"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gtest {
namespace services {
    namespace marshals {

        /**
         * GMock of nixlMarshal::backend. Default actions copy device memory and complete
         * immediately so a full service transfer can run. createBackend wraps the instance
         * in NiceMock; add EXPECT_CALL in the test for the methods you care about.
         */
        class mockMarshalBackend : public nixlMarshal::backend {
        private:
            struct passkey {
                explicit passkey() = default;
            };

            std::vector<nixlMarshal::mem_space_t> supportedMemSpaces_{
                nixlMarshal::mem_space_t::DEVICE};

            class inboundHandle final
                : public nixlMarshal::asyncHandleImpl<inboundHandle,
                                                      nixlMarshal::inboundSlotCompletionData> {
                size_t size_;

            public:
                inboundHandle(std::weak_ptr<nixlMarshal::backend> backend, size_t size)
                    : asyncHandleImpl(std::move(backend)),
                      size_(size) {}

                nixlMarshal::slot_completion_result_t<nixlMarshal::inboundSlotCompletionData>
                checkForCompletionImpl() {
                    return nixlMarshal::inboundSlotCompletionData{size_};
                }
            };

            class outboundHandle final
                : public nixlMarshal::asyncHandleImpl<outboundHandle,
                                                      nixlMarshal::outboundSlotCompletionData> {
                size_t size_;

            public:
                outboundHandle(std::weak_ptr<nixlMarshal::backend> backend, size_t size)
                    : asyncHandleImpl(std::move(backend)),
                      size_(size) {}

                nixlMarshal::slot_completion_result_t<nixlMarshal::outboundSlotCompletionData>
                checkForCompletionImpl() {
                    return nixlMarshal::outboundSlotCompletionData{size_};
                }
            };

            static size_t
            copySlot(const nixlMarshal::slotBuffers &buffers) {
                if (buffers.src.space != nixlMarshal::mem_space_t::DEVICE ||
                    buffers.dst.space != nixlMarshal::mem_space_t::DEVICE) {
                    throw std::runtime_error("mockMarshalBackend: only device memory is supported");
                }
                if (buffers.dst.size < buffers.src.size) {
                    throw std::runtime_error(
                        "mockMarshalBackend: destination size smaller than source");
                }
                const auto err = cudaMemcpy(
                    buffers.dst.data, buffers.src.data, buffers.src.size, cudaMemcpyDefault);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("mockMarshalBackend: cudaMemcpy failed: ") +
                        cudaGetErrorString(err));
                }
                return buffers.src.size;
            }

        public:
            [[nodiscard]] static std::shared_ptr<mockMarshalBackend>
            createBackend();

            explicit mockMarshalBackend(passkey) {
                using testing::Return;
                using testing::ReturnRef;
                using testing::_;

                ON_CALL(*this, getSupportedMemSpaces())
                    .WillByDefault(ReturnRef(supportedMemSpaces_));
                ON_CALL(*this, getSlotMemoryRequirements())
                    .WillByDefault(Return(nixlMarshal::memoryRequirements{{}}));
                ON_CALL(*this, inboundProcessSlot(_, _, _))
                    .WillByDefault([this](const nixlMarshal::slotBuffers &buffers,
                                          const std::string &,
                                          const nixlMarshal::process_slot_input_options_t &) {
                        const auto size = copySlot(buffers);
                        return std::make_unique<inboundHandle>(shared_from_this(), size);
                    });
                ON_CALL(*this, outboundProcessSlot(_, _))
                    .WillByDefault([this](const nixlMarshal::slotBuffers &buffers,
                                          const nixlMarshal::process_slot_input_options_t &) {
                        const auto size = copySlot(buffers);
                        return std::make_unique<outboundHandle>(shared_from_this(), size);
                    });
            }

            MOCK_METHOD(const std::vector<nixlMarshal::mem_space_t> &,
                        getSupportedMemSpaces,
                        (),
                        (const, override));
            MOCK_METHOD(std::unique_ptr<nixlMarshal::inbound_async_handle_t>,
                        inboundProcessSlot,
                        (const nixlMarshal::slotBuffers &buffers,
                         const std::string &metadata,
                         const nixlMarshal::process_slot_input_options_t &opts),
                        (override));
            MOCK_METHOD(std::unique_ptr<nixlMarshal::outbound_async_handle_t>,
                        outboundProcessSlot,
                        (const nixlMarshal::slotBuffers &buffers,
                         const nixlMarshal::process_slot_input_options_t &opts),
                        (override));
            MOCK_METHOD(nixlMarshal::memoryRequirements,
                        getSlotMemoryRequirements,
                        (),
                        (const, override));
        };

        inline std::shared_ptr<mockMarshalBackend>
        mockMarshalBackend::createBackend() {
            return std::make_shared<testing::NiceMock<mockMarshalBackend>>(passkey{});
        }

    } // namespace marshals
} // namespace services
} // namespace gtest

#endif // TEST_GTEST_UNIT_SERVICES_MARSHALS_MOCK_MARSHAL_BACKEND_H
