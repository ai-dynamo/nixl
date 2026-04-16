/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mock_s3_client.h"
#include "mock_client_registry.h"
#include "s3_accel/dell/rdma_interface.h"

namespace gtest::obj {
namespace {

    class mockDellS3Client : public mockS3Client, public iDellS3RdmaClient {
    public:
        mockDellS3Client() = default;

        mockDellS3Client([[maybe_unused]] nixl_b_params_t *custom_params,
                         std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr)
            : mockS3Client(custom_params, executor) {}

        void
        putObjectRdmaAsync([[maybe_unused]] std::string_view key,
                           [[maybe_unused]] uintptr_t data_ptr,
                           [[maybe_unused]] size_t data_len,
                           [[maybe_unused]] size_t offset,
                           std::string_view rdma_desc,
                           put_object_callback_t callback) override {
            if (rdma_desc.empty()) {
                getPendingCallbacks().push_back([callback]() { callback(false); });
            } else {
                getPendingCallbacks().push_back(
                    [callback, this]() { callback(getSimulateSuccess()); });
            }
        }

        void
        getObjectRdmaAsync([[maybe_unused]] std::string_view key,
                           uintptr_t data_ptr,
                           size_t data_len,
                           size_t offset,
                           std::string_view rdma_desc,
                           get_object_callback_t callback) override {
            if (rdma_desc.empty()) {
                getPendingCallbacks().push_back([callback]() { callback(false); });
            } else {
                getPendingCallbacks().push_back([callback, data_ptr, data_len, offset, this]() {
                    if (getSimulateSuccess() && data_ptr && data_len > 0) {
                        char *buffer = reinterpret_cast<char *>(data_ptr);
                        for (size_t i = 0; i < data_len; ++i) {
                            buffer[i] = static_cast<char>('A' + ((i + offset) % 26));
                        }
                    }
                    callback(getSimulateSuccess());
                });
            }
        }
    };

    mockClientRegistrar reg_dell("dell", []() { return std::make_shared<mockDellS3Client>(); });

} // anonymous namespace
} // namespace gtest::obj
