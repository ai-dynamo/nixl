/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_TEST_GTEST_UNIT_OBJ_MOCK_S3_CLIENT_H
#define NIXL_TEST_GTEST_UNIT_OBJ_MOCK_S3_CLIENT_H

#include "obj_backend.h"
#include "obj_executor.h"
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace gtest::obj {

class mockS3Client : public iS3Client {
private:
    bool simulateSuccess_ = true;
    std::shared_ptr<asioThreadPoolExecutor> executor_;
    std::vector<std::function<void()>> pendingCallbacks_;
    std::set<std::string> checkedKeys_;

public:
    mockS3Client() = default;

    mockS3Client([[maybe_unused]] nixl_b_params_t *custom_params,
                 std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr) {
        if (executor) {
            executor_ = std::dynamic_pointer_cast<asioThreadPoolExecutor>(executor);
            if (!executor_)
                throw std::runtime_error(
                    "mockS3Client: executor must be an asioThreadPoolExecutor");
        }
    }

    void
    setSimulateSuccess(bool success) {
        simulateSuccess_ = success;
    }

    void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) override {
        executor_ = std::dynamic_pointer_cast<asioThreadPoolExecutor>(executor);
        if (executor && !executor_)
            throw std::runtime_error("mockS3Client: executor must be an asioThreadPoolExecutor");
    }

    void
    putObjectAsync([[maybe_unused]] std::string_view key,
                   [[maybe_unused]] uintptr_t data_ptr,
                   [[maybe_unused]] size_t data_len,
                   [[maybe_unused]] size_t offset,
                   put_object_callback_t callback) override {
        pendingCallbacks_.push_back([callback, this]() { callback(simulateSuccess_); });
    }

    void
    getObjectAsync([[maybe_unused]] std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) override {
        pendingCallbacks_.push_back([callback, data_ptr, data_len, offset, this]() {
            if (simulateSuccess_ && data_ptr && data_len > 0) {
                char *buffer = reinterpret_cast<char *>(data_ptr);
                for (size_t i = 0; i < data_len; ++i) {
                    buffer[i] = static_cast<char>('A' + ((i + offset) % 26));
                }
            }
            callback(simulateSuccess_);
        });
    }

    bool
    checkObjectExists(std::string_view key) override {
        checkedKeys_.insert(std::string(key));
        return simulateSuccess_;
    }

    void
    execAsync() {
        if (!executor_) throw std::runtime_error("mockS3Client::execAsync: executor not set");
        for (auto &callback : pendingCallbacks_) {
            executor_->Submit([callback]() { callback(); });
        }
        pendingCallbacks_.clear();
        executor_->waitUntilIdle();
    }

    size_t
    getPendingCount() const {
        return pendingCallbacks_.size();
    }

    const std::set<std::string> &
    getCheckedKeys() const {
        return checkedKeys_;
    }

    bool
    hasExecutor() const {
        return executor_ != nullptr;
    }

protected:
    std::vector<std::function<void()>> &
    getPendingCallbacks() {
        return pendingCallbacks_;
    }

    bool
    getSimulateSuccess() const {
        return simulateSuccess_;
    }
};

} // namespace gtest::obj

#endif // NIXL_TEST_GTEST_UNIT_OBJ_MOCK_S3_CLIENT_H
