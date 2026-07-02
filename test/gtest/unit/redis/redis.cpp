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

#include "engine_impl.h"

#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gtest::redis {

class mockRedisClient : public iRedisClient {
public:
    void
    setExecutor(std::shared_ptr<redisThreadPoolExecutor> executor) override {
        executor_ = std::move(executor);
    }

    void
    putKeyAsync(std::string_view key,
                uintptr_t,
                size_t,
                std::shared_ptr<std::promise<nixl_status_t>> promise) override {
        putKeys_.emplace_back(key);
        promise->set_value(NIXL_SUCCESS);
    }

    void
    getKeyAsync(std::string_view key,
                uintptr_t,
                size_t,
                std::shared_ptr<std::promise<nixl_status_t>> promise) override {
        getKeys_.emplace_back(key);
        promise->set_value(NIXL_SUCCESS);
    }

    std::optional<bool>
    checkKeyExistsSync(std::string_view key) override {
        checkedKeys_.emplace_back(key);
        return true;
    }

    const std::vector<std::string> &
    putKeys() const {
        return putKeys_;
    }

private:
    std::shared_ptr<redisThreadPoolExecutor> executor_;
    std::vector<std::string> putKeys_;
    std::vector<std::string> getKeys_;
    std::vector<std::string> checkedKeys_;
};

class redisEngineTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        initParams_.localAgent = "redis-test-agent";
        initParams_.type = "REDIS";
        initParams_.customParams = &customParams_;
        initParams_.enableProgTh = false;
        initParams_.pthrDelay = 0;
        initParams_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;

        mockClient_ = std::make_shared<mockRedisClient>();
        engine_ = std::make_unique<nixlRedisKVEngineImpl>(&initParams_, mockClient_);
    }

    nixlBackendInitParams initParams_;
    nixl_b_params_t customParams_;
    std::shared_ptr<mockRedisClient> mockClient_;
    std::unique_ptr<nixlRedisKVEngineImpl> engine_;
};

TEST_F(redisEngineTest, PostXferDoesNotDispatchPartialCommandsWhenLaterKeyIsMissing) {
    std::vector<char> firstBuffer(16, 'a');
    std::vector<char> secondBuffer(16, 'b');

    nixlBlobDesc registeredRemote(0, firstBuffer.size(), 11, "registered-key");
    nixlBackendMD *registeredMetadata = nullptr;
    ASSERT_EQ(engine_->registerMem(registeredRemote, DRAM_SEG, registeredMetadata), NIXL_SUCCESS);
    ASSERT_NE(registeredMetadata, nullptr);

    nixl_meta_dlist_t localDescs(DRAM_SEG);
    localDescs.addDesc(nixlMetaDesc(
        reinterpret_cast<uintptr_t>(firstBuffer.data()), firstBuffer.size(), 1, nullptr));
    localDescs.addDesc(nixlMetaDesc(
        reinterpret_cast<uintptr_t>(secondBuffer.data()), secondBuffer.size(), 2, nullptr));

    nixl_meta_dlist_t remoteDescs(DRAM_SEG);
    remoteDescs.addDesc(nixlMetaDesc(0, firstBuffer.size(), 11, nullptr));
    remoteDescs.addDesc(nixlMetaDesc(0, secondBuffer.size(), 22, nullptr));

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE,
                                localDescs,
                                remoteDescs,
                                initParams_.localAgent,
                                initParams_.localAgent,
                                handle,
                                nullptr),
              NIXL_SUCCESS);

    EXPECT_EQ(engine_->postXfer(
                  NIXL_WRITE, localDescs, remoteDescs, initParams_.localAgent, handle, nullptr),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_TRUE(mockClient_->putKeys().empty());

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(registeredMetadata), NIXL_SUCCESS);
}

} // namespace gtest::redis
