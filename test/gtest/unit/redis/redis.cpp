/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "redis_backend.h"

#include <cstdlib>
#include <deque>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gtest::redis {

class scopedRedisEnvironment {
public:
    scopedRedisEnvironment() {
        save("REDIS_HOST", host_);
        save("REDIS_PORT", port_);
        save("REDIS_USERNAME", username_);
        save("REDIS_PASSWORD", password_);
        save("REDIS_POOL_SIZE", pool_size_);
    }

    ~scopedRedisEnvironment() {
        restore("REDIS_HOST", host_);
        restore("REDIS_PORT", port_);
        restore("REDIS_USERNAME", username_);
        restore("REDIS_PASSWORD", password_);
        restore("REDIS_POOL_SIZE", pool_size_);
    }

    void
    clear() {
        unsetenv("REDIS_HOST");
        unsetenv("REDIS_PORT");
        unsetenv("REDIS_USERNAME");
        unsetenv("REDIS_PASSWORD");
        unsetenv("REDIS_POOL_SIZE");
    }

private:
    static void
    save(const char *name, std::optional<std::string> &value) {
        if (const char *current = std::getenv(name)) {
            value = current;
        }
    }

    static void
    restore(const char *name, const std::optional<std::string> &value) {
        if (value) {
            setenv(name, value->c_str(), 1);
        } else {
            unsetenv(name);
        }
    }

    std::optional<std::string> host_;
    std::optional<std::string> port_;
    std::optional<std::string> username_;
    std::optional<std::string> password_;
    std::optional<std::string> pool_size_;
};

TEST(redisConfigTest, UsesUnauthenticatedDefaultsWhenCredentialsAreAbsent) {
    scopedRedisEnvironment environment;
    environment.clear();
    nixl_b_params_t params;

    const auto config = RedisConfig::fromBackendParams(&params);

    EXPECT_EQ(config.host, "localhost");
    EXPECT_EQ(config.port, 6379);
    EXPECT_TRUE(config.username.empty());
    EXPECT_TRUE(config.password.empty());
    EXPECT_EQ(config.db, 0);
}

TEST(redisConfigTest, BackendParametersOverrideEnvironmentFallbacks) {
    scopedRedisEnvironment environment;
    environment.clear();
    setenv("REDIS_HOST", "environment-host", 1);
    setenv("REDIS_PORT", "6380", 1);
    setenv("REDIS_USERNAME", "environment-user", 1);
    setenv("REDIS_PASSWORD", "environment-password", 1);
    nixl_b_params_t params = {
        {"host", "parameter-host"},
        {"port", "6381"},
        {"username", "parameter-user"},
        {"password", "parameter-password"},
        {"db", "2"},
    };

    const auto config = RedisConfig::fromBackendParams(&params);

    EXPECT_EQ(config.host, "parameter-host");
    EXPECT_EQ(config.port, 6381);
    EXPECT_EQ(config.username, "parameter-user");
    EXPECT_EQ(config.password, "parameter-password");
    EXPECT_EQ(config.db, 2);
}

TEST(redisConfigTest, RejectsAclUsernameWithoutPassword) {
    scopedRedisEnvironment environment;
    environment.clear();
    nixl_b_params_t params = {{"username", "acl-user"}};

    EXPECT_THROW(RedisConfig::fromBackendParams(&params), std::invalid_argument);
}

TEST(redisConfigTest, UsesDefaultPoolSizeWhenAbsent) {
    scopedRedisEnvironment environment;
    environment.clear();
    nixl_b_params_t params;
    const auto config = RedisConfig::fromBackendParams(&params);
    EXPECT_EQ(config.pool_size, 8);
}

TEST(redisConfigTest, BackendParamOverridesPoolSize) {
    scopedRedisEnvironment environment;
    environment.clear();
    nixl_b_params_t params = {{"pool_size", "4"}};
    const auto config = RedisConfig::fromBackendParams(&params);
    EXPECT_EQ(config.pool_size, 4);
}

TEST(redisConfigTest, EnvVarSetsPoolSize) {
    scopedRedisEnvironment environment;
    environment.clear();
    setenv("REDIS_POOL_SIZE", "16", 1);
    nixl_b_params_t params;
    const auto config = RedisConfig::fromBackendParams(&params);
    EXPECT_EQ(config.pool_size, 16);
}

TEST(redisConfigTest, BackendParamOverridesPoolSizeEnvVar) {
    scopedRedisEnvironment environment;
    environment.clear();
    setenv("REDIS_POOL_SIZE", "16", 1);
    nixl_b_params_t params = {{"pool_size", "2"}};
    const auto config = RedisConfig::fromBackendParams(&params);
    EXPECT_EQ(config.pool_size, 2);
}

TEST(redisConfigTest, InvalidPoolSizeFallsBackToDefault) {
    scopedRedisEnvironment environment;
    environment.clear();
    nixl_b_params_t params = {{"pool_size", "bad"}};
    const auto config = RedisConfig::fromBackendParams(&params);
    EXPECT_EQ(config.pool_size, 8);
}

TEST(redisConfigTest, ZeroPoolSizeFallsBackToDefault) {
    scopedRedisEnvironment environment;
    environment.clear();
    nixl_b_params_t params = {{"pool_size", "0"}};
    const auto config = RedisConfig::fromBackendParams(&params);
    EXPECT_EQ(config.pool_size, 8);
}

TEST(redisConfigTest, NegativePoolSizeFallsBackToDefault) {
    scopedRedisEnvironment environment;
    environment.clear();
    nixl_b_params_t params = {{"pool_size", "-1"}};
    const auto config = RedisConfig::fromBackendParams(&params);
    EXPECT_EQ(config.pool_size, 8);
}

class mockRedisClient : public iRedisClient {
public:
    void
    putKeyAsync(std::string_view key,
                uintptr_t,
                size_t,
                std::shared_ptr<std::promise<nixl_status_t>> promise) override {
        putKeys_.emplace_back(key);
        completeOrQueue(std::move(promise));
    }

    void
    getKeyAsync(std::string_view key,
                uintptr_t,
                size_t,
                std::shared_ptr<std::promise<nixl_status_t>> promise) override {
        getKeys_.emplace_back(key);
        completeOrQueue(std::move(promise));
    }

    std::optional<bool>
    checkKeyExistsSync(std::string_view key) override {
        checkedKeys_.emplace_back(key);
        if (existsResults_.empty()) {
            return true;
        }
        auto result = existsResults_.front();
        existsResults_.pop_front();
        return result;
    }

    void
    setCompletionStatus(nixl_status_t status) {
        completionStatus_ = status;
    }

    void
    setCompleteImmediately(bool value) {
        completeImmediately_ = value;
    }

    void
    completePending(nixl_status_t status) {
        for (auto &promise : pendingPromises_) {
            promise->set_value(status);
        }
        pendingPromises_.clear();
    }

    void
    setExistsResults(std::deque<std::optional<bool>> results) {
        existsResults_ = std::move(results);
    }

    const std::vector<std::string> &
    putKeys() const {
        return putKeys_;
    }

    const std::vector<std::string> &
    getKeys() const {
        return getKeys_;
    }

    const std::vector<std::string> &
    checkedKeys() const {
        return checkedKeys_;
    }

private:
    void
    completeOrQueue(std::shared_ptr<std::promise<nixl_status_t>> promise) {
        if (completeImmediately_) {
            promise->set_value(completionStatus_);
        } else {
            pendingPromises_.push_back(std::move(promise));
        }
    }

    nixl_status_t completionStatus_ = NIXL_SUCCESS;
    bool completeImmediately_ = true;
    std::deque<std::optional<bool>> existsResults_;
    std::vector<std::shared_ptr<std::promise<nixl_status_t>>> pendingPromises_;
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
        engine_ = std::make_unique<nixlRedisKVEngine>(&initParams_, mockClient_);
    }

    nixlBackendMD *
    registerRemote(uint64_t dev_id, std::string key, nixl_mem_t type = OBJ_SEG) {
        nixlBackendMD *metadata = nullptr;
        EXPECT_EQ(engine_->registerMem(nixlBlobDesc(0, 16, dev_id, key), type, metadata),
                  NIXL_SUCCESS);
        EXPECT_NE(metadata, nullptr);
        return metadata;
    }

    nixlBackendReqH *
    prepareTransfer(nixl_xfer_op_t operation, nixl_meta_dlist_t &local, nixl_meta_dlist_t &remote) {
        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(
            engine_->prepXfer(operation, local, remote, initParams_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        EXPECT_NE(handle, nullptr);
        return handle;
    }

    nixlBackendInitParams initParams_;
    nixl_b_params_t customParams_;
    std::shared_ptr<mockRedisClient> mockClient_;
    std::unique_ptr<nixlRedisKVEngine> engine_;
};

TEST_F(redisEngineTest, ReportsRedisBackendCapabilities) {
    EXPECT_FALSE(engine_->supportsRemote());
    EXPECT_TRUE(engine_->supportsLocal());
    EXPECT_FALSE(engine_->supportsNotif());
    EXPECT_EQ(engine_->getSupportedMems(), (nixl_mem_list_t{OBJ_SEG, DRAM_SEG}));

    nixlBackendMD *output = reinterpret_cast<nixlBackendMD *>(1);
    EXPECT_EQ(engine_->loadLocalMD(nullptr, output), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(engine_->connect(initParams_.localAgent), NIXL_SUCCESS);
    EXPECT_EQ(engine_->disconnect(initParams_.localAgent), NIXL_SUCCESS);
}

TEST_F(redisEngineTest, RegistersMetadataKeyAndDevIdFallback) {
    auto *namedMetadata = registerRemote(11, "registered-key");
    auto *fallbackMetadata = registerRemote(22, "");

    std::vector<char> firstBuffer(16, 'a');
    std::vector<char> secondBuffer(16, 'b');
    nixl_meta_dlist_t localDescs(DRAM_SEG);
    localDescs.addDesc(nixlMetaDesc(
        reinterpret_cast<uintptr_t>(firstBuffer.data()), firstBuffer.size(), 1, nullptr));
    localDescs.addDesc(nixlMetaDesc(
        reinterpret_cast<uintptr_t>(secondBuffer.data()), secondBuffer.size(), 2, nullptr));
    nixl_meta_dlist_t remoteDescs(OBJ_SEG);
    remoteDescs.addDesc(nixlMetaDesc(0, firstBuffer.size(), 11, nullptr));
    remoteDescs.addDesc(nixlMetaDesc(0, secondBuffer.size(), 22, nullptr));

    auto *handle = prepareTransfer(NIXL_WRITE, localDescs, remoteDescs);
    EXPECT_EQ(engine_->postXfer(
                  NIXL_WRITE, localDescs, remoteDescs, initParams_.localAgent, handle, nullptr),
              NIXL_IN_PROG);
    EXPECT_EQ(mockClient_->putKeys(), (std::vector<std::string>{"registered-key", "22"}));
    EXPECT_EQ(engine_->checkXfer(handle), NIXL_SUCCESS);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(namedMetadata), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(fallbackMetadata), NIXL_SUCCESS);

    nixlBackendMD *unsupported = reinterpret_cast<nixlBackendMD *>(1);
    EXPECT_EQ(engine_->registerMem(nixlBlobDesc(), VRAM_SEG, unsupported), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(unsupported, nullptr);
}

TEST_F(redisEngineTest, QueryMemPreservesFoundMissingAndErrorResponses) {
    mockClient_->setExistsResults({true, false, std::nullopt});
    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(0, 0, 1), "found"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(0, 0, 2), "missing"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(0, 0, 3), ""));

    std::vector<nixl_query_resp_t> responses;
    EXPECT_EQ(engine_->queryMem(descs, responses), NIXL_ERR_BACKEND);
    ASSERT_EQ(responses.size(), 3U);
    EXPECT_TRUE(responses[0].has_value());
    EXPECT_FALSE(responses[1].has_value());
    EXPECT_FALSE(responses[2].has_value());
    EXPECT_EQ(mockClient_->checkedKeys(), (std::vector<std::string>{"found", "missing", "3"}));
}

TEST_F(redisEngineTest, PrepXferRejectsInvalidRequests) {
    nixl_meta_dlist_t emptyLocal(DRAM_SEG);
    nixl_meta_dlist_t emptyRemote(OBJ_SEG);
    nixlBackendReqH *handle = reinterpret_cast<nixlBackendReqH *>(1);
    EXPECT_EQ(engine_->prepXfer(
                  NIXL_WRITE, emptyLocal, emptyRemote, initParams_.localAgent, handle, nullptr),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, reinterpret_cast<nixlBackendReqH *>(1));

    std::vector<char> buffer(16);
    nixl_meta_dlist_t local(DRAM_SEG);
    local.addDesc(
        nixlMetaDesc(reinterpret_cast<uintptr_t>(buffer.data()), buffer.size(), 1, nullptr));
    nixl_meta_dlist_t remote(OBJ_SEG);
    remote.addDesc(nixlMetaDesc(0, buffer.size(), 2, nullptr));

    handle = nullptr;
    EXPECT_EQ(engine_->prepXfer(static_cast<nixl_xfer_op_t>(99),
                                local,
                                remote,
                                initParams_.localAgent,
                                handle,
                                nullptr),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);

    nixl_meta_dlist_t wrongLocal(OBJ_SEG);
    wrongLocal.addDesc(nixlMetaDesc(0, buffer.size(), 1, nullptr));
    EXPECT_EQ(
        engine_->prepXfer(NIXL_READ, wrongLocal, remote, initParams_.localAgent, handle, nullptr),
        NIXL_ERR_INVALID_PARAM);
}

TEST_F(redisEngineTest, PollsReadUntilClientCompletes) {
    auto *metadata = registerRemote(22, "read-key");
    mockClient_->setCompleteImmediately(false);
    std::vector<char> buffer(16);
    nixl_meta_dlist_t local(DRAM_SEG);
    local.addDesc(
        nixlMetaDesc(reinterpret_cast<uintptr_t>(buffer.data()), buffer.size(), 1, nullptr));
    nixl_meta_dlist_t remote(OBJ_SEG);
    remote.addDesc(nixlMetaDesc(0, buffer.size(), 22, nullptr));

    auto *handle = prepareTransfer(NIXL_READ, local, remote);
    EXPECT_EQ(engine_->postXfer(NIXL_READ, local, remote, initParams_.localAgent, handle, nullptr),
              NIXL_IN_PROG);
    EXPECT_EQ(mockClient_->getKeys(), (std::vector<std::string>{"read-key"}));
    EXPECT_EQ(engine_->checkXfer(handle), NIXL_IN_PROG);
    mockClient_->completePending(NIXL_SUCCESS);
    EXPECT_EQ(engine_->checkXfer(handle), NIXL_SUCCESS);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(metadata), NIXL_SUCCESS);
}

TEST_F(redisEngineTest, PropagatesAsyncClientFailure) {
    auto *metadata = registerRemote(22, "failed-key");
    mockClient_->setCompletionStatus(NIXL_ERR_BACKEND);
    std::vector<char> buffer(16);
    nixl_meta_dlist_t local(DRAM_SEG);
    local.addDesc(
        nixlMetaDesc(reinterpret_cast<uintptr_t>(buffer.data()), buffer.size(), 1, nullptr));
    nixl_meta_dlist_t remote(OBJ_SEG);
    remote.addDesc(nixlMetaDesc(0, buffer.size(), 22, nullptr));

    auto *handle = prepareTransfer(NIXL_WRITE, local, remote);
    EXPECT_EQ(engine_->postXfer(NIXL_WRITE, local, remote, initParams_.localAgent, handle, nullptr),
              NIXL_IN_PROG);
    EXPECT_EQ(engine_->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(metadata), NIXL_SUCCESS);
}

TEST_F(redisEngineTest, RejectsNullTransferHandles) {
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine_->postXfer(NIXL_WRITE, local, remote, initParams_.localAgent, handle, nullptr),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine_->checkXfer(nullptr), NIXL_ERR_INVALID_PARAM);
}

TEST_F(redisEngineTest, PostXferDoesNotDispatchPartialCommandsWhenLaterKeyIsMissing) {
    std::vector<char> firstBuffer(16, 'a');
    std::vector<char> secondBuffer(16, 'b');

    auto *registeredMetadata = registerRemote(11, "registered-key", DRAM_SEG);

    nixl_meta_dlist_t localDescs(DRAM_SEG);
    localDescs.addDesc(nixlMetaDesc(
        reinterpret_cast<uintptr_t>(firstBuffer.data()), firstBuffer.size(), 1, nullptr));
    localDescs.addDesc(nixlMetaDesc(
        reinterpret_cast<uintptr_t>(secondBuffer.data()), secondBuffer.size(), 2, nullptr));

    nixl_meta_dlist_t remoteDescs(DRAM_SEG);
    remoteDescs.addDesc(nixlMetaDesc(0, firstBuffer.size(), 11, nullptr));
    remoteDescs.addDesc(nixlMetaDesc(0, secondBuffer.size(), 22, nullptr));

    auto *handle = prepareTransfer(NIXL_WRITE, localDescs, remoteDescs);
    EXPECT_EQ(engine_->postXfer(
                  NIXL_WRITE, localDescs, remoteDescs, initParams_.localAgent, handle, nullptr),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_TRUE(mockClient_->putKeys().empty());

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(registeredMetadata), NIXL_SUCCESS);
}

} // namespace gtest::redis
