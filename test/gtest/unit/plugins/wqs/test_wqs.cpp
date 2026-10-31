// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include "wqs_backend.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <unordered_map>

namespace {
using namespace std::chrono_literals;

nixl_b_params_t
paramsFor(const nixlWqsConfig &config) {
    return {{"unix_path", config.unixPath},
            {"staging_pool_bytes", std::to_string(config.stagingPoolBytes)},
            {"max_inflight", std::to_string(config.maxInflight)},
            {"max_batch_entries", std::to_string(config.maxBatchEntries)},
            {"payload_bytes", std::to_string(config.payloadBytes)},
            {"key_prefix", config.keyPrefix}};
}

class fakeWqsClient final : public iWqsClient {
public:
    int64_t
    getSize(const std::string &key) override {
        events.push_back("size:" + key);
        if (throwSize) {
            throw std::runtime_error("injected getSize failure");
        }
        if (sizeError) {
            return -1;
        }
        const auto it = objects.find(key);
        return it == objects.end() ? 0 : static_cast<int64_t>(it->second.size());
    }

    std::vector<int>
    read(const std::vector<std::string> &keys,
         const std::vector<void *> &buffers,
         const std::vector<size_t> &sizes) override {
        {
            std::unique_lock lock(gateMutex);
            if (blockReads) {
                entered.set_value();
                gate.wait(lock, [this] { return !blockReads; });
            }
        }
        ++readBatches;
        std::vector<int> results;
        for (size_t i = 0; i < keys.size(); ++i) {
            events.push_back("read:" + keys[i]);
            readSizes.push_back(sizes[i]);
            const auto &value = objects.at(keys[i]);
            std::memcpy(buffers[i], value.data(), std::min(value.size(), sizes[i]));
            results.push_back(shortRead ? static_cast<int>(sizes[i] - 1) :
                                          static_cast<int>(sizes[i]));
        }
        if (badResultCount) {
            results.pop_back();
        }
        return results;
    }

    std::vector<int>
    write(const std::vector<std::string> &keys,
          const std::vector<void *> &buffers,
          const std::vector<size_t> &sizes) override {
        ++writeBatches;
        if (writeError) {
            return std::vector<int>(keys.size(), -1);
        }
        for (size_t i = 0; i < keys.size(); ++i) {
            events.push_back("write:" + keys[i]);
            objects[keys[i]] = std::string(static_cast<char *>(buffers[i]), sizes[i]);
        }
        return std::vector<int>(keys.size(), 0);
    }

    void
    unblock() {
        std::lock_guard lock(gateMutex);
        blockReads = false;
        gate.notify_all();
    }

    std::unordered_map<std::string, std::string> objects;
    std::vector<std::string> events;
    std::vector<size_t> readSizes;
    size_t readBatches = 0;
    size_t writeBatches = 0;
    bool shortRead = false;
    bool badResultCount = false;
    bool writeError = false;
    bool throwSize = false;
    bool sizeError = false;
    bool blockReads = false;
    std::promise<void> entered;
    std::mutex gateMutex;
    std::condition_variable gate;
};

class wqsEngineTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        nixlWqsConfig config;
        config.stagingPoolBytes = size_t{2} << 20;
        initialize(config);
    }

    void
    TearDown() override {
        client->unblock();
        for (auto *handle : handles) {
            if (handle) {
                EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
            }
        }
        engine.reset();
    }

    void
    initialize(const nixlWqsConfig &config) {
        engine.reset();
        params = paramsFor(config);
        init.localAgent = "wqs-test";
        init.type = "WQS";
        init.customParams = &params;
        init.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        client = std::make_shared<fakeWqsClient>();
        pool = std::make_shared<nixlWqsBufferPool>(config.stagingPoolBytes, false);
        session = std::make_shared<nixlWqsSession>(config, client, pool);
        engine = std::make_unique<nixlWqsEngine>(&init, session);
    }

    void
    descriptors(const std::vector<std::string> &keys, size_t bytes = 17) {
        storage.assign(keys.size() * bytes, 'x');
        nixlBackendMD *dram = nullptr;
        ASSERT_EQ(
            engine->registerMem(
                nixlBlobDesc(reinterpret_cast<uintptr_t>(storage.data()), storage.size(), 0, ""),
                DRAM_SEG,
                dram),
            NIXL_SUCCESS);
        dramMetadata = dram;
        for (size_t i = 0; i < keys.size(); ++i) {
            nixlBackendMD *object = nullptr;
            ASSERT_EQ(engine->registerMem(nixlBlobDesc(0, bytes, i + 1, keys[i]), OBJ_SEG, object),
                      NIXL_SUCCESS);
            local.addDesc(nixlMetaDesc(
                reinterpret_cast<uintptr_t>(storage.data() + i * bytes), bytes, 0, dram));
            remote.addDesc(nixlMetaDesc(0, bytes, i + 1, object));
        }
    }

    nixlBackendReqH *
    prepare(nixl_xfer_op_t operation) {
        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(engine->prepXfer(operation, local, remote, init.localAgent, handle),
                  NIXL_SUCCESS);
        if (handle) {
            handles.push_back(handle);
        }
        return handle;
    }

    nixl_status_t
    wait(nixlBackendReqH *handle) {
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        nixl_status_t status;
        do {
            status = engine->checkXfer(handle);
            std::this_thread::yield();
        } while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline);
        return status;
    }

    nixl_status_t
    transfer(nixl_xfer_op_t operation, nixlBackendReqH *&handle) {
        EXPECT_EQ(engine->postXfer(operation, local, remote, init.localAgent, handle),
                  NIXL_IN_PROG);
        return wait(handle);
    }

    nixlWqsConfig
    config() const {
        return session->config();
    }

    nixl_b_params_t params;
    nixlBackendInitParams init;
    std::shared_ptr<fakeWqsClient> client;
    std::shared_ptr<nixlWqsBufferPool> pool;
    std::shared_ptr<nixlWqsSession> session;
    std::unique_ptr<nixlWqsEngine> engine;
    nixl_meta_dlist_t local{DRAM_SEG};
    nixl_meta_dlist_t remote{OBJ_SEG};
    std::vector<char> storage;
    std::vector<nixlBackendReqH *> handles;
    nixlBackendMD *dramMetadata = nullptr;
};

TEST(wqsConfigTest, RejectsInvalidAndUnknownParameters) {
    for (const auto &value : {"0", "-1", "2x", "", "99999999999999999999999999999"}) {
        EXPECT_THROW((void)nixlWqsConfig::fromParams({{"max_inflight", value}}),
                     std::invalid_argument);
    }
    EXPECT_THROW((void)nixlWqsConfig::fromParams({{"staging_pool_bytes", "4096"}}),
                 std::invalid_argument);
    EXPECT_THROW((void)nixlWqsConfig::fromParams({{"unix_path", "relative"}}),
                 std::invalid_argument);
    EXPECT_THROW((void)nixlWqsConfig::fromParams({{"unix_path", "/" + std::string(108, 'x')}}),
                 std::invalid_argument);
    EXPECT_THROW((void)nixlWqsConfig::fromParams({{"payload_bytes", "16"}}), std::invalid_argument);
    EXPECT_THROW((void)nixlWqsConfig::fromParams({{"typo", "1"}}), std::invalid_argument);
}

TEST(wqsBufferTest, AlignsAllocationsAndRejectsOverflow) {
    nixlWqsBufferPool pool(1024, false);
    EXPECT_EQ(pool.allocate(17), pool.data());
    EXPECT_EQ(pool.allocate(17), static_cast<char *>(pool.data()) + 512);
    EXPECT_FALSE(pool.canFit(1));
    EXPECT_THROW((void)pool.allocate(1), std::length_error);
    pool.reset();
    EXPECT_EQ(pool.allocate(1024), pool.data());
}

TEST_F(wqsEngineTest, SupportsOnlyLocalDramAndObjects) {
    EXPECT_TRUE(engine->supportsLocal());
    EXPECT_FALSE(engine->supportsRemote());
    EXPECT_FALSE(engine->supportsNotif());
    EXPECT_EQ(engine->connect(init.localAgent), NIXL_SUCCESS);
    EXPECT_EQ(engine->connect("other"), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(engine->getSupportedMems(), (nixl_mem_list_t{DRAM_SEG, OBJ_SEG}));
}

TEST_F(wqsEngineTest, ReadsWithoutQueryMemAndUsesActualLengths) {
    descriptors({"a", "b"});
    client->objects = {{"a", std::string(17, 'a')}, {"b", std::string(17, 'b')}};
    auto *handle = prepare(NIXL_READ);
    EXPECT_TRUE(client->events.empty());
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_SUCCESS);
    EXPECT_EQ(client->events, (std::vector<std::string>{"size:a", "size:b", "read:a", "read:b"}));
    EXPECT_EQ(client->readSizes, (std::vector<size_t>{17, 17}));
    EXPECT_EQ(std::string(storage.begin(), storage.end()),
              std::string(17, 'a') + std::string(17, 'b'));
}

TEST_F(wqsEngineTest, QueriesSizeAgainWhenReadHandleIsReposted) {
    descriptors({"a"});
    client->objects["a"] = std::string(17, 'a');
    auto *handle = prepare(NIXL_READ);
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_SUCCESS);
    client->objects["a"] = std::string(18, 'b');
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_ERR_MISMATCH);
    EXPECT_EQ(client->readBatches, 1);
    EXPECT_EQ(std::count(client->events.begin(), client->events.end(), "size:a"), 2);
}

TEST_F(wqsEngineTest, QueryMemDoesNotCacheTheNextReadLength) {
    descriptors({"a"});
    client->objects["a"] = std::string(17, 'a');
    nixl_reg_dlist_t query(OBJ_SEG);
    query.addDesc(nixlBlobDesc(0, 0, 0, "a"));
    std::vector<nixl_query_resp_t> response;
    ASSERT_EQ(engine->queryMem(query, response), NIXL_SUCCESS);
    ASSERT_EQ(response.size(), 1);
    ASSERT_TRUE(response[0]);
    EXPECT_EQ(response[0]->at("size"), "17");
    auto *handle = prepare(NIXL_READ);
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_SUCCESS);
    EXPECT_EQ(std::count(client->events.begin(), client->events.end(), "size:a"), 2);
}

TEST_F(wqsEngineTest, RejectsDifferentStoredLengthsBeforeReading) {
    descriptors({"a", "b"});
    client->objects = {{"a", std::string(17, 'a')}, {"b", std::string(18, 'b')}};
    auto *handle = prepare(NIXL_READ);
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_ERR_MISMATCH);
    EXPECT_EQ(client->readBatches, 0);
    EXPECT_EQ(storage, std::vector<char>(34, 'x'));
}

TEST_F(wqsEngineTest, MissingKeyFailsWithoutDataRead) {
    descriptors({"missing"});
    auto *handle = prepare(NIXL_READ);
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(client->readBatches, 0);
}

TEST_F(wqsEngineTest, QueryReturnsSizeOrMissAndPreservesNegativeErrors) {
    client->objects["a"] = std::string(17, 'a');
    nixl_reg_dlist_t query(OBJ_SEG);
    query.addDesc(nixlBlobDesc(0, 0, 0, "a"));
    query.addDesc(nixlBlobDesc(0, 0, 0, "missing"));
    std::vector<nixl_query_resp_t> response;
    ASSERT_EQ(engine->queryMem(query, response), NIXL_SUCCESS);
    ASSERT_EQ(response.size(), 2);
    ASSERT_TRUE(response[0]);
    EXPECT_EQ(response[0]->at("size"), "17");
    EXPECT_FALSE(response[1]);
    client->sizeError = true;
    EXPECT_EQ(engine->queryMem(query, response), NIXL_ERR_BACKEND);
    ASSERT_EQ(response.size(), 2);
    EXPECT_FALSE(response[0]);
    EXPECT_FALSE(response[1]);
}

TEST_F(wqsEngineTest, SplitsReadsByEntryLimitAndQueriesEveryObject) {
    auto limits = config();
    limits.maxBatchEntries = 1;
    initialize(limits);
    descriptors({"a", "b"});
    client->objects = {{"a", std::string(17, 'a')}, {"b", std::string(17, 'b')}};
    auto *handle = prepare(NIXL_READ);
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_SUCCESS);
    EXPECT_EQ(client->readBatches, 2);
    EXPECT_EQ(client->events, (std::vector<std::string>{"size:a", "read:a", "size:b", "read:b"}));
}

TEST_F(wqsEngineTest, ReadLengthMismatchDoesNotCopyStagingData) {
    descriptors({"a"});
    client->objects["a"] = std::string(17, 'a');
    client->shortRead = true;
    auto *handle = prepare(NIXL_READ);
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_ERR_MISMATCH);
    EXPECT_EQ(storage, std::vector<char>(17, 'x'));
}

TEST_F(wqsEngineTest, RejectsMalformedReadResults) {
    descriptors({"a"});
    client->objects["a"] = std::string(17, 'a');
    client->badResultCount = true;
    auto *handle = prepare(NIXL_READ);
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(storage, std::vector<char>(17, 'x'));
}

TEST_F(wqsEngineTest, WritesUnalignedLogicalValuesWithoutSizeQueries) {
    descriptors({"a", "b"});
    auto *handle = prepare(NIXL_WRITE);
    EXPECT_EQ(transfer(NIXL_WRITE, handle), NIXL_SUCCESS);
    EXPECT_EQ(client->objects.at("a"), std::string(17, 'x'));
    EXPECT_EQ(client->objects.at("b"), std::string(17, 'x'));
    EXPECT_EQ(client->events, (std::vector<std::string>{"write:a", "write:b"}));
}

TEST_F(wqsEngineTest, ReportsWriteFailures) {
    descriptors({"a"});
    client->writeError = true;
    auto *handle = prepare(NIXL_WRITE);
    EXPECT_EQ(transfer(NIXL_WRITE, handle), NIXL_ERR_BACKEND);
}

TEST_F(wqsEngineTest, SplitsBatchesByPoolCapacity) {
    descriptors({"a", "b", "c"}, size_t{1} << 20);
    auto *handle = prepare(NIXL_WRITE);
    EXPECT_EQ(transfer(NIXL_WRITE, handle), NIXL_SUCCESS);
    EXPECT_EQ(client->writeBatches, 2);
    EXPECT_EQ(client->objects.at("c").size(), size_t{1} << 20);
}

TEST_F(wqsEngineTest, SplitsBatchesByWirePayload) {
    auto limits = config();
    limits.payloadBytes = 128;
    initialize(limits);
    descriptors({std::string(48, 'a'), std::string(48, 'b')});
    auto *handle = prepare(NIXL_WRITE);
    EXPECT_EQ(transfer(NIXL_WRITE, handle), NIXL_SUCCESS);
    EXPECT_EQ(client->writeBatches, 2);
}

TEST_F(wqsEngineTest, AppliesKeyPrefixToReadsAndWrites) {
    auto limits = config();
    limits.keyPrefix = "model/";
    initialize(limits);
    descriptors({"a"});
    auto *write = prepare(NIXL_WRITE);
    EXPECT_EQ(transfer(NIXL_WRITE, write), NIXL_SUCCESS);
    EXPECT_EQ(client->objects.at("model/a"), std::string(17, 'x'));
    auto *read = prepare(NIXL_READ);
    EXPECT_EQ(transfer(NIXL_READ, read), NIXL_SUCCESS);
    EXPECT_NE(std::find(client->events.begin(), client->events.end(), "size:model/a"),
              client->events.end());
}

TEST_F(wqsEngineTest, RejectsInvalidObjectRegistrations) {
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 17, 1, ""), OBJ_SEG, md), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(1, 17, 1, "a"), OBJ_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, pool->size() + 1, 1, "a"), OBJ_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 17, 1, "a"), VRAM_SEG, md),
              NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 17, 1, "a"), OBJ_SEG, md), NIXL_SUCCESS);
    nixlBackendMD *duplicate = nullptr;
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 17, 1, "b"), OBJ_SEG, duplicate),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

TEST_F(wqsEngineTest, RejectsPartialObjectReadsAndOutOfRangeDram) {
    descriptors({"a"});
    nixlBackendReqH *handle = nullptr;
    remote[0].addr = 1;
    EXPECT_EQ(engine->prepXfer(NIXL_READ, local, remote, init.localAgent, handle),
              NIXL_ERR_INVALID_PARAM);
    remote[0].addr = 0;
    ++local[0].addr;
    EXPECT_EQ(engine->prepXfer(NIXL_READ, local, remote, init.localAgent, handle),
              NIXL_ERR_INVALID_PARAM);
}

TEST_F(wqsEngineTest, RejectsDuplicateKeysWithinAWrite) {
    descriptors({"a", "a"});
    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, init.localAgent, handle),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_TRUE(client->events.empty());
}

TEST_F(wqsEngineTest, InvalidatesPreparedMetadataAfterDeregistration) {
    descriptors({"a"});
    nixlBackendMD *clone = nullptr;
    ASSERT_EQ(engine->loadLocalMD(remote[0].metadataP, clone), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(remote[0].metadataP), NIXL_SUCCESS);
    remote[0].metadataP = clone;
    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine->prepXfer(NIXL_READ, local, remote, init.localAgent, handle),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->unloadMD(clone), NIXL_SUCCESS);
}

TEST_F(wqsEngineTest, ReleasesLeasesAfterPartialPreparationFailure) {
    descriptors({"a", "b"});
    auto *handle = prepare(NIXL_READ);
    --remote[1].len;
    EXPECT_EQ(engine->postXfer(NIXL_READ, local, remote, init.localAgent, handle),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->deregisterMem(dramMetadata), NIXL_SUCCESS);
}

TEST_F(wqsEngineTest, WorkerRecoversAfterClientExceptions) {
    descriptors({"a"});
    client->objects["a"] = std::string(17, 'a');
    client->throwSize = true;
    auto *handle = prepare(NIXL_READ);
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_ERR_BACKEND);
    client->throwSize = false;
    EXPECT_EQ(transfer(NIXL_READ, handle), NIXL_SUCCESS);
}

TEST_F(wqsEngineTest, BoundsInflightWorkAndRejectsActiveReposts) {
    auto limits = config();
    limits.maxInflight = 1;
    initialize(limits);
    descriptors({"a"});
    client->objects["a"] = std::string(17, 'a');
    client->blockReads = true;
    auto entered = client->entered.get_future();
    auto *first = prepare(NIXL_READ);
    auto *second = prepare(NIXL_READ);
    ASSERT_EQ(engine->postXfer(NIXL_READ, local, remote, init.localAgent, first), NIXL_IN_PROG);
    ASSERT_EQ(entered.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(engine->checkXfer(first), NIXL_IN_PROG);
    EXPECT_EQ(engine->postXfer(NIXL_READ, local, remote, init.localAgent, first),
              NIXL_ERR_REPOST_ACTIVE);
    EXPECT_EQ(engine->postXfer(NIXL_READ, local, remote, init.localAgent, second),
              NIXL_ERR_NOT_ALLOWED);
    EXPECT_EQ(engine->checkXfer(second), NIXL_ERR_NOT_POSTED);
    client->unblock();
    EXPECT_EQ(wait(first), NIXL_SUCCESS);
    EXPECT_EQ(engine->disconnect(init.localAgent), NIXL_SUCCESS);
    EXPECT_EQ(transfer(NIXL_READ, second), NIXL_SUCCESS);
}

TEST_F(wqsEngineTest, ReleaseWaitsUntilReadStopsAccessingCallerBuffer) {
    descriptors({"a"});
    client->objects["a"] = std::string(17, 'a');
    client->blockReads = true;
    auto entered = client->entered.get_future();
    auto *handle = prepare(NIXL_READ);
    ASSERT_EQ(engine->postXfer(NIXL_READ, local, remote, init.localAgent, handle), NIXL_IN_PROG);
    ASSERT_EQ(entered.wait_for(5s), std::future_status::ready);
    auto released =
        std::async(std::launch::async, [this, handle] { return engine->releaseReqH(handle); });
    EXPECT_EQ(released.wait_for(20ms), std::future_status::timeout);
    client->unblock();
    EXPECT_EQ(released.get(), NIXL_SUCCESS);
    handles[0] = nullptr;
    EXPECT_EQ(storage, std::vector<char>(17, 'a'));
}

TEST_F(wqsEngineTest, DeregistrationWaitsForInflightBufferUse) {
    descriptors({"a"});
    client->objects["a"] = std::string(17, 'a');
    client->blockReads = true;
    auto entered = client->entered.get_future();
    auto *handle = prepare(NIXL_READ);
    ASSERT_EQ(engine->postXfer(NIXL_READ, local, remote, init.localAgent, handle), NIXL_IN_PROG);
    ASSERT_EQ(entered.wait_for(5s), std::future_status::ready);
    auto deregistered =
        std::async(std::launch::async, [this] { return engine->deregisterMem(dramMetadata); });
    EXPECT_EQ(deregistered.wait_for(20ms), std::future_status::timeout);
    client->unblock();
    EXPECT_EQ(deregistered.get(), NIXL_SUCCESS);
    EXPECT_EQ(wait(handle), NIXL_SUCCESS);
}
} // namespace
