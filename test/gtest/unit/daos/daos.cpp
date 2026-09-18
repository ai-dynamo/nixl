/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "daos_backend.h"

namespace {

class MockDfsObject final : public iDfsObject {
public:
    explicit MockDfsObject(std::string path) : path(std::move(path)) {}

    std::string path;
};

class MockDfsClient final : public iDfsClient {
public:
    int
    open(std::string_view path, bool write, std::shared_ptr<iDfsObject> &object) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string key(path);
        if (!write && files_.count(key) == 0) {
            return ENOENT;
        }
        files_[key];
        object = std::make_shared<MockDfsObject>(key);
        return 0;
    }

    int
    submitRead(const std::shared_ptr<iDfsObject> &object,
               uintptr_t data_ptr,
               size_t data_len,
               uint64_t offset,
               completion_t completion) override {
        if (shouldBackpressure()) {
            return EAGAIN;
        }
        const auto path = std::static_pointer_cast<MockDfsObject>(object)->path;
        schedule([this, path, data_ptr, data_len, offset, completion = std::move(completion)] {
            size_t bytes_read = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = files_.find(path);
                if (it != files_.end() && offset < it->second.size()) {
                    bytes_read =
                        std::min(data_len, it->second.size() - static_cast<size_t>(offset));
                    std::memcpy(
                        reinterpret_cast<void *>(data_ptr), it->second.data() + offset, bytes_read);
                }
            }
            completion(0, bytes_read);
        });
        return 0;
    }

    int
    submitWrite(const std::shared_ptr<iDfsObject> &object,
                uintptr_t data_ptr,
                size_t data_len,
                uint64_t offset,
                completion_t completion) override {
        if (shouldBackpressure()) {
            return EAGAIN;
        }
        const auto path = std::static_pointer_cast<MockDfsObject>(object)->path;
        schedule([this, path, data_ptr, data_len, offset, completion = std::move(completion)] {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto &file = files_[path];
                if (file.size() < offset + data_len) {
                    file.resize(offset + data_len);
                }
                std::memcpy(file.data() + offset, reinterpret_cast<void *>(data_ptr), data_len);
            }
            completion(0, data_len);
        });
        return 0;
    }

    int
    exists(std::string_view path, bool &result) override {
        std::lock_guard<std::mutex> lock(mutex_);
        result = files_.count(std::string(path)) != 0;
        return 0;
    }

    void
    put(std::string path, std::vector<char> value) {
        std::lock_guard<std::mutex> lock(mutex_);
        files_[std::move(path)] = std::move(value);
    }

    void
    deferCompletions(bool value) {
        defer_ = value;
    }

    void
    backpressureOnSubmission(size_t submission) {
        submissionCount_.store(0, std::memory_order_relaxed);
        eagainSubmission_.store(submission, std::memory_order_relaxed);
    }

    size_t
    submissionCount() const {
        return submissionCount_.load(std::memory_order_relaxed);
    }

    void
    completeAll() {
        std::deque<std::function<void()>> pending;
        {
            std::lock_guard<std::mutex> lock(pendingMutex_);
            pending.swap(pending_);
        }
        for (auto &completion : pending) {
            completion();
        }
    }

private:
    bool
    shouldBackpressure() {
        const size_t submission = submissionCount_.fetch_add(1, std::memory_order_relaxed) + 1;
        return submission == eagainSubmission_.load(std::memory_order_relaxed);
    }

    void
    schedule(std::function<void()> operation) {
        if (!defer_) {
            operation();
            return;
        }
        std::lock_guard<std::mutex> lock(pendingMutex_);
        pending_.push_back(std::move(operation));
    }

    std::mutex mutex_;
    std::map<std::string, std::vector<char>> files_;
    bool defer_ = false;
    std::atomic<size_t> submissionCount_{0};
    std::atomic<size_t> eagainSubmission_{0};
    std::mutex pendingMutex_;
    std::deque<std::function<void()>> pending_;
};

nixl_status_t
waitFor(nixlDaosEngine &engine, nixlBackendReqH *handle) {
    for (int attempt = 0; attempt < 1000; ++attempt) {
        const nixl_status_t status = engine.checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            return status;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return NIXL_IN_PROG;
}

class DaosEngineTest : public testing::Test {
protected:
    void
    SetUp() override {
        init_.localAgent = "daos-test-agent";
        init_.type = "DAOS";
        init_.customParams = &params_;
        client_ = std::make_shared<MockDfsClient>();
        engine_ = std::make_unique<nixlDaosEngine>(&init_, client_);
    }

    nixl_b_params_t params_;
    nixlBackendInitParams init_;
    std::shared_ptr<MockDfsClient> client_;
    std::unique_ptr<nixlDaosEngine> engine_;
};

TEST_F(DaosEngineTest, ReportsObjectStorageCapabilities) {
    EXPECT_TRUE(engine_->supportsLocal());
    EXPECT_FALSE(engine_->supportsRemote());
    EXPECT_FALSE(engine_->supportsNotif());
    EXPECT_EQ(engine_->getSupportedMems(), (nixl_mem_list_t{DRAM_SEG, OBJ_SEG}));
}

TEST_F(DaosEngineTest, ReportsEventQueueTuningParameters) {
    const auto params = nixlDaosEngine::getPluginParams();
    EXPECT_EQ(params.at("object_class"), "");
    EXPECT_EQ(params.at("object_class_hint"), "");
    EXPECT_EQ(params.at("oclass_id"), "0");
    EXPECT_EQ(params.at("num_event_queues"), "1");
    EXPECT_EQ(params.at("max_inflight_per_queue"), "1024");
    EXPECT_EQ(params.at("submission_batch_size"), "32");
    EXPECT_EQ(params.at("completion_batch_size"), "128");
    EXPECT_EQ(params.at("progress_poll_timeout_us"), "1000");
    EXPECT_EQ(params.at("progress_cpu_affinity"), "");
}

TEST_F(DaosEngineTest, WritesAndReadsAtOffset) {
    nixlBlobDesc registration(0, 16, 7, "layers/0/cache.bin");
    nixlBackendMD *metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(registration, OBJ_SEG, metadata), NIXL_SUCCESS);
    ASSERT_NE(metadata, nullptr);

    std::vector<char> source{'n', 'i', 'x', 'l'};
    nixl_meta_dlist_t local_write(DRAM_SEG);
    nixl_meta_dlist_t remote_write(OBJ_SEG);
    local_write.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(source.data()), source.size(), 0));
    remote_write.addDesc(nixlMetaDesc(3, source.size(), 7, metadata));

    nixlBackendReqH *write_handle = nullptr;
    ASSERT_EQ(
        engine_->prepXfer(NIXL_WRITE, local_write, remote_write, init_.localAgent, write_handle),
        NIXL_SUCCESS);
    ASSERT_EQ(
        engine_->postXfer(NIXL_WRITE, local_write, remote_write, init_.localAgent, write_handle),
        NIXL_IN_PROG);
    EXPECT_EQ(waitFor(*engine_, write_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->releaseReqH(write_handle), NIXL_SUCCESS);

    std::vector<char> destination(source.size());
    nixl_meta_dlist_t local_read(DRAM_SEG);
    nixl_meta_dlist_t remote_read(OBJ_SEG);
    local_read.addDesc(
        nixlMetaDesc(reinterpret_cast<uintptr_t>(destination.data()), destination.size(), 0));
    remote_read.addDesc(nixlMetaDesc(3, destination.size(), 7, metadata));

    nixlBackendReqH *read_handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_READ, local_read, remote_read, init_.localAgent, read_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_READ, local_read, remote_read, init_.localAgent, read_handle),
              NIXL_IN_PROG);
    EXPECT_EQ(waitFor(*engine_, read_handle), NIXL_SUCCESS);
    EXPECT_EQ(destination, source);
    EXPECT_EQ(engine_->releaseReqH(read_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(metadata), NIXL_SUCCESS);
}

TEST_F(DaosEngineTest, QueryReportsExistingAndMissingPaths) {
    client_->put("present", {'x'});

    nixl_reg_dlist_t descriptors(OBJ_SEG);
    descriptors.addDesc(nixlBlobDesc(0, 1, 1, "present"));
    descriptors.addDesc(nixlBlobDesc(0, 1, 2, "missing"));
    std::vector<nixl_query_resp_t> response;
    EXPECT_EQ(engine_->queryMem(descriptors, response), NIXL_SUCCESS);
    ASSERT_EQ(response.size(), 2);
    EXPECT_TRUE(response[0].has_value());
    EXPECT_FALSE(response[1].has_value());
}

TEST_F(DaosEngineTest, RepostsCompletedRequestHandle) {
    nixlBlobDesc registration(0, 4, 1, "reposted");
    nixlBackendMD *metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(registration, OBJ_SEG, metadata), NIXL_SUCCESS);

    std::vector<char> source{'d', 'a', 'o', 's'};
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(source.data()), source.size(), 0));
    remote.addDesc(nixlMetaDesc(0, source.size(), 1, metadata));

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local, remote, init_.localAgent, handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local, remote, init_.localAgent, handle), NIXL_IN_PROG);
    ASSERT_EQ(waitFor(*engine_, handle), NIXL_SUCCESS);

    const std::array<char, 4> replacement{'n', 'i', 'x', 'l'};
    std::copy(replacement.begin(), replacement.end(), source.begin());
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local, remote, init_.localAgent, handle), NIXL_IN_PROG);
    EXPECT_EQ(waitFor(*engine_, handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    std::vector<char> destination(source.size());
    nixl_meta_dlist_t local_read(DRAM_SEG);
    nixl_meta_dlist_t remote_read(OBJ_SEG);
    local_read.addDesc(
        nixlMetaDesc(reinterpret_cast<uintptr_t>(destination.data()), destination.size(), 0));
    remote_read.addDesc(nixlMetaDesc(0, destination.size(), 1, metadata));

    nixlBackendReqH *read_handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_READ, local_read, remote_read, init_.localAgent, read_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_READ, local_read, remote_read, init_.localAgent, read_handle),
              NIXL_IN_PROG);
    EXPECT_EQ(waitFor(*engine_, read_handle), NIXL_SUCCESS);
    EXPECT_EQ(destination, source);
    EXPECT_EQ(engine_->releaseReqH(read_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(metadata), NIXL_SUCCESS);
}

TEST_F(DaosEngineTest, RetriesSubmissionBackpressure) {
    nixlBlobDesc registration(0, 8, 1, "backpressure");
    nixlBackendMD *metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(registration, OBJ_SEG, metadata), NIXL_SUCCESS);

    std::array<char, 8> source{'d', 'a', 'o', 's', 'n', 'i', 'x', 'l'};
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(source.data()), 4, 0));
    local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(source.data() + 4), 4, 0));
    remote.addDesc(nixlMetaDesc(0, 4, 1, metadata));
    remote.addDesc(nixlMetaDesc(4, 4, 1, metadata));

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local, remote, init_.localAgent, handle), NIXL_SUCCESS);
    client_->backpressureOnSubmission(2);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local, remote, init_.localAgent, handle), NIXL_IN_PROG);
    EXPECT_EQ(waitFor(*engine_, handle), NIXL_SUCCESS);
    EXPECT_EQ(client_->submissionCount(), 3);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(metadata), NIXL_SUCCESS);
}

TEST_F(DaosEngineTest, ReleaseRetriesSubmissionBackpressure) {
    nixlBlobDesc registration(0, 4, 1, "release-backpressure");
    nixlBackendMD *metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(registration, OBJ_SEG, metadata), NIXL_SUCCESS);

    std::array<char, 4> source{'d', 'a', 'o', 's'};
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(source.data()), source.size(), 0));
    remote.addDesc(nixlMetaDesc(0, source.size(), 1, metadata));

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local, remote, init_.localAgent, handle), NIXL_SUCCESS);
    client_->backpressureOnSubmission(1);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local, remote, init_.localAgent, handle), NIXL_IN_PROG);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(client_->submissionCount(), 2);

    EXPECT_EQ(engine_->deregisterMem(metadata), NIXL_SUCCESS);
}

TEST_F(DaosEngineTest, RefusesToReleaseARequestBeforeEventCompletion) {
    client_->deferCompletions(true);
    nixlBlobDesc registration(0, 4, 1, "deferred");
    nixlBackendMD *metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(registration, OBJ_SEG, metadata), NIXL_SUCCESS);

    std::vector<char> source{'d', 'a', 'o', 's'};
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(source.data()), source.size(), 0));
    remote.addDesc(nixlMetaDesc(0, source.size(), 1, metadata));

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local, remote, init_.localAgent, handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local, remote, init_.localAgent, handle), NIXL_IN_PROG);
    EXPECT_EQ(engine_->checkXfer(handle), NIXL_IN_PROG);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_ERR_NOT_ALLOWED);

    client_->completeAll();
    EXPECT_EQ(engine_->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(metadata), NIXL_SUCCESS);
}

} // namespace
