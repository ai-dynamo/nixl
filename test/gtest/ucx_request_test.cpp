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
#include <functional>

#include "ucx_backend.h"
#include "ucx_backend_req.h"

namespace {

class ControlledUcxRequest : public nixlUcxBackendReqH {
public:
    explicit ControlledUcxRequest(bool &released)
        : nixlUcxBackendReqH(nullptr),
          released_(released) {}

    nixl_status_t result = NIXL_IN_PROG;

    nixl_status_t
    status() override {
        return result;
    }

    void
    release() override {
        released_ = true;
    }

private:
    bool &released_;
};

TEST(UcxRequestLifetime, RefusesReleaseUntilTransferCompletes) {
    nixl_b_params_t params;
    nixlBackendInitParams init;
    init.localAgent = "RequestLifetime";
    init.type = "UCX";
    init.enableProgTh = false;
    init.syncMode = NIXL_THREAD_SYNC_STRICT;
    init.customParams = &params;
    auto engine = nixlUcxEngine::create(init);
    ASSERT_NE(engine, nullptr);
    ASSERT_FALSE(engine->getInitErr());

    bool released = false;
    auto *request = new ControlledUcxRequest(released);
    ASSERT_EQ(engine->releaseReqH(request), NIXL_ERR_REPOST_ACTIVE);
    EXPECT_FALSE(released);
    request->result = NIXL_SUCCESS;
    EXPECT_EQ(engine->releaseReqH(request), NIXL_SUCCESS);
    EXPECT_TRUE(released);
}

class PostingUcxEngine : public nixlUcxEngine {
public:
    explicit PostingUcxEngine(const nixlBackendInitParams &init) : nixlUcxEngine(init) {}

    nixlUcxWorker *
    worker() const {
        return getSharedWorker(0).get();
    }

    std::function<nixl_status_t(nixlUcxBackendReqH *)> post;

protected:
    nixl_status_t
    sendXferRange(const nixl_xfer_op_t &operation,
                  const nixl_meta_dlist_t &local,
                  const nixl_meta_dlist_t &remote,
                  const std::string &remote_agent,
                  nixlBackendReqH *handle,
                  size_t start,
                  size_t end) const override {
        if (post) {
            return post(static_cast<nixlUcxBackendReqH *>(handle));
        }
        return nixlUcxEngine::sendXferRange(
            operation, local, remote, remote_agent, handle, start, end);
    }
};

class UcxRequestDrain : public testing::TestWithParam<size_t> {
protected:
    struct Receive {
        nixlUcxReq request = nullptr;
        ucs_status_t result = UCS_INPROGRESS;
        char buffer = 0;
    };

    std::unique_ptr<PostingUcxEngine> engine_;
    std::unique_ptr<nixlUcxBackendReqH> request_;
    ucx_connection_ptr_t connection_;
    nixlBackendMD *local_md_ = nullptr;
    nixlBackendMD *remote_md_ = nullptr;
    std::array<char, 64> memory_{};
    ucp_context_h context_ = nullptr;
    ucp_worker_h worker_ = nullptr;
    std::array<Receive, 3> receives_{};

    void
    SetUp() override {
        nixl_b_params_t params;
        nixlBackendInitParams init;
        init.localAgent = "RequestDrain";
        init.type = "UCX";
        init.enableProgTh = false;
        init.syncMode = NIXL_THREAD_SYNC_STRICT;
        init.customParams = &params;
        engine_ = std::make_unique<PostingUcxEngine>(init);
        ASSERT_FALSE(engine_->getInitErr());
        ASSERT_EQ(engine_->connect(init.localAgent), NIXL_SUCCESS);
        const nixlBlobDesc memory(
            reinterpret_cast<uintptr_t>(memory_.data()), memory_.size(), 0, "");
        ASSERT_EQ(engine_->registerMem(memory, DRAM_SEG, local_md_), NIXL_SUCCESS);
        ASSERT_EQ(engine_->loadLocalMD(local_md_, remote_md_), NIXL_SUCCESS);
        connection_ = static_cast<nixlUcxPublicMetadata *>(remote_md_)->conn;
        request_ = std::make_unique<nixlUcxBackendReqH>(engine_->worker());
        request_->init(connection_, *connection_->getEp(engine_->worker()->getId()));

        // Unmatched receives provide real UCX requests with deterministic completion.
        // Their separate worker is progressed explicitly; status/free are worker-independent.
        ucp_params_t context_params{};
        context_params.field_mask = UCP_PARAM_FIELD_FEATURES;
        context_params.features = UCP_FEATURE_TAG;
        ucp_config_t *config = nullptr;
        ASSERT_EQ(ucp_config_read(nullptr, nullptr, &config), UCS_OK);
        const auto status = ucp_init(&context_params, config, &context_);
        ucp_config_release(config);
        ASSERT_EQ(status, UCS_OK);
        ucp_worker_params_t worker_params{};
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
        ASSERT_EQ(ucp_worker_create(context_, &worker_params, &worker_), UCS_OK);
    }

    void
    addPending(size_t index) {
        auto &receive = receives_[index];
        ucp_request_param_t params{};
        params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        params.user_data = &receive;
        params.cb.recv = [](void *, ucs_status_t status, const ucp_tag_recv_info_t *, void *data) {
            static_cast<Receive *>(data)->result = status;
        };
        receive.request =
            ucp_tag_recv_nbx(worker_, &receive.buffer, 1, index + 1, ~ucp_tag_t{0}, &params);
        ASSERT_TRUE(UCS_PTR_IS_PTR(receive.request));
        ASSERT_EQ(request_->append(NIXL_IN_PROG, receive.request), NIXL_SUCCESS);
    }

    void
    cancel(size_t index) {
        auto &receive = receives_[index];
        ASSERT_NE(receive.request, nullptr);
        ASSERT_EQ(receive.result, UCS_INPROGRESS);
        ucp_request_cancel(worker_, receive.request);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (receive.result == UCS_INPROGRESS && std::chrono::steady_clock::now() < deadline) {
            ucp_worker_progress(worker_);
        }
        ASSERT_EQ(receive.result, UCS_ERR_CANCELED);
    }

    nixl_status_t
    post() {
        nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
        nixlBackendReqH *handle = request_.get();
        return engine_->postXfer(NIXL_READ, local, remote, "RequestDrain", handle);
    }

    void
    TearDown() override {
        for (size_t i = 0; i < receives_.size(); ++i) {
            if (UCS_PTR_IS_PTR(receives_[i].request) && receives_[i].result == UCS_INPROGRESS) {
                cancel(i);
            }
        }
        if (request_) {
            EXPECT_NE(request_->status(), NIXL_IN_PROG);
            request_->release();
            request_.reset();
        }
        connection_.reset();
        if (remote_md_) {
            EXPECT_EQ(engine_->unloadMD(remote_md_), NIXL_SUCCESS);
        }
        if (local_md_) {
            EXPECT_EQ(engine_->deregisterMem(local_md_), NIXL_SUCCESS);
        }
        engine_.reset();
        if (worker_) {
            ucp_worker_destroy(worker_);
        }
        if (context_) {
            ucp_cleanup(context_);
        }
    }
};

TEST_F(UcxRequestDrain, EmptyAndImmediateSuccess) {
    EXPECT_EQ(request_->status(), NIXL_SUCCESS);
    EXPECT_EQ(request_->append(NIXL_SUCCESS, nullptr), NIXL_SUCCESS);
    EXPECT_EQ(request_->status(), NIXL_SUCCESS);
    engine_->post = [](nixlUcxBackendReqH *) { return NIXL_SUCCESS; };
    EXPECT_EQ(post(), NIXL_SUCCESS);
}

TEST_F(UcxRequestDrain, LoopbackReadWriteAndRepost) {
    for (const auto operation : {NIXL_READ, NIXL_WRITE}) {
        for (int round = 0; round < 2; ++round) {
            std::fill_n(memory_.begin(), 32, 'x');
            std::fill(memory_.begin() + 32, memory_.end(), 0);
            nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
            for (size_t i = 0; i < 4; ++i) {
                nixlMetaDesc local_desc, remote_desc;
                local_desc.addr = reinterpret_cast<uintptr_t>(memory_.data() + i * 8 +
                                                              (operation == NIXL_READ ? 32 : 0));
                local_desc.len = 8;
                local_desc.devId = 0;
                local_desc.metadataP = local_md_;
                remote_desc.addr = reinterpret_cast<uintptr_t>(memory_.data() + i * 8 +
                                                               (operation == NIXL_WRITE ? 32 : 0));
                remote_desc.len = 8;
                remote_desc.devId = 0;
                remote_desc.metadataP = remote_md_;
                local.addDesc(local_desc);
                remote.addDesc(remote_desc);
            }
            nixlBackendReqH *handle = request_.get();
            auto status = engine_->postXfer(operation, local, remote, "RequestDrain", handle);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
                status = engine_->checkXfer(handle);
            }
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_TRUE(std::all_of(
                memory_.begin() + 32, memory_.end(), [](char value) { return value == 'x'; }));
        }
    }
}

TEST_F(UcxRequestDrain, RepostClearsPreviousError) {
    engine_->post = [](nixlUcxBackendReqH *) { return NIXL_ERR_BACKEND; };
    EXPECT_EQ(post(), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine_->checkXfer(request_.get()), NIXL_ERR_BACKEND);
    engine_->post = [](nixlUcxBackendReqH *) { return NIXL_SUCCESS; };
    EXPECT_EQ(post(), NIXL_SUCCESS);
    EXPECT_EQ(engine_->checkXfer(request_.get()), NIXL_SUCCESS);
}

TEST_F(UcxRequestDrain, PostingFailureWaitsForPreviouslyPostedWork) {
    engine_->post = [&](nixlUcxBackendReqH *request) {
        addPending(0);
        addPending(1);
        return request->append(NIXL_ERR_BACKEND, nullptr);
    };
    ASSERT_EQ(post(), NIXL_IN_PROG);
    EXPECT_EQ(engine_->releaseReqH(request_.get()), NIXL_ERR_REPOST_ACTIVE);
    ASSERT_NO_FATAL_FAILURE(cancel(1));
    EXPECT_EQ(engine_->checkXfer(request_.get()), NIXL_IN_PROG);
    ASSERT_NO_FATAL_FAILURE(cancel(0));
    EXPECT_EQ(engine_->checkXfer(request_.get()), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine_->checkXfer(request_.get()), NIXL_ERR_BACKEND);
}

TEST_P(UcxRequestDrain, AsyncFailureWaitsForEveryOtherRequest) {
    for (size_t i = 0; i < receives_.size(); ++i) {
        ASSERT_NO_FATAL_FAILURE(addPending(i));
    }
    const size_t failed = GetParam();
    ASSERT_NO_FATAL_FAILURE(cancel(failed));
    EXPECT_EQ(engine_->checkXfer(request_.get()), NIXL_IN_PROG);
    EXPECT_EQ(engine_->releaseReqH(request_.get()), NIXL_ERR_REPOST_ACTIVE);
    ASSERT_NO_FATAL_FAILURE(cancel((failed + 1) % receives_.size()));
    EXPECT_EQ(engine_->checkXfer(request_.get()), NIXL_IN_PROG);
    ASSERT_NO_FATAL_FAILURE(cancel((failed + 2) % receives_.size()));
    EXPECT_EQ(engine_->checkXfer(request_.get()), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(engine_->checkXfer(request_.get()), NIXL_ERR_REMOTE_DISCONNECT);
}

INSTANTIATE_TEST_SUITE_P(ErrorPosition, UcxRequestDrain, testing::Values(0, 1, 2));

} // namespace
