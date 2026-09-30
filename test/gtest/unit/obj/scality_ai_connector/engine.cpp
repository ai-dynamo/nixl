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

// ScalityObjEngineImpl with a mock REST client: object queries, which register
// no memory.

#include <gtest/gtest.h>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nixl_types.h"
#include "rest_accel/scality_ai_connector/engine_impl.h"

namespace gtest::obj {
namespace {

    // One request issued by the engine, with the callback that completes it.
    struct restCall {
        enum class kind { PUT, GET, BODY };
        kind method;
        std::string key;
        uintptr_t addr = 0;
        size_t len = 0;
        size_t offset = 0;
        std::string rdmaDesc;
        bool pastEndOk = false;
        std::function<void(bool)> done;
    };

    // Records every request; the test completes them when and how it wants.
    // HEAD answers come from `exists`; a key missing from it is an error.
    class mockRestClient : public iRestClient {
    public:
        std::vector<restCall> calls;
        std::map<std::string, std::optional<bool>> exists;

        void
        putObjectRdmaAsync(std::string_view key,
                           uintptr_t data_ptr,
                           size_t data_len,
                           size_t offset,
                           std::string_view rdma_desc,
                           put_object_callback_t callback) override {
            calls.push_back({restCall::kind::PUT,
                             std::string(key),
                             data_ptr,
                             data_len,
                             offset,
                             std::string(rdma_desc),
                             false,
                             std::move(callback)});
        }

        void
        getObjectRdmaAsync(std::string_view key,
                           uintptr_t data_ptr,
                           size_t data_len,
                           size_t offset,
                           std::string_view rdma_desc,
                           bool past_end_ok,
                           get_object_callback_t callback) override {
            calls.push_back({restCall::kind::GET,
                             std::string(key),
                             data_ptr,
                             data_len,
                             offset,
                             std::string(rdma_desc),
                             past_end_ok,
                             std::move(callback)});
        }

        void
        getObjectBodyAsync(std::string_view key,
                           void *dst,
                           size_t data_len,
                           size_t offset,
                           get_object_callback_t callback) override {
            calls.push_back({restCall::kind::BODY,
                             std::string(key),
                             reinterpret_cast<uintptr_t>(dst),
                             data_len,
                             offset,
                             "",
                             false,
                             std::move(callback)});
        }

        void
        checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) override {
            auto it = exists.find(std::string(key));
            callback(it == exists.end() ? std::nullopt : it->second);
        }

        void
        complete(size_t i, bool ok) {
            calls.at(i).done(ok);
        }

        void
        completeAll(bool ok) {
            for (auto &call : calls) {
                call.done(ok);
            }
        }
    };

    const std::string test_agent = "agent";

    class scalityEngineTest : public testing::Test {
    protected:
        void
        makeEngine() {
            init_.localAgent = test_agent;
            init_.customParams = &params_;
            rest_ = std::make_shared<mockRestClient>();
            engine_ = std::make_unique<ScalityObjEngineImpl>(&init_, rest_);
        }

        nixl_b_params_t params_;
        nixlBackendInitParams init_;
        std::shared_ptr<mockRestClient> rest_;
        std::unique_ptr<ScalityObjEngineImpl> engine_;
    };

} // namespace

// ---------------------------------------------------------------------------
// queryMem: object existence through HEAD
// ---------------------------------------------------------------------------

TEST_F(scalityEngineTest, QueryMemReportsWhichObjectsExist) {
    makeEngine();
    rest_->exists = {{"present", true}, {"absent", false}};

    nixl_reg_dlist_t descs(OBJ_SEG);
    for (const char *key : {"present", "absent"}) {
        nixlBlobDesc desc = {};
        desc.metaInfo = key;
        descs.addDesc(desc);
    }
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(descs, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 2u);
    EXPECT_TRUE(resp[0].has_value()) << "an existing object must be reported";
    EXPECT_FALSE(resp[1].has_value()) << "a missing object must not be reported";
}

TEST_F(scalityEngineTest, QueryMemFailsWhenACheckFails) {
    makeEngine();
    rest_->exists = {{"present", true}}; // "broken" is missing: its HEAD fails

    nixl_reg_dlist_t descs(OBJ_SEG);
    for (const char *key : {"present", "broken"}) {
        nixlBlobDesc desc = {};
        desc.metaInfo = key;
        descs.addDesc(desc);
    }
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(descs, resp), NIXL_ERR_BACKEND);
    ASSERT_EQ(resp.size(), 2u);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_FALSE(resp[1].has_value());
}

} // namespace gtest::obj
