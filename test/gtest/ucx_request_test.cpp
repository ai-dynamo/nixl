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

} // namespace
