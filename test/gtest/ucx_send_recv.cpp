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

#include "nixl.h"
#include "ucx_backend.h"

#include "gtest/gtest.h"

#include <string>

namespace {

[[nodiscard]] std::unique_ptr<nixlUcxEngine>
makeTestEngine(const std::string &name) {
    nixl_b_params_t custom_params;
    nixlBackendInitParams init_params;
    init_params.localAgent = name;
    init_params.enableProgTh = true;
    init_params.customParams = &custom_params;
    init_params.type = "UCX";
    return nixlUcxEngine::create(init_params);
}

[[nodiscard]] std::string
localConnInfo(nixlUcxEngine *engine) {
    std::string conn;
    EXPECT_TRUE(bool(engine));
    EXPECT_EQ(engine->getConnInfo(conn), NIXL_SUCCESS);
    return conn;
}

[[nodiscard]] nixlBackendMD *
registerMemory(nixlUcxEngine *engine, void *data, const std::size_t size) {
    nixlBackendMD *result = nullptr;
    EXPECT_TRUE(bool(engine));
    nixlBlobDesc desc;
    desc.addr = uintptr_t(data);
    desc.len = size;
    EXPECT_EQ(engine->registerMem(desc, DRAM_SEG, result), NIXL_SUCCESS);
    EXPECT_TRUE(bool(result));
    return result;
}

struct testRequest {
    testRequest(const nixl_xfer_op_t op, const std::string &tag, const std::string &remote_agent)
        : op(op),
          tag(tag),
          remote_agent(remote_agent),
          dlist(DRAM_SEG) {}

    nixl_xfer_op_t op;
    std::string tag;
    std::string remote_agent;
    nixlBackendReqH *req = nullptr;
    nixl_meta_dlist_t dlist;
};

struct testEngine {
    testEngine(const std::string &name, const char data)
        : name(name),
          engine(makeTestEngine(name)),
          conn(localConnInfo(engine.get())),
          buffer(64, data),
          reg(registerMemory(engine.get(), buffer.data(), buffer.size()))
    {}

    ~testEngine() {
        engine->deregisterMem(reg);
    }

    [[nodiscard]] testRequest
    prepXfer(const nixl_xfer_op_t operation,
             const std::string &tag,
             const std::string &remote_agent) {
        testRequest result(operation, tag, remote_agent);
        nixlMetaDesc desc;
        desc.addr = uintptr_t(buffer.data());
        desc.len = buffer.size();
        result.dlist.addDesc(desc);
        EXPECT_EQ(engine->prepTagXfer(operation,
                                      result.dlist,
                                      tag,
                                      remote_agent,
                                      result.req), NIXL_SUCCESS);
        return result;
    }

    [[nodiscard]] bool
    postXfer(testRequest &req) {
        const auto status = engine->postTagXfer(req.op,
                                                req.dlist,
                                                req.tag,
                                                req.remote_agent,
                                                req.req);
        if (status == NIXL_SUCCESS) {
            return true;
        }

        if (status == NIXL_IN_PROG) {
            return false;
        }
        throw std::runtime_error("TODO: Better error message");
    }

    const std::string name;
    const std::unique_ptr<nixlUcxEngine> engine;
    const std::string conn;

    std::vector<char> buffer;
    nixlBackendMD *reg = nullptr;
};

} // namespace

TEST(UcxSendRecv, Basic1) {
    testEngine t1("agent", 'a');
    testEngine t2("guard", 'b');

    EXPECT_EQ(t1.engine->loadRemoteConnInfo(t2.name, t2.conn), NIXL_SUCCESS);
    EXPECT_EQ(t2.engine->loadRemoteConnInfo(t1.name, t1.conn), NIXL_SUCCESS);

    const std::string tag = "tag";

    testRequest req1 = t1.prepXfer(NIXL_SEND, tag, t2.name);
    testRequest req2 = t2.prepXfer(NIXL_RECV, tag, t1.name);

    EXPECT_FALSE(t1.postXfer(req1));

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    EXPECT_NE(t1.buffer, t2.buffer);

    EXPECT_TRUE(t2.postXfer(req2));

    EXPECT_EQ(t1.buffer, t2.buffer);

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    EXPECT_EQ(t1.engine->checkXfer(req1.req), NIXL_SUCCESS);
}

TEST(UcxSendRecv, Basic2) {
    testEngine t1("agent", 'a');
    testEngine t2("guard", 'b');

    EXPECT_EQ(t1.engine->loadRemoteConnInfo(t2.name, t2.conn), NIXL_SUCCESS);
    EXPECT_EQ(t2.engine->loadRemoteConnInfo(t1.name, t1.conn), NIXL_SUCCESS);

    const std::string tag = "tag";

    testRequest req1 = t1.prepXfer(NIXL_SEND, tag, t2.name);
    testRequest req2 = t2.prepXfer(NIXL_RECV, tag, t1.name);

    EXPECT_FALSE(t2.postXfer(req2));

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    EXPECT_NE(t1.buffer, t2.buffer);

    EXPECT_TRUE(t1.postXfer(req1));

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // TODO: Do this "officially":
    const auto r = dynamic_cast<nixlUcxBackendRecvH *>(req2.req);
    EXPECT_EQ(r->status, NIXL_SUCCESS);

    EXPECT_EQ(t1.buffer, t2.buffer);
}
