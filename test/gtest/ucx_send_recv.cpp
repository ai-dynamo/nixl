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

struct testStruct {
    testStruct(const std::string &name, const char data)
        : name(name),
          engine(makeTestEngine(name)),
          conn(localConnInfo(engine.get())),
          buffer(64, data),
          reg(registerMemory(engine.get(), buffer.data(), buffer.size()))
    {}

    ~testStruct() {
        engine->deregisterMem(reg);
    }

    const std::string name;
    const std::unique_ptr<nixlUcxEngine> engine;
    const std::string conn;

    std::vector<char> buffer;
    nixlBackendMD *reg = nullptr;
};

} // namespace

TEST(UcxSendRecv, Basic) {
    testStruct t1("agent", 'a');
    testStruct t2("guard", 'b');

    EXPECT_EQ(t1.engine->loadRemoteConnInfo(t2.name, t2.conn), NIXL_SUCCESS);
    EXPECT_EQ(t2.engine->loadRemoteConnInfo(t1.name, t1.conn), NIXL_SUCCESS);

    const std::string tag = "tag";

    nixlBackendReqH *req1 = nullptr;
    nixl_meta_dlist_t dlist1(DRAM_SEG);
    nixlMetaDesc desc1;
    desc1.addr = uintptr_t(t1.buffer.data());
    desc1.len = t1.buffer.size();
    dlist1.addDesc(desc1);

    EXPECT_EQ(t1.engine->prepTagXfer(NIXL_SEND,
                                     dlist1,
                                     tag,
                                     t2.name,
                                     req1), NIXL_SUCCESS);

    nixlBackendReqH *req2 = nullptr;
    nixl_meta_dlist_t dlist2(DRAM_SEG);
    nixlMetaDesc desc2;
    desc2.addr = uintptr_t(t2.buffer.data());
    desc2.len = t2.buffer.size();
    dlist2.addDesc(desc2);

    EXPECT_EQ(t2.engine->prepTagXfer(NIXL_RECV,
                                     dlist2,
                                     tag,
                                     t1.name,
                                     req2), NIXL_SUCCESS);


    EXPECT_EQ(t1.engine->postTagXfer(NIXL_SEND,
                                     dlist1,
                                     tag,
                                     t2.name,
                                     req1), NIXL_IN_PROG);

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    EXPECT_NE(t1.buffer, t2.buffer);

    EXPECT_EQ(t2.engine->postTagXfer(NIXL_RECV,
                                     dlist2,
                                     tag,
                                     t1.name,
                                     req2), NIXL_SUCCESS);

    EXPECT_EQ(t1.buffer, t2.buffer);

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    EXPECT_EQ(t1.engine->checkXfer(req1), NIXL_SUCCESS);
}
