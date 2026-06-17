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

#include <cstdlib>
#include <list>
#include <memory>
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

char char_data = 0;
std::size_t data_size = 100;

struct testRequest {
    testRequest(const nixl_xfer_op_t op, const std::string &tag, const std::string &remote_agent)
        : op(op),
          tag(tag),
          remote_agent(remote_agent),
          dlist(std::make_unique<nixl_meta_dlist_t>(DRAM_SEG)),
          buffer(new char[data_size]),
          data(char_data++) {
        std::memset(buffer.get(), data, data_size);
    }

    nixl_xfer_op_t op;
    std::string tag;
    std::string remote_agent;
    nixlBackendReqH *req = nullptr;
    std::unique_ptr<nixl_meta_dlist_t> dlist;
    std::unique_ptr<char[]> buffer;
    char data;
};

struct testEngine {
    explicit testEngine(const std::string &name)
        : name(name),
          engine(makeTestEngine(name)),
          conn(localConnInfo(engine.get())) {}

    ~testEngine() {
        for (const auto reg : registrations) {
            engine->deregisterMem(reg);
        }
    }

    [[nodiscard]] testRequest
    prepXfer(const nixl_xfer_op_t operation,
             const std::string &tag,
             const std::string &remote_agent,
             const std::string &custom = "") {
        testRequest result(operation, tag, remote_agent);
        registrations.emplace_back(registerMemory(engine.get(), result.buffer.get(), data_size));
        nixlMetaDesc desc;
        desc.addr = uintptr_t(result.buffer.get());
        desc.len = data_size;
        result.dlist->addDesc(desc);
        result.dlist->addDesc(desc);
        result.dlist->addDesc(desc);
        result.dlist->addDesc(desc);
        result.dlist->addDesc(desc);
        nixl_opt_b_args_t args;
        args.customParam = custom;
        EXPECT_EQ(engine->prepTagXfer(operation, *result.dlist, tag, remote_agent, result.req),
                  NIXL_SUCCESS);
        return result;
    }

    bool
    postXfer(testRequest &req, const std::string &custom = "") {
        nixl_opt_b_args_t args;
        args.customParam = custom;
        const auto status =
            engine->postTagXfer(req.op, *req.dlist, req.tag, req.remote_agent, req.req, &args);
        if (status == NIXL_SUCCESS) {
            return true;
        }

        if (status == NIXL_IN_PROG) {
            return false;
        }
        throw std::runtime_error("TODO: Better error message");
    }

    [[nodiscard]] nixl_status_t
    checkXfer(testRequest &req) const {
        return engine->checkXfer(req.req);
    }

    void
    release(testRequest &req) {
        EXPECT_NE(req.req, nullptr);
        const auto status = engine->releaseReqH(req.req);
        EXPECT_EQ(status, NIXL_SUCCESS);
    }

    const std::string name;
    const std::unique_ptr<nixlUcxEngine> engine;
    const std::string conn;

    std::vector<nixlBackendMD *> registrations;
};

const std::string the_tag = "theTAG";
const auto delay = std::chrono::milliseconds(234);

[[nodiscard]]
bool
buffer_eq(const testRequest &req1, const testRequest &req2) noexcept {
    return std::memcmp(req1.buffer.get(), req2.buffer.get(), data_size) == 0;
}

void
basicSendBeforeRecv(const std::string &customParam) {
    testEngine t1("agent");
    testEngine t2("guard");

    EXPECT_EQ(t1.engine->loadRemoteConnInfo(t2.name, t2.conn), NIXL_SUCCESS);
    EXPECT_EQ(t2.engine->loadRemoteConnInfo(t1.name, t1.conn), NIXL_SUCCESS);

    testRequest req1 = t1.prepXfer(NIXL_SEND, the_tag, t2.name);
    testRequest req2 = t2.prepXfer(NIXL_RECV, the_tag, t1.name);

    t1.postXfer(req1, customParam);

    std::this_thread::sleep_for(delay);

    EXPECT_FALSE(buffer_eq(req1, req2));

    t2.postXfer(req2);

    std::this_thread::sleep_for(delay);

    EXPECT_EQ(t1.checkXfer(req1), NIXL_SUCCESS);
    EXPECT_EQ(t2.checkXfer(req2), NIXL_SUCCESS);
    EXPECT_TRUE(buffer_eq(req1, req2));

    t1.release(req1);
    t2.release(req2);
}

void
basicRecvBeforeSend(const std::string &customParam) {
    testEngine t1("agent");
    testEngine t2("guard");

    EXPECT_EQ(t1.engine->loadRemoteConnInfo(t2.name, t2.conn), NIXL_SUCCESS);
    EXPECT_EQ(t2.engine->loadRemoteConnInfo(t1.name, t1.conn), NIXL_SUCCESS);

    testRequest req1 = t1.prepXfer(NIXL_SEND, the_tag, t2.name);
    testRequest req2 = t2.prepXfer(NIXL_RECV, the_tag, t1.name);

    EXPECT_FALSE(t2.postXfer(req2));

    std::this_thread::sleep_for(delay);

    EXPECT_FALSE(buffer_eq(req1, req2));

    t1.postXfer(req1, customParam);

    std::this_thread::sleep_for(delay);

    EXPECT_EQ(t1.checkXfer(req1), NIXL_SUCCESS);
    EXPECT_EQ(t2.checkXfer(req2), NIXL_SUCCESS);
    EXPECT_TRUE(buffer_eq(req1, req2));

    t1.release(req1);
    t2.release(req2);
}

void
queuedSendBeforeRecv(const std::string &customParam, const std::size_t iterations = 42) {
    testEngine t1("agent");
    testEngine t2("guard");

    EXPECT_EQ(t1.engine->loadRemoteConnInfo(t2.name, t2.conn), NIXL_SUCCESS);
    EXPECT_EQ(t2.engine->loadRemoteConnInfo(t1.name, t1.conn), NIXL_SUCCESS);

    std::list<std::pair<testRequest, testRequest>> requests;

    for (std::size_t i = 0; i < iterations; ++i) {
        requests.emplace_back(t1.prepXfer(NIXL_SEND, the_tag, t2.name),
                              t2.prepXfer(NIXL_RECV, the_tag, t1.name));

        t1.postXfer(requests.back().first, customParam);
    }

    std::this_thread::sleep_for(delay);

    for (auto &pair : requests) {
        EXPECT_FALSE(buffer_eq(pair.first, pair.second));

        t2.postXfer(pair.second);
    }

    std::this_thread::sleep_for(delay);

    for (auto &pair : requests) {
        EXPECT_EQ(t1.checkXfer(pair.first), NIXL_SUCCESS);
        EXPECT_EQ(t2.checkXfer(pair.second), NIXL_SUCCESS);
        EXPECT_TRUE(buffer_eq(pair.first, pair.second));

        t1.release(pair.first);
        t2.release(pair.second);
    }
}

void
queuedRecvBeforeSend(const std::string &customParam, const std::size_t iterations = 42) {
    testEngine t1("agent");
    testEngine t2("guard");

    EXPECT_EQ(t1.engine->loadRemoteConnInfo(t2.name, t2.conn), NIXL_SUCCESS);
    EXPECT_EQ(t2.engine->loadRemoteConnInfo(t1.name, t1.conn), NIXL_SUCCESS);

    std::list<std::pair<testRequest, testRequest>> requests;

    for (std::size_t i = 0; i < iterations; ++i) {
        requests.emplace_back(t1.prepXfer(NIXL_SEND, the_tag, t2.name),
                              t2.prepXfer(NIXL_RECV, the_tag, t1.name));

        EXPECT_FALSE(t2.postXfer(requests.back().second));
    }

    std::this_thread::sleep_for(delay);

    for (auto &pair : requests) {
        EXPECT_FALSE(buffer_eq(pair.first, pair.second));

        t1.postXfer(pair.first, customParam);
    }

    std::this_thread::sleep_for(delay);

    for (auto &pair : requests) {
        EXPECT_EQ(t1.checkXfer(pair.first), NIXL_SUCCESS);
        EXPECT_EQ(t2.checkXfer(pair.second), NIXL_SUCCESS);
        EXPECT_TRUE(buffer_eq(pair.first, pair.second));

        t1.release(pair.first);
        t2.release(pair.second);
    }
}

} // namespace

// TEST(UcxSendRecv, EagerRecvSend100) {
//     data_size = 100;
//     basicRecvBeforeSend("eager");
// }

TEST(UcxSendRecv, RndvRecvSend100) {
    data_size = 100;
    basicRecvBeforeSend("rndv");
}

TEST(UcxSendRecv, RndvRecvSend10000) {
    data_size = 10000;
    basicRecvBeforeSend("rndv");
}

TEST(UcxSendRecv, RndvRecvSend1000000) {
    data_size = 1000000;
    basicRecvBeforeSend("rndv");
}

// TEST(UcxSendRecv, AutoRecvSend100) {
//     data_size = 100;
//     basicRecvBeforeSend("");
// }

// TEST(UcxSendRecv, AutoRecvSend10000) {
//     data_size = 10000;
//     basicRecvBeforeSend("");
// }

// TEST(UcxSendRecv, AutoRecvSend1000000) {
//     data_size = 1000000;
//     basicRecvBeforeSend("");
// }

// TEST(UcxSendRecv, EagerSendRecv100) {
//     data_size = 100;
//     basicSendBeforeRecv("eager");
// }

TEST(UcxSendRecv, RndvSendRecv100) {
    data_size = 100;
    basicSendBeforeRecv("rndv");
}

TEST(UcxSendRecv, RndvSendRecv10000) {
    data_size = 10000;
    basicSendBeforeRecv("rndv");
}

TEST(UcxSendRecv, RndvSendRecv1000000) {
    data_size = 1000000;
    basicSendBeforeRecv("rndv");
}

// TEST(UcxSendRecv, AutoSendRecv100) {
//     data_size = 100;
//     basicSendBeforeRecv("");
// }

// TEST(UcxSendRecv, AutoSendRecv10000) {
//     data_size = 10000;
//     basicSendBeforeRecv("");
// }

// TEST(UcxSendRecv, AutoSendRecv1000000) {
//     data_size = 1000000;
//     basicSendBeforeRecv("");
// }

// TEST(UcxSendRecv, QueuedEagerRecvSend100) {
//     data_size = 100;
//     queuedRecvBeforeSend("eager");
// }

TEST(UcxSendRecv, QueuedRndvRecvSend100) {
    data_size = 100;
    queuedRecvBeforeSend("rndv");
}

TEST(UcxSendRecv, QueuedRndvRecvSend10000) {
    data_size = 10000;
    queuedRecvBeforeSend("rndv");
}

TEST(UcxSendRecv, QueuedRndvRecvSend1000000) {
    data_size = 1000000;
    queuedRecvBeforeSend("rndv");
}

// TEST(UcxSendRecv, QueuedAutoRecvSend100) {
//     data_size = 100;
//     queuedRecvBeforeSend("");
// }

// TEST(UcxSendRecv, QueuedAutoRecvSend10000) {
//     data_size = 10000;
//     queuedRecvBeforeSend("");
// }

// TEST(UcxSendRecv, QueuedAutoRecvSend1000000) {
//     data_size = 1000000;
//     queuedRecvBeforeSend("");
// }

// TEST(UcxSendRecv, QueuedEagerSendRecv100) {
//     data_size = 100;
//     queuedSendBeforeRecv("eager");
// }

TEST(UcxSendRecv, QueuedRndvSendRecv100) {
    data_size = 100;
    queuedSendBeforeRecv("rndv");
}

TEST(UcxSendRecv, QueuedRndvSendRecv10000) {
    data_size = 10000;
    queuedSendBeforeRecv("rndv");
}

TEST(UcxSendRecv, QueuedRndvSendRecv1000000) {
    data_size = 1000000;
    queuedSendBeforeRecv("rndv");
}

// TEST(UcxSendRecv, QueuedAutoSendRecv100) {
//     data_size = 100;
//     queuedSendBeforeRecv("");
// }

// TEST(UcxSendRecv, QueuedAutoSendRecv10000) {
//     data_size = 10000;
//     queuedSendBeforeRecv("");
// }

// TEST(UcxSendRecv, QueuedAutoSendRecv1000000) {
//     data_size = 1000000;
//     queuedSendBeforeRecv("");
// }
