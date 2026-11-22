/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Compile with tests/mocks first in the include path. This tests ownership only.
#include "worker/nixl/nixl_memos_context.h"
#include <cassert>
#include <set>
#include <thread>
#include <limits>

int
main() {
    using namespace nixlbench;
    for (size_t count : {1, 8}) {
        std::vector<std::unique_ptr<MemosWorkerContext>> contexts;
        assert(createMemosContexts(count, "test", {}, contexts) == 0);
        assert(nixlAgent::live == static_cast<int>(count));
        assert(nixlAgent::backends == static_cast<int>(count));
        std::set<std::string> names;
        std::vector<std::thread> workers;
        for (auto &context : contexts) {
            assert(names.insert(context->name()).second);
            auto *agent = context->agent();
            auto buffer = std::shared_ptr<void>(new int(42), [owner = agent->id](void *memory) {
                // Every registration must disappear before its allocation.
                assert(!nixlAgent::alive.contains(owner));
                delete static_cast<int *>(memory);
            });
            assert(context->registerMemory({1}, buffer) == 0);
            assert(context->registerMemory({2}) == 0);
            workers.emplace_back([agent] {
                auto *request = new nixlXferReqH{agent->id};
                auto *list = new nixlDlistH{agent->id};
                releaseRequest(*agent, request);
                releasePreparedList(*agent, list);
                assert(!request && !list);
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }
        contexts.clear();
        assert(!nixlAgent::live && !nixlAgent::registered && !nixlAgent::backends);
    }
    for (int failure : {0, 3, 7}) {
        nixlAgent::fail_backend = nixlAgent::next + failure;
        std::vector<std::unique_ptr<MemosWorkerContext>> contexts;
        assert(createMemosContexts(8, "failure", {}, contexts) != 0);
        assert(contexts.empty() && !nixlAgent::live && !nixlAgent::backends);
    }
    nixlAgent::fail_backend = -1;
    {
        MemosWorkerContext context("registration", 0);
        assert(context.initialize({}) == 0);
        assert(context.registerMemory({1}) == 0);
        nixlAgent::fail_registration = nixlAgent::register_calls;
        assert(context.registerMemory({2}) != 0);
    }
    assert(!nixlAgent::live && !nixlAgent::registered);
    assert(nixlAgent::releases == 9 && nixlAgent::lists == 9);
    std::cout << "Ownership, partial failure and drain checks PASS (mock API)\n";
}
