/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Compile with tests/mocks first in the include path. This tests ownership only.
#include "worker/nixl/nixl_memos_context.h"
#include <stdexcept>
#include <set>
#include <thread>
#include <limits>

void
check(bool value) {
    if (!value) {
        throw std::runtime_error("MEMOS ownership assertion failed");
    }
}

int
main() {
    using namespace nixlbench;
    for (size_t count : {1, 8}) {
        std::vector<std::unique_ptr<MemosWorkerContext>> contexts;
        check(createMemosContexts(count, "test", {}, contexts) == 0);
        check(nixlAgent::live == static_cast<int>(count));
        check(nixlAgent::backends == static_cast<int>(count));
        std::set<std::string> names;
        std::vector<std::thread> workers;
        for (auto &context : contexts) {
            check(names.insert(context->name()).second);
            auto *agent = context->agent();
            auto buffer = std::shared_ptr<void>(new int(42), [owner = agent->id](void *memory) {
                // Every registration must disappear before its allocation.
                check(!nixlAgent::alive.contains(owner));
                delete static_cast<int *>(memory);
            });
            check(context->registerMemory({1}, buffer) == 0);
            check(context->registerMemory({2}) == 0);
            workers.emplace_back([agent] {
                auto *request = new nixlXferReqH{agent->id};
                auto *list = new nixlDlistH{agent->id};
                nixlDlistH *remote = nullptr;
                check(releaseMemosHandles(*agent, request, list, remote));
                check(!request && !list && !remote);
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }
        contexts.clear();
        check(!nixlAgent::live && !nixlAgent::registered && !nixlAgent::backends);
    }
    for (int failure : {0, 3, 7}) {
        nixlAgent::fail_backend = nixlAgent::next + failure;
        std::vector<std::unique_ptr<MemosWorkerContext>> contexts;
        check(createMemosContexts(8, "failure", {}, contexts) != 0);
        check(contexts.empty() && !nixlAgent::live && !nixlAgent::backends);
    }
    nixlAgent::fail_backend = -1;
    {
        MemosWorkerContext context("registration", 0);
        check(context.initialize({}) == 0);
        check(context.registerMemory({1}) == 0);
        nixlAgent::fail_registration = nixlAgent::register_calls;
        check(context.registerMemory({2}) != 0);
    }
    check(!nixlAgent::live && !nixlAgent::registered);
    check(nixlAgent::releases == 9 && nixlAgent::lists == 9);
    {
        MemosWorkerContext context("release-errors", 0);
        check(context.initialize({}) == NIXL_SUCCESS);
        auto &agent = *context.agent();
        auto *request = new nixlXferReqH{agent.id};
        auto *local = new nixlDlistH{agent.id};
        auto *remote = new nixlDlistH{agent.id};
        nixlAgent::fail_release = true;
        check(!releaseMemosHandles(agent, request, local, remote));
        check(request && local && remote); // All owners remain intact.
        nixlAgent::fail_release = false;
        nixlAgent::fail_list = true;
        check(!releaseMemosHandles(agent, request, local, remote));
        check(!request && local && remote); // Request drained, lists retained.
        nixlAgent::fail_list = false;
        check(releaseMemosHandles(agent, request, local, remote));
        check(!request && !local && !remote);
    }
    std::cout << "Ownership, partial failure and drain checks PASS (mock API)\n";
}
