/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <atomic>
#include <iostream>
#include <string>
#include <thread>

#include "ucx_backend.h"
#include "test_utils.h"

// TODO: Find out what this comment means:
// Temporarily while fixing CI/CD pipeline
// #define USE_PTHREAD false

std::atomic<bool> ready[2];
std::atomic<bool> done[2];
std::atomic<bool> disconnect[2];

std::string conn_info[2];

void
test_thread(const unsigned id, const bool progress_thread) {
    const std::string my_name = "Agent" + std::to_string(id);
    const std::string other = "Agent" + std::to_string(1 - id);

    nixl_b_params_t custom_params;
    nixlBackendInitParams init_params;
    init_params.localAgent = my_name;
    init_params.enableProgTh = progress_thread;
    init_params.customParams = &custom_params;
    init_params.type = "UCX";

    std::cout << my_name << " Started\n";

    const auto ucx = nixlUcxEngine::create(init_params);

    if (!progress_thread) {
        ucx->progress();
    }

    ucx->getConnInfo(conn_info[id]);

    ready[id].store(true);
    while (!ready[!id].load())
        ;

    const auto ret = ucx->loadRemoteConnInfo(other, conn_info[!id]);
    nixl_exit_on_failure((ret == NIXL_SUCCESS), "Failed to load remote conn info", my_name);

    //one-sided connect
    if (!id) {
        const auto ret = ucx->connect(other);
        nixl_exit_on_failure((ret == NIXL_SUCCESS), "Failed to connect", my_name);
    }

    done[id].store(true);
    while (!done[!id].load()) {
        if (id && !progress_thread) {
            ucx->progress();
        }
    }

    std::cout << "Thread passed with id " << id << "\n";

    //test one-sided disconnect
    if (!id) {
        ucx->disconnect(other);
    }

    disconnect[id].store(true);
    //wait for other
    while (!disconnect[!id].load())
        ;

    if (!progress_thread) {
        ucx->progress();
    }

    std::cout << "Thread disconnected with id " << id << "\n";
}

void
test_perform(const unsigned first, const bool progress_thread) {
    conn_info[0].clear();
    conn_info[1].clear();
    ready[0].store(false);
    ready[1].store(false);
    done[0].store(false);
    done[1].store(false);
    disconnect[0].store(false);
    disconnect[1].store(false);

    std::cout << "Multithread test start\n";
    std::cout << "Progress thread " << progress_thread << "\n";

    std::thread th1(test_thread, first, progress_thread);
    std::thread th2(test_thread, 1 - first, progress_thread);

    th1.join();
    th2.join();

    std::cout << "Multithread test done\n";
}

int
main() {
    for (unsigned i = 0; i < 12; ++i) {
        test_perform(i & 1, true);
        test_perform(i & 1, false);
    }
    return 0;
}
