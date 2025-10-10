/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>
#include <thread>

#include "ucx_backend.h"
#include "test_utils.h"


// Temporarily while fixing CI/CD pipeline
#define USE_PTHREAD false

volatile bool ready[2]  = {false, false};
volatile bool done[2]  = {false, false};
volatile bool disconnect[2]  = {false, false};
std::string conn_info[2];

void test_thread(int id)
{
    nixlBackendInitParams init_params;
    nixl_b_params_t       custom_params;
    nixl_status_t         ret;

    std::string my_name("Agent1");
    std::string other("Agent2");

    if(id){
        my_name = "Agent2";
        other = "Agent1";
    }

    init_params.localAgent   = my_name;
    init_params.enableProgTh = USE_PTHREAD;
    init_params.customParams = &custom_params;
    init_params.type         = "UCX";

    std::cout << my_name << " Started\n";

    auto ucx = nixlUcxEngine::create(init_params).release();

    if (!USE_PTHREAD) ucx->progress();

    ucx->getConnInfo(conn_info[id]);

    ready[id] = true;
    //wait for other
    while(!ready[!id]);

    ret = ucx->loadRemoteConnInfo(other, conn_info[!id]);
    nixl_exit_on_failure((ret == NIXL_SUCCESS), "Failed to load remote conn info", my_name);

    //one-sided connect
    if(!id)
        ret = ucx->connect(other);

    nixl_exit_on_failure((ret == NIXL_SUCCESS), "Failed to connect", my_name);

    done[id] = true;
    while(!done[!id])
        if (!USE_PTHREAD && id) ucx->progress();

    std::cout << "Thread passed with id " << id << "\n";

    //test one-sided disconnect
    if(!id)
        ucx->disconnect(other);

    disconnect[id] = true;
    //wait for other
    while(!disconnect[!id]);

    if (!USE_PTHREAD) ucx->progress();

    std::cout << "Thread disconnected with id " << id << "\n";

    delete ucx;
}

int main()
{
    std::cout << "Multithread test start \n";
    std::thread th1(test_thread, 0);
    std::thread th2(test_thread, 1);

    th1.join();
    th2.join();
}
