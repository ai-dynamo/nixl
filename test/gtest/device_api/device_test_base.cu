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

#include "device_test_base.cuh"

nixlAgentConfig
DeviceApiTestBase::getConfig() {
    return nixlAgentConfig(true, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 0, 100000);
}

nixl_b_params_t
DeviceApiTestBase::getBackendParams() {
    nixl_b_params_t params;
    params["num_workers"] = "2";
    return params;
}

void
DeviceApiTestBase::SetUp() {
    cudaError_t cuda_error = cudaSetDevice(0);
    if (cuda_error != cudaSuccess) {
        FAIL() << "Failed to set CUDA device 0: " << cudaGetErrorString(cuda_error);
    }

    for (size_t i = 0; i < 2; i++) {
        agents.emplace_back(std::make_unique<nixlAgent>(getAgentName(i), getConfig()));
        nixlBackendH *backend_handle = nullptr;
        nixl_status_t status =
            agents.back()->createBackend("UCX", getBackendParams(), backend_handle);
        ASSERT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(backend_handle, nullptr);
        backend_handles.push_back(backend_handle);
    }
}

void
DeviceApiTestBase::TearDown() {
    agents.clear();
}

void
DeviceApiTestBase::registerMem(nixlAgent &agent,
                               const std::vector<MemBuffer> &buffers,
                               nixl_mem_t mem_type) {
    auto reg_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
    agent.registerMem(reg_list);
}

void
DeviceApiTestBase::completeWireup(size_t from_agent, size_t to_agent) {
    nixl_notifs_t notifs;
    nixl_status_t status = getAgent(from_agent).genNotif(getAgentName(to_agent), notifMsg);
    ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to complete wireup";

    do {
        nixl_status_t ret = getAgent(to_agent).getNotifs(notifs);
        ASSERT_EQ(ret, NIXL_SUCCESS) << "Failed to get notifications during wireup";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (notifs.size() == 0);
}

void
DeviceApiTestBase::exchangeMD(size_t from_agent, size_t to_agent) {
    for (size_t i = 0; i < agents.size(); i++) {
        nixl_blob_t md;
        nixl_status_t status = agents[i]->getLocalMD(md);
        ASSERT_EQ(status, NIXL_SUCCESS);

        for (size_t j = 0; j < agents.size(); j++) {
            if (i == j) continue;
            std::string remote_agent_name;
            status = agents[j]->loadRemoteMD(md, remote_agent_name);
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_EQ(remote_agent_name, getAgentName(i));
        }
    }

    completeWireup(from_agent, to_agent);
}

void
DeviceApiTestBase::invalidateMD() {
    for (size_t i = 0; i < agents.size(); i++) {
        for (size_t j = 0; j < agents.size(); j++) {
            if (i == j) continue;
            nixl_status_t status = agents[j]->invalidateRemoteMD(getAgentName(i));
            ASSERT_EQ(status, NIXL_SUCCESS);
        }
    }
}

void
DeviceApiTestBase::createRegisteredMem(nixlAgent &agent,
                                       size_t size,
                                       size_t count,
                                       nixl_mem_t mem_type,
                                       std::vector<MemBuffer> &out) {
    while (count-- != 0) {
        out.emplace_back(size, mem_type);
    }

    registerMem(agent, out, mem_type);
}

nixlAgent &
DeviceApiTestBase::getAgent(size_t idx) {
    return *agents[idx];
}

std::string
DeviceApiTestBase::getAgentName(size_t idx) {
    return absl::StrFormat("agent_%d", idx);
}

void
DeviceApiTestBase::initTiming(unsigned long long **start_time_ptr,
                              unsigned long long **end_time_ptr) {
    cudaMalloc(start_time_ptr, sizeof(unsigned long long));
    cudaMalloc(end_time_ptr, sizeof(unsigned long long));
    cudaMemset(*start_time_ptr, 0, sizeof(unsigned long long));
    cudaMemset(*end_time_ptr, 0, sizeof(unsigned long long));
}

void
DeviceApiTestBase::getTiming(unsigned long long *start_time_ptr,
                             unsigned long long *end_time_ptr,
                             unsigned long long &start_time_cpu,
                             unsigned long long &end_time_cpu) {
    cudaMemcpy(&start_time_cpu, start_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end_time_cpu, end_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
}

const char *
DeviceApiTestBase::getGpuXferLevelStr(nixl_gpu_level_t level) {
    switch (level) {
    case nixl_gpu_level_t::WARP:
        return "WARP";
    case nixl_gpu_level_t::BLOCK:
        return "BLOCK";
    case nixl_gpu_level_t::THREAD:
        return "THREAD";
    default:
        return "UNKNOWN";
    }
}
