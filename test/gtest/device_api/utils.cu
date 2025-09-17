/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.cuh"

namespace gtest {
namespace gpu {

const char *GetGpuXferLevelStr(nixl_gpu_level_t level) {
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

void initTiming(unsigned long long **start_time_ptr, unsigned long long **end_time_ptr) {
    cudaMalloc(start_time_ptr, sizeof(unsigned long long));
    cudaMalloc(end_time_ptr, sizeof(unsigned long long));
    cudaMemset(*start_time_ptr, 0, sizeof(unsigned long long));
    cudaMemset(*end_time_ptr, 0, sizeof(unsigned long long));
}

void getTiming(unsigned long long *start_time_ptr,
               unsigned long long *end_time_ptr,
               unsigned long long &start_time_cpu,
               unsigned long long &end_time_cpu) {
    cudaMemcpy(&start_time_cpu, start_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end_time_cpu, end_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
}

void logResults(size_t size,
                size_t count,
                size_t num_iters,
                unsigned long long start_time_cpu,
                unsigned long long end_time_cpu) {
    auto total_time = NS_TO_SEC(end_time_cpu - start_time_cpu);
    double total_size = size * count * num_iters;
    auto bandwidth = total_size / total_time / (1024 * 1024);
    printf("Device API Results: %zux%zux%zu=%.0f bytes in %f seconds (%.2f MB/s)\n",
           size, count, num_iters, total_size, total_time, bandwidth);
}

} // namespace gpu
} // namespace gtest

nixlAgentConfig DeviceApiTestBase::getConfig() {
    return nixlAgentConfig(true,
                          false,
                          0,
                          nixl_thread_sync_t::NIXL_THREAD_SYNC_RW,
                          0,
                          100000);
}

nixl_b_params_t DeviceApiTestBase::getBackendParams() {
    nixl_b_params_t params;
    params["num_workers"] = "2";
    return params;
}

void DeviceApiTestBase::SetUp() {
    if (cudaSetDevice(0) != cudaSuccess) {
        FAIL() << "Failed to set CUDA device 0";
    }

    for (size_t i = 0; i < 2; i++) {
        agents.emplace_back(std::make_unique<nixlAgent>(getAgentName(i), getConfig()));
        nixlBackendH *backend_handle = nullptr;
        nixl_status_t status = agents.back()->createBackend("UCX", getBackendParams(), backend_handle);
        ASSERT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(backend_handle, nullptr);
        backend_handles.push_back(backend_handle);
    }
}

void DeviceApiTestBase::TearDown() {
    agents.clear();
    backend_handles.clear();
}

template<typename Desc>
nixlDescList<Desc> DeviceApiTestBase::makeDescList(const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type) {
    nixlDescList<Desc> desc_list(mem_type);
    for (const auto &buffer : buffers) {
        desc_list.addDesc(Desc(buffer, buffer.getSize(), uint64_t(DEV_ID)));
    }
    return desc_list;
}

void DeviceApiTestBase::registerMem(nixlAgent &agent, const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type) {
    auto reg_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
    agent.registerMem(reg_list);
}

void DeviceApiTestBase::completeWireup(size_t from_agent, size_t to_agent) {
    nixl_notifs_t notifs;
    nixl_status_t status = getAgent(from_agent).genNotif(getAgentName(to_agent), NOTIF_MSG);
    ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to complete wireup";

    do {
        nixl_status_t ret = getAgent(to_agent).getNotifs(notifs);
        ASSERT_EQ(ret, NIXL_SUCCESS) << "Failed to get notifications during wireup";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (notifs.size() == 0);
}

void DeviceApiTestBase::exchangeMD(size_t from_agent, size_t to_agent) {
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

void DeviceApiTestBase::invalidateMD() {
    for (size_t i = 0; i < agents.size(); i++) {
        for (size_t j = 0; j < agents.size(); j++) {
            if (i == j) continue;
            nixl_status_t status = agents[j]->invalidateRemoteMD(getAgentName(i));
            ASSERT_EQ(status, NIXL_SUCCESS);
        }
    }
}

void DeviceApiTestBase::createRegisteredMem(nixlAgent &agent,
                                           size_t size,
                                           size_t count,
                                           nixl_mem_t mem_type,
                                           std::vector<MemBuffer> &out) {
    while (count-- != 0) {
        out.emplace_back(size, mem_type);
    }

    registerMem(agent, out, mem_type);
}

nixlAgent &DeviceApiTestBase::getAgent(size_t idx) {
    return *agents[idx];
}

std::string DeviceApiTestBase::getAgentName(size_t idx) {
    return absl::StrFormat("agent_%d", idx);
}

template nixlDescList<nixlBasicDesc> DeviceApiTestBase::makeDescList<nixlBasicDesc>(const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type);
