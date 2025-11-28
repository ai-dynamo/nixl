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

template class DeviceApiTestBase<nixl_gpu_level_t>;
template class DeviceApiTestBase<DeviceTestParams>;

template<typename ParamType>
nixlAgentConfig DeviceApiTestBase<ParamType>::getConfig() {
    return nixlAgentConfig(true, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 0, 100000);
}

template<typename ParamType>
nixl_b_params_t DeviceApiTestBase<ParamType>::getBackendParams() {
    nixl_b_params_t params;
    params["num_workers"] = std::to_string(numUcxWorkers);
    return params;
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::SetUp() {
    const cudaError_t cuda_error = cudaSetDevice(0);
    if (cuda_error != cudaSuccess) {
        FAIL() << "Failed to set CUDA device 0: " << cudaGetErrorString(cuda_error);
    }

    for (size_t i = 0; i < 2; i++) {
        agents_.emplace_back(std::make_unique<nixlAgent>(getAgentName(i), getConfig()));
        nixlBackendH *backend_handle = nullptr;
        const nixl_status_t status =
            agents_.back()->createBackend("UCX", getBackendParams(), backend_handle);
        ASSERT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(backend_handle, nullptr);
        backendHandles_.push_back(backend_handle);
    }
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::TearDown() {
    agents_.clear();
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::registerMem(nixlAgent &agent,
                                   const std::vector<MemBuffer> &buffers,
                                   nixl_mem_t mem_type) {
    auto reg_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
    agent.registerMem(reg_list);
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::exchangeMD(size_t from_agent, size_t to_agent) {
    for (size_t i = 0; i < agents_.size(); i++) {
        nixl_blob_t md;
        nixl_status_t status = agents_[i]->getLocalMD(md);
        ASSERT_EQ(status, NIXL_SUCCESS);

        for (size_t j = 0; j < agents_.size(); j++) {
            if (i == j) continue;
            std::string remote_agent_name;
            status = agents_[j]->loadRemoteMD(md, remote_agent_name);
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_EQ(remote_agent_name, getAgentName(i));
        }
    }
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::invalidateMD() {
    for (size_t i = 0; i < agents_.size(); i++) {
        for (size_t j = 0; j < agents_.size(); j++) {
            if (i == j) continue;
            const nixl_status_t status = agents_[j]->invalidateRemoteMD(getAgentName(i));
            ASSERT_EQ(status, NIXL_SUCCESS);
        }
    }
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::createRegisteredMem(nixlAgent &agent, size_t size, size_t count,
                                           nixl_mem_t mem_type, std::vector<MemBuffer> &out) {
    while (count-- != 0) {
        out.emplace_back(size, mem_type);
    }
    registerMem(agent, out, mem_type);
}

template<typename ParamType>
nixlAgent &DeviceApiTestBase<ParamType>::getAgent(size_t idx) {
    return *agents_[idx];
}

template<typename ParamType>
std::string DeviceApiTestBase<ParamType>::getAgentName(size_t idx) {
    return absl::StrFormat("agent_%d", idx);
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::createXferRequest(const std::vector<MemBuffer> &src_buffers,
                                         const std::vector<MemBuffer> &dst_buffers,
                                         nixl_mem_t mem_type,
                                         nixlXferReqH *&xfer_req,
                                         nixlGpuXferReqH &gpu_req_handle,
                                         const std::string &custom_param) {
    nixl_opt_args_t extra_params = {};
    extra_params.hasNotif = true;
    extra_params.notifMsg = std::string(notificationMessage);
    extra_params.backends = {backendHandles_[senderAgent]};
    if (!custom_param.empty()) {
        extra_params.customParam = custom_param;
    }

    xfer_req = nullptr;
    const nixl_status_t status =
        getAgent(senderAgent)
            .createXferReq(NIXL_WRITE,
                          makeDescList<nixlBasicDesc>(src_buffers, mem_type),
                          makeDescList<nixlBasicDesc>(dst_buffers, mem_type),
                          getAgentName(receiverAgent),
                          xfer_req,
                          &extra_params);

    ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create xfer request";
    ASSERT_NE(xfer_req, nullptr);

    const nixl_status_t gpu_status = getAgent(senderAgent).createGpuXferReq(*xfer_req, gpu_req_handle);
    ASSERT_EQ(gpu_status, NIXL_SUCCESS) << "Failed to create GPU xfer request";
    ASSERT_NE(gpu_req_handle, nullptr);
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::cleanupXferRequest(nixlXferReqH *xfer_req, nixlGpuXferReqH gpu_req_handle) {
    getAgent(senderAgent).releaseGpuXferReq(gpu_req_handle);
    const nixl_status_t status = getAgent(senderAgent).releaseXferReq(xfer_req);
    ASSERT_EQ(status, NIXL_SUCCESS);
    invalidateMD();
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::launchAndCheckKernel(const NixlDeviceKernelParams &params) {
    const auto result = launchNixlDeviceKernel(params);
    ASSERT_EQ(result.status, NIXL_SUCCESS)
        << "Kernel execution failed with status: " << result.status;
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::setupWriteTest(size_t size, size_t count, nixl_mem_t mem_type,
                                      TestSetupData &data) {
    createRegisteredMem(getAgent(senderAgent), size, count, mem_type, data.srcBuffers);
    createRegisteredMem(getAgent(receiverAgent), size, count, mem_type, data.dstBuffers);
    exchangeMD(senderAgent, receiverAgent);
    createXferRequest(data.srcBuffers, data.dstBuffers, mem_type,
                     data.xferReq, data.gpuReqHandle);
}

template<typename ParamType>
void DeviceApiTestBase<ParamType>::setupWithSignal(const std::vector<size_t> &sizes,
                                       nixl_mem_t mem_type,
                                       TestSetupData &data) {
    for (const auto size : sizes) {
        data.srcBuffers.emplace_back(size, mem_type);
        data.dstBuffers.emplace_back(size, mem_type);
    }

    nixl_opt_args_t signal_params = {.backends = {backendHandles_[receiverAgent]}};
    size_t signal_size;
    nixl_status_t status = getAgent(receiverAgent).getGpuSignalSize(signal_size, &signal_params);
    ASSERT_EQ(status, NIXL_SUCCESS);

    data.srcBuffers.emplace_back(signal_size, mem_type);
    data.dstBuffers.emplace_back(signal_size, mem_type);

    registerMem(getAgent(senderAgent), data.srcBuffers, mem_type);
    registerMem(getAgent(receiverAgent), data.dstBuffers, mem_type);

    std::vector<MemBuffer> signal_only = {data.dstBuffers.back()};
    auto signal_desc_list = makeDescList<nixlBlobDesc>(signal_only, mem_type);
    status = getAgent(receiverAgent).prepGpuSignal(signal_desc_list, &signal_params);
    ASSERT_EQ(status, NIXL_SUCCESS);

    exchangeMD(senderAgent, receiverAgent);
    createXferRequest(data.srcBuffers, data.dstBuffers, mem_type,
                     data.xferReq, data.gpuReqHandle);
}
