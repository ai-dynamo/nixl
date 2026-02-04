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

#include "device_test_base.cuh"

namespace nixl::test::device_api {
template class deviceApiTestBase<nixl_gpu_level_t>;
template class deviceApiTestBase<device_test_params_t>;

template<typename paramType>
nixlAgentConfig
deviceApiTestBase<paramType>::getConfig() {
    return nixlAgentConfig(true, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 0, 100000);
}

template<typename paramType>
nixl_b_params_t
deviceApiTestBase<paramType>::getBackendParams() {
    nixl_b_params_t params;
    params["num_workers"] = std::to_string(defaultNumUcxWorkers);
    return params;
}

template<typename paramType>
void
deviceApiTestBase<paramType>::SetUp() {
    if constexpr (!std::is_same_v<paramType, nixl_gpu_level_t>) {
        if (getSendMode() == send_mode_t::MULTI_CHANNEL) {
            setenv("UCX_RC_GDA_NUM_CHANNELS", "32", 1);
        }
    }

    const cudaError_t cuda_error = cudaSetDevice(0);
    ASSERT_EQ(cuda_error, cudaSuccess)
        << "Failed to set CUDA device 0: " << cudaGetErrorString(cuda_error);

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

template<typename paramType>
void
deviceApiTestBase<paramType>::registerMem(nixlAgent &agent,
                                          const std::vector<testArray<uint8_t>> &buffers,
                                          nixl_mem_t mem_type) {
    auto reg_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
    agent.registerMem(reg_list);
}

template<typename paramType>
void
deviceApiTestBase<paramType>::exchangeMD(size_t from_agent, size_t to_agent) {
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

template<typename paramType>
void
deviceApiTestBase<paramType>::invalidateMD() {
    for (size_t i = 0; i < agents_.size(); i++) {
        for (size_t j = 0; j < agents_.size(); j++) {
            if (i == j) continue;
            const nixl_status_t status = agents_[j]->invalidateRemoteMD(getAgentName(i));
            ASSERT_EQ(status, NIXL_SUCCESS);
        }
    }
}

template<typename paramType>
void
deviceApiTestBase<paramType>::createRegisteredMem(nixlAgent &agent,
                                                  size_t size,
                                                  size_t count,
                                                  nixl_mem_t mem_type,
                                                  std::vector<testArray<uint8_t>> &buffers_out) {
    while (count-- != 0) {
        buffers_out.emplace_back(size, mem_type);
    }
    registerMem(agent, buffers_out, mem_type);
}

template<typename paramType>
nixlAgent &
deviceApiTestBase<paramType>::getAgent(size_t idx) const {
    return *agents_[idx];
}

template<typename paramType>
std::string
deviceApiTestBase<paramType>::getAgentName(size_t idx) const {
    return absl::StrFormat("agent_%d", idx);
}

template<typename paramType>
void
deviceApiTestBase<paramType>::createXferRequest(const std::vector<testArray<uint8_t>> &src_buffers,
                                                nixl_mem_t src_mem_type,
                                                const std::vector<testArray<uint8_t>> &dst_buffers,
                                                nixl_mem_t dst_mem_type,
                                                nixlXferReqH *&xfer_req,
                                                nixlGpuXferReqH &gpu_req_handle,
                                                std::string_view custom_param) {
    nixl_opt_args_t extra_params;
    extra_params.hasNotif = true;
    extra_params.notifMsg = std::string(notificationMessage);
    extra_params.backends = {getBackendHandle(senderAgent)};
    if (!custom_param.empty()) {
        extra_params.customParam = std::string(custom_param);
    }

    xfer_req = nullptr;
    const nixl_status_t status =
        getAgent(senderAgent)
            .createXferReq(NIXL_WRITE,
                           makeDescList<nixlBasicDesc>(src_buffers, src_mem_type),
                           makeDescList<nixlBasicDesc>(dst_buffers, dst_mem_type),
                           getAgentName(receiverAgent),
                           xfer_req,
                           &extra_params);

    ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create xfer request";
    ASSERT_NE(xfer_req, nullptr);

    const nixl_status_t gpu_status =
        getAgent(senderAgent).createGpuXferReq(*xfer_req, gpu_req_handle);
    ASSERT_EQ(gpu_status, NIXL_SUCCESS) << "Failed to create GPU xfer request";
    ASSERT_NE(gpu_req_handle, nullptr);
}

template<typename paramType>
void
deviceApiTestBase<paramType>::cleanupXferRequest(nixlXferReqH *xfer_req,
                                                 nixlGpuXferReqH gpu_req_handle) {
    getAgent(senderAgent).releaseGpuXferReq(gpu_req_handle);
    const nixl_status_t status = getAgent(senderAgent).releaseXferReq(xfer_req);
    ASSERT_EQ(status, NIXL_SUCCESS);
    invalidateMD();
}

template<typename paramType>
void
deviceApiTestBase<paramType>::launchAndCheckKernel(const deviceKernelParams &params) {
    const nixl_status_t status = launchDeviceKernel(params);
    ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel execution failed with status: " << status;
}

template<typename paramType>
void
deviceApiTestBase<paramType>::setupWriteTest(size_t size,
                                             size_t count,
                                             nixl_mem_t src_mem_type,
                                             nixl_mem_t dst_mem_type,
                                             testSetupData &setup_data) {
    createRegisteredMem(getAgent(senderAgent), size, count, src_mem_type, setup_data.srcBuffers);
    createRegisteredMem(getAgent(receiverAgent), size, count, dst_mem_type, setup_data.dstBuffers);
    exchangeMD(senderAgent, receiverAgent);
    createXferRequest(setup_data.srcBuffers,
                      src_mem_type,
                      setup_data.dstBuffers,
                      dst_mem_type,
                      setup_data.xferReq,
                      setup_data.gpuReqHandle);
}

template<typename paramType>
void
deviceApiTestBase<paramType>::setupWithSignal(const std::vector<size_t> &sizes,
                                              nixl_mem_t src_mem_type,
                                              nixl_mem_t dst_mem_type,
                                              testSetupData &setup_data) {
    for (const auto size : sizes) {
        setup_data.srcBuffers.emplace_back(size, src_mem_type);
        setup_data.dstBuffers.emplace_back(size, dst_mem_type);
    }

    nixl_opt_args_t signal_params = {.backends = {getBackendHandle(receiverAgent)}};
    size_t signal_size;
    nixl_status_t status = getAgent(receiverAgent).getGpuSignalSize(signal_size, &signal_params);
    ASSERT_EQ(status, NIXL_SUCCESS);

    setup_data.srcBuffers.emplace_back(signal_size, src_mem_type);
    setup_data.dstBuffers.emplace_back(signal_size, dst_mem_type);

    registerMem(getAgent(senderAgent), setup_data.srcBuffers, src_mem_type);
    registerMem(getAgent(receiverAgent), setup_data.dstBuffers, dst_mem_type);

    // Create signal descriptor list from just the last buffer
    nixlDescList<nixlBlobDesc> signal_desc_list(dst_mem_type);
    const auto &signal_buffer = setup_data.dstBuffers.back();
    signal_desc_list.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(signal_buffer.get()),
                                          signal_buffer.size(),
                                          uint64_t(deviceId_)));
    status = getAgent(receiverAgent).prepGpuSignal(signal_desc_list, &signal_params);
    ASSERT_EQ(status, NIXL_SUCCESS);

    exchangeMD(senderAgent, receiverAgent);
    createXferRequest(setup_data.srcBuffers,
                      src_mem_type,
                      setup_data.dstBuffers,
                      dst_mem_type,
                      setup_data.xferReq,
                      setup_data.gpuReqHandle);
}
} // namespace nixl::test::device_api
