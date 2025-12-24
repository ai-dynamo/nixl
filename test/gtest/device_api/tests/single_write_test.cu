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

#include "common/device_test_base.cuh"

#include <cstring>

namespace gtest::nixl::gpu::single_write {

using single_write_params_t = std::tuple<nixl_gpu_level_t, send_mode_t, nixl_mem_t>;

inline std::string
memTypeStr(nixl_mem_t mem_type) {
    switch (mem_type) {
    case VRAM_SEG:
        return "vram";
    case DRAM_SEG:
        return "dram";
    default:
        return "unknown";
    }
}

struct singleWriteNameGenerator {
    static std::string
    name(const testing::TestParamInfo<single_write_params_t> &info) {
        const auto level = std::get<0>(info.param);
        const auto mode = std::get<1>(info.param);
        const auto dst_mem_type = std::get<2>(info.param);

        return std::string("UCX_") + std::string(getGpuLevelStr(level)) + "_" +
            std::string(getSendModeStr(mode)) + "_vram_to_" + memTypeStr(dst_mem_type);
    }
};

namespace {

    class singleWriteTest : public deviceApiTestBase<single_write_params_t> {
    public:
        [[nodiscard]] static std::vector<single_write_params_t>
        getParams() {
            std::vector<single_write_params_t> params;
            const auto base = deviceApiTestBase<single_write_params_t>::getDeviceTestParams();
            const std::vector<nixl_mem_t> dst_types = {VRAM_SEG, DRAM_SEG};

            for (const auto &p : base) {
                for (const auto dst : dst_types) {
                    params.emplace_back(std::get<0>(p), std::get<1>(p), dst);
                }
            }

            return params;
        }

    protected:
        [[nodiscard]] nixl_mem_t
        getDstMemType() const {
            return std::get<2>(GetParam());
        }

        void
        runTest(testSetupData &setup_data, size_t size, size_t num_iters) {
            nixlDeviceKernelParams params;
            params.operation = nixl_device_operation_t::SINGLE_WRITE;
            params.level = getLevel();
            params.numThreads = defaultNumThreads;
            params.numBlocks = 1;
            params.numIters = num_iters;
            params.reqHandle = setup_data.gpuReqHandle;
            params.singleWrite = {0, 0, 0, size, defaultChannelId};

            applySendMode(params, getSendMode());

            launchAndCheckKernel(params);
        }
    };

} // namespace

TEST_P(singleWriteTest, Basic) {
    constexpr size_t size = defaultBufferSize;
    constexpr size_t count = defaultBufferCount;
    const nixl_mem_t dst_mem_type = getDstMemType();

    testSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWriteTest(size, count, VRAM_SEG, dst_mem_type, setup_data));

    auto *src = reinterpret_cast<uint32_t *>(setup_data.srcBuffers[0].get());
    constexpr uint32_t pattern = testPattern2;
    cudaMemset(src, 0, size);
    cudaMemcpy(src, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);

    ASSERT_NO_FATAL_FAILURE(runTest(setup_data, size, defaultNumIters));

    uint32_t dst = 0;
    if (dst_mem_type == DRAM_SEG) {
        std::memcpy(&dst, setup_data.dstBuffers[0].get(), sizeof(uint32_t));
    } else {
        cudaMemcpy(&dst,
                   reinterpret_cast<uint32_t *>(setup_data.dstBuffers[0].get()),
                   sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
    }
    ASSERT_EQ(dst, pattern);
}

// Test with multiple workers using custom worker_id parameter
TEST_P(singleWriteTest, MultipleWorkers) {
    constexpr size_t size = 4096;
    constexpr size_t num_iters = 100;
    const nixl_mem_t dst_mem_type = getDstMemType();

    std::vector<std::vector<testArray<uint8_t>>> src_buffers(numUcxWorkers);
    std::vector<std::vector<testArray<uint8_t>>> dst_buffers(numUcxWorkers);
    std::vector<std::vector<uint32_t>> patterns(numUcxWorkers);

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        createRegisteredMem(getAgent(senderAgent), size, 1, VRAM_SEG, src_buffers[worker_id]);
        createRegisteredMem(getAgent(receiverAgent), size, 1, dst_mem_type, dst_buffers[worker_id]);

        constexpr size_t num_elements = size / sizeof(uint32_t);
        patterns[worker_id].resize(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            patterns[worker_id][i] = 0xDEAD0000 | static_cast<uint32_t>(worker_id);
        }
        cudaMemcpy(src_buffers[worker_id][0].get(),
                   patterns[worker_id].data(),
                   size,
                   cudaMemcpyHostToDevice);
    }

    exchangeMD(senderAgent, receiverAgent);

    std::vector<nixlXferReqH *> xfer_reqs(numUcxWorkers);
    std::vector<nixlGpuXferReqH> gpu_req_handles(numUcxWorkers);

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        const std::string custom_param = "worker_id=" + std::to_string(worker_id);
        createXferRequest(src_buffers[worker_id],
                          VRAM_SEG,
                          dst_buffers[worker_id],
                          dst_mem_type,
                          xfer_reqs[worker_id],
                          gpu_req_handles[worker_id],
                          custom_param);
    }

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        nixlDeviceKernelParams params;
        params.operation = nixl_device_operation_t::SINGLE_WRITE;
        params.level = getLevel();
        params.numThreads = defaultNumThreads;
        params.numBlocks = 1;
        params.numIters = num_iters;
        params.reqHandle = gpu_req_handles[worker_id];
        params.singleWrite = {0, 0, 0, size, defaultChannelId};

        applySendMode(params, getSendMode());

        const nixl_status_t status = launchNixlDeviceKernel(params);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel launch failed for worker " << worker_id;
    }

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        std::vector<uint32_t> received(size / sizeof(uint32_t));
        if (dst_mem_type == DRAM_SEG) {
            std::memcpy(received.data(), dst_buffers[worker_id][0].get(), size);
        } else {
            cudaMemcpy(
                received.data(), dst_buffers[worker_id][0].get(), size, cudaMemcpyDeviceToHost);
        }

        ASSERT_EQ(received, patterns[worker_id])
            << "Worker " << worker_id << " data verification failed";
    }

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        getAgent(senderAgent).releaseGpuXferReq(gpu_req_handles[worker_id]);
        nixl_status_t status = getAgent(senderAgent).releaseXferReq(xfer_reqs[worker_id]);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }

    invalidateMD();
}

} // namespace gtest::nixl::gpu::single_write

using gtest::nixl::gpu::single_write::singleWriteTest;
using gtest::nixl::gpu::single_write::singleWriteNameGenerator;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         singleWriteTest,
                         testing::ValuesIn(singleWriteTest::getParams()),
                         singleWriteNameGenerator::name);
