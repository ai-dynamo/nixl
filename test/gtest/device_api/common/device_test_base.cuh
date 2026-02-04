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

#ifndef NIXL_DEVICE_TEST_BASE_CUH
#define NIXL_DEVICE_TEST_BASE_CUH

#include <gtest/gtest.h>
#include <nixl.h>
#include <nixl_device.cuh>
#include "common.h"
#include "device_utils.cuh"
#include "device_kernels.cuh"
#include "test_array.h"

#include <cuda_runtime.h>
#include <absl/strings/str_format.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <tuple>

template<typename paramType = nixl_gpu_level_t>
class deviceApiTestBase : public testing::TestWithParam<paramType> {
public:
    static constexpr size_t defaultNumThreads = 32;
    static constexpr size_t defaultNumIters = 100;
    static constexpr size_t defaultBufferSize = 128;
    static constexpr size_t defaultBufferCount = 32;
    static constexpr size_t defaultNumUcxWorkers = 1;
    static constexpr unsigned multiChannelCount = 32;
    static constexpr std::string_view notificationMessage = "notification";

    static constexpr uint64_t testSignalIncrement = 42;
    static constexpr uint32_t testPattern1 = 0xDEADBEEF;
    static constexpr uint32_t testPattern2 = 0xCAFEBABE;

    [[nodiscard]] static const std::vector<nixl_gpu_level_t> &
    getTestLevels() {
        static const std::vector<nixl_gpu_level_t> testLevels = {
            nixl_gpu_level_t::BLOCK,
            nixl_gpu_level_t::WARP,
            nixl_gpu_level_t::THREAD,
        };
        return testLevels;
    }

    [[nodiscard]] static const std::vector<nixl_gpu_level_t> &
    getPartialWriteTestLevels() {
        static const std::vector<nixl_gpu_level_t> partialWriteLevels = {
            nixl_gpu_level_t::WARP,
            nixl_gpu_level_t::THREAD,
        };
        return partialWriteLevels;
    }

    [[nodiscard]] static std::vector<device_test_params_t>
    getDeviceTestParams() {
        return generateDeviceTestParams(getTestLevels());
    }

    [[nodiscard]] static std::vector<device_test_params_t>
    getPartialWriteDeviceTestParams() {
        return generateDeviceTestParams(getPartialWriteTestLevels());
    }

    nixl_gpu_level_t
    getLevel() const {
        if constexpr (std::is_same_v<paramType, nixl_gpu_level_t>) {
            return this->GetParam();
        } else {
            return this->GetParam().level;
        }
    }

    template<typename testType = paramType>
    std::enable_if_t<!std::is_same_v<testType, nixl_gpu_level_t>, send_mode_t>
    getSendMode() const {
        return this->GetParam().mode;
    }

    template<typename testType = paramType>
    std::enable_if_t<std::is_same_v<testType, device_test_params_t>, nixl_mem_t>
    getDstMemType() const {
        return this->GetParam().mem_type;
    }

protected:
    static constexpr size_t senderAgent = 0;
    static constexpr size_t receiverAgent = 1;
    static constexpr nixl_mem_t srcMemType = VRAM_SEG;
    static constexpr size_t defaultBufferAllocCount = 1;

    struct testSetupData {
        std::vector<testArray<uint8_t>> srcBuffers;
        std::vector<testArray<uint8_t>> dstBuffers;
        nixlXferReqH *xferReq = nullptr;
        nixlGpuXferReqH gpuReqHandle = nullptr;

        struct cleanupGuard {
            nixlXferReqH **xferReqPtr_;
            nixlGpuXferReqH *gpuReqHandlePtr_;
            deviceApiTestBase *testBase_;

            cleanupGuard(nixlXferReqH **xfer_req_ptr,
                         nixlGpuXferReqH *gpu_req_handle_ptr,
                         deviceApiTestBase *test_base)
                : xferReqPtr_(xfer_req_ptr),
                  gpuReqHandlePtr_(gpu_req_handle_ptr),
                  testBase_(test_base) {}

            ~cleanupGuard() {
                if (xferReqPtr_ && *xferReqPtr_ && testBase_) {
                    testBase_->cleanupXferRequest(*xferReqPtr_, *gpuReqHandlePtr_);
                }
            }

            cleanupGuard(const cleanupGuard &) = delete;
            cleanupGuard &
            operator=(const cleanupGuard &) = delete;
        };

        cleanupGuard
        makeCleanupGuard(deviceApiTestBase *test_base) {
            return cleanupGuard(&xferReq, &gpuReqHandle, test_base);
        }
    };

    static nixlAgentConfig
    getConfig();

    static std::vector<uint8_t>
    generateTestPattern(size_t size, size_t offset = 0) {
        constexpr size_t patternModulo = 256;
        std::vector<uint8_t> pattern(size);
        for (size_t i = 0; i < size; ++i) {
            pattern[i] = static_cast<uint8_t>((offset * patternModulo + i) % patternModulo);
        }
        return pattern;
    }

    nixl_b_params_t
    getBackendParams();

    void
    SetUp() override;

    template<typename descType>
    [[nodiscard]] nixlDescList<descType>
    makeDescList(const std::vector<testArray<uint8_t>> &buffers, nixl_mem_t mem_type) {
        nixlDescList<descType> desc_list(mem_type);
        for (const auto &buffer : buffers) {
            desc_list.addDesc(descType(
                reinterpret_cast<uintptr_t>(buffer.get()), buffer.size(), uint64_t(deviceId_)));
        }
        return desc_list;
    }

    void
    registerMem(nixlAgent &agent,
                const std::vector<testArray<uint8_t>> &buffers,
                nixl_mem_t mem_type);
    void
    exchangeMD(size_t from_agent, size_t to_agent);
    void
    invalidateMD();

    void
    createRegisteredMem(nixlAgent &agent,
                        size_t size,
                        size_t count,
                        nixl_mem_t mem_type,
                        std::vector<testArray<uint8_t>> &buffers_out);

    [[nodiscard]] nixlAgent &
    getAgent(size_t idx) const;
    [[nodiscard]] std::string
    getAgentName(size_t idx) const;

    void
    createXferRequest(const std::vector<testArray<uint8_t>> &src_buffers,
                      nixl_mem_t src_mem_type,
                      const std::vector<testArray<uint8_t>> &dst_buffers,
                      nixl_mem_t dst_mem_type,
                      nixlXferReqH *&xfer_req,
                      nixlGpuXferReqH &gpu_req_handle,
                      std::string_view custom_param = "");

    void
    cleanupXferRequest(nixlXferReqH *xfer_req, nixlGpuXferReqH gpu_req_handle);
    void
    launchAndCheckKernel(const nixlDeviceKernelParams &params);
    void
    setupWriteTest(size_t size,
                   size_t count,
                   nixl_mem_t src_mem_type,
                   nixl_mem_t dst_mem_type,
                   testSetupData &setup_data);
    void
    setupWithSignal(const std::vector<size_t> &sizes,
                    nixl_mem_t src_mem_type,
                    nixl_mem_t dst_mem_type,
                    testSetupData &setup_data);

    void
    initializeTestData(const std::vector<size_t> &sizes, testSetupData &setup_data) {
        for (size_t i = 0; i < sizes.size(); ++i) {
            const auto pattern = generateTestPattern(sizes[i], i);
            setup_data.srcBuffers[i].copyFromHost(pattern.data(), sizes[i]);
        }
    }

    [[nodiscard]] bool
    verifyTestData(const std::vector<size_t> &sizes, const testSetupData &setup_data) {
        for (size_t i = 0; i < sizes.size(); ++i) {
            const auto expected_pattern = generateTestPattern(sizes[i], i);
            std::vector<uint8_t> received_data(sizes[i]);
            setup_data.dstBuffers[i].copyToHost(received_data.data(), sizes[i]);
            if (received_data != expected_pattern) return false;
        }

        return true;
    }

    [[nodiscard]] nixlBackendH *
    getBackendHandle(size_t idx) const {
        return backendHandles_.at(idx);
    }

private:
    static constexpr uint64_t deviceId_ = 0;
    std::vector<std::unique_ptr<nixlAgent>> agents_;
    std::vector<nixlBackendH *> backendHandles_;

    static std::vector<device_test_params_t>
    generateDeviceTestParams(const std::vector<nixl_gpu_level_t> &levels) {
        std::vector<device_test_params_t> params;
        const std::vector<send_mode_t> modes{
            send_mode_t::NODELAY_WITH_REQ,
            send_mode_t::NODELAY_WITHOUT_REQ,
            send_mode_t::WITHOUT_NODELAY_WITHOUT_REQ,
            send_mode_t::MULTI_CHANNEL,
        };
        const std::vector<nixl_mem_t> mem_types{VRAM_SEG, DRAM_SEG};

        for (const auto &level : levels) {
            for (const auto &mode : modes) {
                for (const auto &mem_type : mem_types) {
                    params.emplace_back(level, mode, mem_type);
                }
            }
        }
        return params;
    }
};

#endif // NIXL_DEVICE_TEST_BASE_CUH
