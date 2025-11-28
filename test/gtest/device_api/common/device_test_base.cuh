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

#include <cuda_runtime.h>
#include <absl/strings/str_format.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <tuple>

template<typename ParamType = nixl_gpu_level_t>
class DeviceApiTestBase : public testing::TestWithParam<ParamType> {
public:
    static constexpr size_t defaultNumThreads = 32;
    static constexpr size_t defaultNumIters = 100;
    static constexpr size_t defaultBufferSize = 128;
    static constexpr size_t defaultBufferCount = 32;
    static constexpr std::string_view notificationMessage = "notification";

    static constexpr uint64_t testSignalIncrement = 42;
    static constexpr uint32_t testPattern1 = 0xDEADBEEF;
    static constexpr uint32_t testPattern2 = 0xCAFEBABE;

    [[nodiscard]] static const std::vector<nixl_gpu_level_t> &getTestLevels() {
        static const std::vector<nixl_gpu_level_t> testLevels = {
            nixl_gpu_level_t::BLOCK,
            nixl_gpu_level_t::WARP,
            nixl_gpu_level_t::THREAD,
        };
        return testLevels;
    }

    [[nodiscard]] static const std::vector<nixl_gpu_level_t> &getPartialWriteTestLevels() {
        static const std::vector<nixl_gpu_level_t> partialWriteLevels = {
            nixl_gpu_level_t::WARP,
            nixl_gpu_level_t::THREAD,
        };
        return partialWriteLevels;
    }

    [[nodiscard]] static std::vector<DeviceTestParams> getDeviceTestParams() {
        std::vector<DeviceTestParams> params;
        const auto &levels = getTestLevels();
        const std::vector<SendMode> modes = {
            SendMode::NODELAY_WITH_REQ,
            SendMode::NODELAY_WITHOUT_REQ,
            SendMode::WITHOUT_NODELAY_WITHOUT_REQ,
        };

        for (const auto level : levels) {
            for (const auto mode : modes) {
                params.emplace_back(level, mode);
            }
        }
        return params;
    }

    [[nodiscard]] static std::vector<DeviceTestParams> getPartialWriteDeviceTestParams() {
        std::vector<DeviceTestParams> params;
        const auto &levels = getPartialWriteTestLevels();
        const std::vector<SendMode> modes = {
            SendMode::NODELAY_WITH_REQ,
            SendMode::NODELAY_WITHOUT_REQ,
            SendMode::WITHOUT_NODELAY_WITHOUT_REQ,
        };

        for (const auto level : levels) {
            for (const auto mode : modes) {
                params.emplace_back(level, mode);
            }
        }
        return params;
    }

    nixl_gpu_level_t getLevel() const {
        if constexpr (std::is_same_v<ParamType, nixl_gpu_level_t>) {
            return this->GetParam();
        } else {
            return std::get<0>(this->GetParam());
        }
    }

    template<typename T = ParamType>
    std::enable_if_t<std::is_same_v<T, DeviceTestParams>, SendMode>
    getSendMode() const {
        return std::get<1>(this->GetParam());
    }

protected:
    static constexpr size_t senderAgent = 0;
    static constexpr size_t receiverAgent = 1;
    static constexpr size_t numUcxWorkers = 32;
    static constexpr unsigned defaultChannelId = 0;

    struct TestSetupData {
        std::vector<MemBuffer> srcBuffers;
        std::vector<MemBuffer> dstBuffers;
        nixlXferReqH *xferReq = nullptr;
        nixlGpuXferReqH gpuReqHandle = nullptr;

        struct CleanupGuard {
            nixlXferReqH **xfer_req_ptr_;
            nixlGpuXferReqH *gpu_req_handle_ptr_;
            DeviceApiTestBase *test_base_;

            CleanupGuard(nixlXferReqH **xfer_req_ptr,
                        nixlGpuXferReqH *gpu_req_handle_ptr,
                        DeviceApiTestBase *test_base)
                : xfer_req_ptr_(xfer_req_ptr),
                  gpu_req_handle_ptr_(gpu_req_handle_ptr),
                  test_base_(test_base) {}

            ~CleanupGuard() {
                if (xfer_req_ptr_ && *xfer_req_ptr_ && test_base_) {
                    test_base_->cleanupXferRequest(*xfer_req_ptr_, *gpu_req_handle_ptr_);
                }
            }

            CleanupGuard(const CleanupGuard&) = delete;
            CleanupGuard& operator=(const CleanupGuard&) = delete;
        };

        CleanupGuard makeCleanupGuard(DeviceApiTestBase *test_base) {
            return CleanupGuard(&xferReq, &gpuReqHandle, test_base);
        }
    };

    static nixlAgentConfig getConfig();
    static void generateTestPattern(std::vector<uint8_t> &pattern, size_t size, size_t offset = 0) {
        constexpr size_t patternModulo = 256;
        pattern.resize(size);
        for (size_t i = 0; i < size; ++i) {
            pattern[i] = static_cast<uint8_t>((offset * patternModulo + i) % patternModulo);
        }
    }

    static void copyToDevice(void *dst, const void *src, size_t size) {
        const cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy to device failed: ") +
                                   cudaGetErrorString(err));
        }
    }

    static void copyFromDevice(void *dst, const void *src, size_t size) {
        const cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy from device failed: ") +
                                   cudaGetErrorString(err));
        }
    }

    nixl_b_params_t getBackendParams();
    void SetUp() override;
    void TearDown() override;

    template<typename Desc>
    [[nodiscard]] nixlDescList<Desc>
    makeDescList(const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type) {
        nixlDescList<Desc> descList(mem_type);
        for (const auto &buffer : buffers) {
            descList.addDesc(Desc(buffer.toUintptr(), buffer.getSize(), uint64_t(deviceId_)));
        }
        return descList;
    }

    void registerMem(nixlAgent &agent, const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type);
    void exchangeMD(size_t from_agent, size_t to_agent);
    void invalidateMD();

    void createRegisteredMem(nixlAgent &agent, size_t size, size_t count,
                            nixl_mem_t mem_type, std::vector<MemBuffer> &out);

    [[nodiscard]] nixlAgent &getAgent(size_t idx);
    [[nodiscard]] std::string getAgentName(size_t idx);

    void createXferRequest(const std::vector<MemBuffer> &srcBuffers,
                          const std::vector<MemBuffer> &dstBuffers,
                          nixl_mem_t mem_type,
                          nixlXferReqH *&xferReq,
                          nixlGpuXferReqH &gpuReqHandle,
                          const std::string &customParam = "");

    void cleanupXferRequest(nixlXferReqH *xferReq, nixlGpuXferReqH gpuReqHandle);
    void launchAndCheckKernel(const NixlDeviceKernelParams &params);
    void setupWriteTest(size_t size, size_t count, nixl_mem_t mem_type, TestSetupData &data);
    void setupWithSignal(const std::vector<size_t> &sizes,
                        nixl_mem_t mem_type,
                        TestSetupData &data);

    void initializeTestData(const std::vector<size_t> &sizes, TestSetupData &data) {
        for (size_t i = 0; i < sizes.size(); ++i) {
            std::vector<uint8_t> pattern;
            generateTestPattern(pattern, sizes[i], i);
            copyToDevice(data.srcBuffers[i].get(), pattern.data(), sizes[i]);
        }
    }

    void verifyTestData(const std::vector<size_t> &sizes, const TestSetupData &data) {
        for (size_t i = 0; i < sizes.size(); ++i) {
            std::vector<uint8_t> expected_pattern;
            std::vector<uint8_t> received_data(sizes[i]);

            generateTestPattern(expected_pattern, sizes[i], i);
            copyFromDevice(received_data.data(), data.dstBuffers[i].get(), sizes[i]);

            ASSERT_EQ(received_data, expected_pattern)
                << "Data verification failed for buffer " << i;
        }
    }

    std::vector<nixlBackendH *> backendHandles_;

private:
    static constexpr uint64_t deviceId_ = 0;
    std::vector<std::unique_ptr<nixlAgent>> agents_;
};

#endif // NIXL_DEVICE_TEST_BASE_CUH
