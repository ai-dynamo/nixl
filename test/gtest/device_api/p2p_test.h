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

#ifndef TEST_GTEST_DEVICE_API_P2P_TEST_H
#define TEST_GTEST_DEVICE_API_P2P_TEST_H

#include "agent.h"
#include "kernel_params.h"
#include "mem_type_array.h"
#include "params.h"

#include "gtest/gtest.h"

#include <vector>

namespace nixl::device_api {
class p2pTest : public testing::TestWithParam<params> {
protected:
    explicit p2pTest(const std::vector<size_t> &sizes = {});

    [[nodiscard]] nixl_mem_t
    getDstMemType() const {
        return GetParam().memType;
    }

    [[nodiscard]] nixl_gpu_level_t
    getLevel() const {
        return GetParam().level;
    }

    void
    addDataBuffers();

    void
    addSignalBuffers();

    [[nodiscard]] size_t
    dataBufferCount() const {
        return sizes_.size();
    }

    [[nodiscard]] memTypeArray<uint8_t> &
    srcBuffer(size_t index) {
        return srcBuffers_[index];
    }

    [[nodiscard]] const memTypeArray<uint8_t> &
    dstBuffer(size_t index) const {
        return dstBuffers_[index];
    }

    [[nodiscard]] nixlGpuXferReqH
    createGpuXferReq();

private:
    agent sender_;
    agent receiver_;
    const std::vector<size_t> sizes_;
    std::vector<memTypeArray<uint8_t>> srcBuffers_;
    std::vector<memTypeArray<uint8_t>> dstBuffers_;
};
} // namespace nixl::device_api
#endif // TEST_GTEST_DEVICE_API_P2P_TEST_H
