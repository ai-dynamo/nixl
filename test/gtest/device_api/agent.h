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

#ifndef TEST_GTEST_DEVICE_API_AGENT_H
#define TEST_GTEST_DEVICE_API_AGENT_H

#include "mem_type_array.h"

#include <nixl.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nixl::device_api {
class agent {
public:
    explicit agent(const std::string &, std::optional<unsigned> = std::nullopt);

    [[nodiscard]] nixlAgent &
    get() noexcept {
        return agent_;
    }

    void
    registerMem(const std::vector<memTypeArray<uint8_t>> &);

    [[nodiscard]] nixl_blob_t
    getLocalMD() const;

    void
    loadRemoteMD(const nixl_blob_t &);

    [[nodiscard]] nixlGpuXferReqH
    createGpuXferReq(const std::vector<memTypeArray<uint8_t>> &,
                     const std::vector<memTypeArray<uint8_t>> &);

    [[nodiscard]] size_t
    getGpuSignalSize() const;

    void
    prepGpuSignal(memTypeArray<uint8_t> &);

private:
    [[nodiscard]] nixlBackendH *createBackend(std::optional<unsigned>);

    [[nodiscard]] nixlXferReqH *
    createXferReq(const std::vector<memTypeArray<uint8_t>> &,
                  const std::vector<memTypeArray<uint8_t>> &);

    nixlAgent agent_;
    nixlBackendH *backendHandle_;
    std::unique_ptr<std::string, std::function<void(std::string *)>> remoteAgentName_;
    std::unique_ptr<nixlXferReqH, std::function<void(nixlXferReqH *)>> xferReq_ = nullptr;
    std::unique_ptr<void, std::function<void(nixlGpuXferReqH)>> gpuXferReq_ = nullptr;
};
} // namespace nixl::device_api
#endif // TEST_GTEST_DEVICE_API_AGENT_H
