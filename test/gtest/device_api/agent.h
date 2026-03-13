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

#ifndef NIXL_TEST_GTEST_DEVICE_API_AGENT_CUH
#define NIXL_TEST_GTEST_DEVICE_API_AGENT_CUH

#include "mem_buffer.h"
#include "nixl.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace nixl::gpu {
class agent {
public:
    agent(const std::string &);
    agent(const agent &) = delete;
    agent &
    operator=(const agent &) = delete;
    agent(agent &&) = delete;
    agent &
    operator=(agent &&) = delete;
    ~agent() = default;

    void
    registerMem(const memBuffer &);

    [[nodiscard]] nixl_blob_t
    getLocalMD();

    void
    loadRemoteMD(const nixl_blob_t &);

    [[nodiscard]] void *
    regAndPrepLocalMemView(const memBuffer &);

    [[nodiscard]] void *
    prepRemoteMemView(const memBuffer &);

    [[nodiscard]] void *
    prepRemoteMemView(const std::vector<memBuffer> &);

private:
    [[nodiscard]] void *
    prepLocalMemView(const memBuffer &);

    void
    addMemView(void *);

    nixlAgent agent_;
    std::optional<std::string> remoteAgentName_;
    std::unordered_set<std::unique_ptr<void, std::function<void(void *)>>> memViews_;
};
} // namespace nixl::gpu

#endif // NIXL_TEST_GTEST_DEVICE_API_AGENT_CUH
