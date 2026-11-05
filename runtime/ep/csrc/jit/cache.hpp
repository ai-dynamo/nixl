/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace nixl_ep_v2::jit {

class KernelRuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;

public:
    KernelRuntimeCache() = default;

    void clear() {
        cache.clear();
    }

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        // Hit the runtime cache
        if (const auto iterator = cache.find(dir_path); iterator != cache.end())
            return iterator->second;

        if (KernelRuntime::check_validity(dir_path))
            return cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);
        return nullptr;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace nixl_ep_v2::jit
