/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file redis_engine.cpp
 * @brief Factory wiring for nixlRedisKVEngine; creates nixlRedisKVEngineImpl at construction.
 */

#include "redis_engine.h"
#include "engine_impl.h"
#include <memory>

namespace {

std::unique_ptr<nixlKVEngineImpl>
createRedisKVEngineImpl(const nixlBackendInitParams *init_params) {
    return std::make_unique<nixlRedisKVEngineImpl>(init_params);
}

} // namespace

nixlRedisKVEngine::nixlRedisKVEngine(const nixlBackendInitParams *init_params)
    : nixlKVEngine(init_params, createRedisKVEngineImpl(init_params)) {}
