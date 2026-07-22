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
 * @file inmemkv_backend.h
 * @brief INMEMKV example plugin - thin engine wrapper around nixlKVEngine.
 *
 * Architecture:
 *
 *   nixlInMemKVEngine          Registered with NIXL; extends nixlKVEngine
 *        |
 *        v
 *   nixlKVEngine               Shared thin wrapper in src/plugins/kv/
 *        |
 *        v
 *   nixlInMemKVEngineImpl      Plugin-specific logic (inmemkv_engine_impl.*)
 *        |
 *        v
 *   InMemKVStore (iKVStore)    In-memory put/get/exists (inmemkv_store.*)
 *
 * See INMEMKV_ARCHITECTURE.md and src/plugins/kv/kv_engine_impl.h for details.
 */

#ifndef NIXL_EXAMPLES_PLUGINS_INMEMKV_INMEMKV_BACKEND_H
#define NIXL_EXAMPLES_PLUGINS_INMEMKV_INMEMKV_BACKEND_H

#include "kv_engine.h"
#include <memory>

/**
 * @class nixlInMemKVEngine
 * @brief INMEMKV backend engine registered with the NIXL plugin loader.
 *
 * This class is intentionally thin: it wires the shared nixlKVEngine wrapper
 * to a nixlInMemKVEngineImpl instance at construction time.
 */
class nixlInMemKVEngine : public nixlKVEngine {
public:
    /**
     * @brief Constructs the INMEMKV engine and its implementation.
     * @param init_params NIXL backend initialization parameters.
     */
    explicit nixlInMemKVEngine(const nixlBackendInitParams *init_params);

    ~nixlInMemKVEngine() override = default;
};

#endif // NIXL_EXAMPLES_PLUGINS_INMEMKV_INMEMKV_BACKEND_H
