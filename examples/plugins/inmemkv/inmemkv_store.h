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
 * @file inmemkv_store.h
 * @brief In-memory implementation of the shared iKVStore interface for INMEMKV.
 */

#ifndef NIXL_EXAMPLES_PLUGINS_INMEMKV_INMEMKV_STORE_H
#define NIXL_EXAMPLES_PLUGINS_INMEMKV_INMEMKV_STORE_H

#include "kv_store.h"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

/**
 * @class InMemKVStore
 * @brief In-memory implementation of iKVStore using unordered_map.
 */
class InMemKVStore : public iKVStore {
public:
    nixl_status_t
    put(std::string_view key, const uint8_t *data, size_t len) override;

    nixl_status_t
    get(std::string_view key, uint8_t *buffer, size_t len, size_t &bytes_read) const override;

    bool
    exists(std::string_view key) const override;

private:
    mutable std::unordered_map<std::string, std::vector<uint8_t>> store_;
};

#endif // NIXL_EXAMPLES_PLUGINS_INMEMKV_INMEMKV_STORE_H
