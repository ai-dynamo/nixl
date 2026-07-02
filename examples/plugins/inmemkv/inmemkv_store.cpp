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
 * @file inmemkv_store.cpp
 * @brief InMemKVStore put/get/exists implementation.
 */

#include "inmemkv_store.h"
#include <algorithm>
#include <cstring>

nixl_status_t
InMemKVStore::put(std::string_view key, const uint8_t *data, size_t len) {
    auto &value = store_[std::string(key)];
    value.resize(len);
    if (len > 0) {
        std::memcpy(value.data(), data, len);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
InMemKVStore::get(std::string_view key, uint8_t *buffer, size_t len, size_t &bytes_read) const {
    auto it = store_.find(std::string(key));
    if (it == store_.end()) {
        bytes_read = 0;
        return NIXL_ERR_NOT_FOUND;
    }

    bytes_read = std::min(it->second.size(), len);
    if (bytes_read > 0) {
        std::memcpy(buffer, it->second.data(), bytes_read);
    }
    return NIXL_SUCCESS;
}

bool
InMemKVStore::exists(std::string_view key) const {
    return store_.find(std::string(key)) != store_.end();
}
