/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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
#ifndef NIXL_SRC_PLUGINS_RADOS_RADOS_CONFIG_H
#define NIXL_SRC_PLUGINS_RADOS_RADOS_CONFIG_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/backend.h"
#include "nixl_types.h"

namespace nixl::rados {

using ceph_options_t = std::vector<std::pair<std::string, std::string>>;

struct backendConfig {
    std::string conf;
    std::string pool;
    std::string name;
    std::optional<std::string> ns;
    uint64_t status_wait_us = 0;
    ceph_options_t ceph_options;
};

inline constexpr const char *kCephPrefix = "ceph.";
inline constexpr std::size_t kCephPrefixLength = 5;

inline constexpr uint64_t kStatusWaitUsCeiling = 1000000;

// librados' LIBRADOS_ALL_NSPACES widens operations across every namespace
// instead of selecting one; kept as a literal so this parser stays librados-free.
inline constexpr const char *kAllNamespacesSentinel = "\x01";

// Strict ASCII decimal ([0-9]+), leading zeros valid, ceiling-bounded; the
// running total is checked per digit, so long digit strings fail without overflow.
inline bool
parseStatusWaitUs(const std::string &value, uint64_t &budget_us) {
    if (value.empty()) {
        return false;
    }
    uint64_t parsed = 0;
    for (const char ch : value) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        parsed = parsed * 10 + static_cast<uint64_t>(ch - '0');
        if (parsed > kStatusWaitUsCeiling) {
            return false;
        }
    }
    budget_us = parsed;
    return true;
}

inline std::optional<backendConfig>
parseCustomParams(const nixl_b_params_t &params) {
    const auto required = [&params](const char *key, std::string &value) {
        const auto opt = nixl::getBackendParamOptional<std::string>(params, key);
        if (!opt || opt->empty()) {
            return false;
        }
        value = *opt;
        return true;
    };

    backendConfig config;
    if (!required("conf", config.conf) || !required("pool", config.pool) ||
        !required("name", config.name)) {
        return std::nullopt;
    }

    for (const auto &[key, value] : params) {
        if (key == "conf" || key == "pool" || key == "name") {
            continue;
        }
        // Reserved ahead of the unknown-key rule; the exact match leaves
        // `ceph.status_wait_us` a normal Ceph option.
        if (key == "status_wait_us") {
            if (!parseStatusWaitUs(value, config.status_wait_us)) {
                return std::nullopt;
            }
            continue;
        }
        if (key == "namespace") {
            if (value.empty() || value == kAllNamespacesSentinel ||
                value.find('\0') != std::string::npos) {
                return std::nullopt;
            }
            config.ns = value;
            continue;
        }
        if (key.rfind(kCephPrefix, 0) != 0 || key.size() == kCephPrefixLength) {
            return std::nullopt;
        }
        config.ceph_options.emplace_back(key.substr(kCephPrefixLength), value);
    }
    std::sort(config.ceph_options.begin(),
              config.ceph_options.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    return config;
}

} // namespace nixl::rados

#endif
