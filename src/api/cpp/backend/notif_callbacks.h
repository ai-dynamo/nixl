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
#ifndef NIXL_SRC_API_CPP_BACKEND_NOTIF_CALLBACKS_H
#define NIXL_SRC_API_CPP_BACKEND_NOTIF_CALLBACKS_H

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using nixl_notif_callback_t = std::function<void(std::string &&, std::string &&)>;

struct nixlNotifCallback {
    nixlNotifCallback(const std::string &prefix, const nixl_notif_callback_t &callback)
        : prefix(prefix),
          callback(callback) {}

    std::string prefix;
    nixl_notif_callback_t callback;
};

[[nodiscard]] inline bool
operator<(const nixlNotifCallback &l, const nixlNotifCallback &r) noexcept {
    return l.prefix < r.prefix;
}

class nixlNotifCallbacks {
public:
    nixlNotifCallbacks() = default;

    [[nodiscard]] bool
    hasAnyCallback() const noexcept {
        return bool(default_) || !callbacks_.empty();
    }

    [[nodiscard]] bool
    hasDefaultCallback() const noexcept {
        return bool(default_);
    }

    void
    setDefaultCallback(const nixl_notif_callback_t &callback) {
        default_ = callback;
    }

    void
    addCallback(const std::string &prefix, const nixl_notif_callback_t &callback) {
        checkNewCallback(prefix, callback);

        callbacks_.emplace_back(prefix, callback);

        if (callbacks_.size() == 1) {
            commonPrefixSize_ = prefix.size();
        } else if ((commonPrefixSize_ > 0) && (commonPrefixSize_ == prefix.size())) {
            std::sort(callbacks_.begin(), callbacks_.end());
        } else {
            commonPrefixSize_ = 0;
        }
    }

    void
    call(std::string &&remote, std::string &&message) const {
        if (const iterator iter = findCallback(message); iter != callbacks_.end()) {
            iter->callback(std::move(remote), std::move(message));
        } else if (default_) {
            default_(std::move(remote), std::move(message));
        }
    }

private:
    using iterator = std::vector<nixlNotifCallback>::const_iterator;

    void
    checkNewCallback(const std::string &prefix, const nixl_notif_callback_t &callback) const {
        if (!callback) {
            throw std::runtime_error("Empty notification callback function not allowed with prefix");
        }

        if (prefix.empty()) {
            throw std::runtime_error("Empty notification callback prefix is not allowed");
        }

        if (hasOverlap(prefix)) {
            throw std::runtime_error("New notification callback prefix overlaps previous prefix");
        }
    }

    [[nodiscard]] static bool
    isPrefixOf(const std::string &prefix, const std::string &string) noexcept {
        return (string.size() >= prefix.size()) && (std::memcmp(prefix.data(), string.data(), prefix.size()) == 0);
    }

    [[nodiscard]] bool
    hasOverlap(const std::string &prefix) const noexcept {
        for (const auto &cb : callbacks_) {
            if (isPrefixOf(prefix, cb.prefix) || isPrefixOf(cb.prefix, prefix)) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] iterator
    findCallback(const std::string &message) const {
        if (callbacks_.empty()) {
            return callbacks_.end();
        }

        // Missing a delimiter of prefix in message makes straight binary
        // search only possible when all prefixes have the same size.

        if (commonPrefixSize_ > 0) {
            return findBinarySearch(message);
        } else {
            return findLinearScan(message);
        }
    }

    [[nodiscard]] iterator
    findBinarySearch(const std::string &message) const {
        if (message.size() < commonPrefixSize_) {
            return callbacks_.end();
        }

        const std::string prefix = message.substr(0, commonPrefixSize_);
        const auto iter =
            std::lower_bound(callbacks_.begin(),
                             callbacks_.end(),
                             prefix,
                             [](const nixlNotifCallback &l, const std::string &r) {
                                 return l.prefix < r;
                             });

        if ((iter == callbacks_.end()) || (prefix != iter->prefix)) {
            return callbacks_.end();
        }
        return iter;
    }

    [[nodiscard]] iterator
    findLinearScan(const std::string &message) const {
        return std::find_if(callbacks_.begin(), callbacks_.end(), [&](const nixlNotifCallback &cb){
            return isPrefixOf(cb.prefix, message);
        });
    }

    size_t commonPrefixSize_ = 0;
    nixl_notif_callback_t default_;
    std::vector<nixlNotifCallback> callbacks_;
};

#endif
