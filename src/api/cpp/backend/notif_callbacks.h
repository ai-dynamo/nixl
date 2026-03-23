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

#include <cassert>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using nixl_notif_callback_t = std::function<void(std::string &&, std::string &&)>;

struct nixlNotifCallback {
    nixlNotifCallback(const std::string &prefix, const nixl_notif_callback_t &callback)
        : prefix(prefix),
          callback(callback)
    {}

    std::string prefix;
    nixl_notif_callback_t callback;
};

[[nodiscard]] inline bool operator<(const nixlNotifCallback &l, const nixlNotifCallback &r) noexcept {
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
        assert(callback);
        assert(!prefix.empty());

        callbacks_.emplace_back(prefix, callback);

        if(callbacks_.size() == 1) {
            fixed_size_ = prefix.size();
        } else if((fixed_size_ > 0) && (fixed_size_ == prefix.size())) {
            std::sort(callbacks_.begin(), callbacks_.end());
        }
        else {
            fixed_size_ = 0;
        }
    }

    [[nodiscard]] bool
    call(std::string &&remote, std::string &&message) const {
        if(callbacks_.empty()) {
            return call_default(std::move(remote), std::move(message));
        }

        if(fixed_size_ > 0) {
            return call_binary_search(std::move(remote), std::move(message));
        } else {
            return call_linear_scan(std::move(remote), std::move(message));
        }
    }

private:
    [[nodiscard]] bool
    call_default(std::string &&remote, std::string &&message) const {
        if(default_) {
            default_(std::move(remote), std::move(message));
            return true;
        }
        return false;
    }

    [[nodiscard]] bool
    call_binary_search(std::string &&remote, std::string &&message) const {
        if(message.size() >= fixed_size_) {
            const std::string prefix = message.substr(0, fixed_size_);
            const auto iter = std::lower_bound(callbacks_.begin(), callbacks_.end(), message, [&](const nixlNotifCallback &l, const std::string &r) {
                    return l.prefix < prefix;
            } );

            if((iter != callbacks_.end()) && (prefix == iter->prefix)) {
                iter->callback(std::move(remote), std::move(message));
                return true;
            }
        }
        return call_default(std::move(remote), std::move(message));
    }

    [[nodiscard]] bool
    call_linear_scan(std::string &&remote, std::string &&message) const {
        for(const auto &cb : callbacks_) {
            if(message.compare(0, cb.prefix.size(), cb.prefix) == 0) {
                cb.callback(std::move(remote), std::move(message));
                return true;
            }
        }
        return call_default(std::move(remote), std::move(message));
    }

    size_t fixed_size_ = 0;
    nixl_notif_callback_t default_;
    std::vector<nixlNotifCallback> callbacks_;
};

#endif
