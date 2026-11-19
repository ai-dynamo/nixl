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

#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "nixl_types.h"

enum class nixl_notif_route_t {
    NIXL_NOTIF_ROUTE_CALLBACK,
    NIXL_NOTIF_ROUTE_UNMATCHED,
};

/**
 * @class nixlNotifRouter
 * @brief The notification callbacks of an agent, keyed by prefix. Prefixes do not overlap, so the
 *        largest prefix not greater than a message is the only one that can match it.
 */
class nixlNotifRouter {
public:
    /**
     * @return NIXL_SUCCESS; NIXL_ERR_INVALID_PARAM for an empty prefix or callback, or a prefix
     *         that overlaps another one
     */
    [[nodiscard]] nixl_status_t
    add(const std::string &prefix, const nixl_notif_callback_t &callback) {
        const std::lock_guard lock(mutex_);
        if (prefix.empty() || !callback || overlaps(prefix)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        callbacks_.emplace(prefix, callback);
        return NIXL_SUCCESS;
    }

    /** @return NIXL_SUCCESS; NIXL_ERR_NOT_FOUND for a prefix that was not added */
    [[nodiscard]] nixl_status_t
    remove(const std::string &prefix) {
        const std::lock_guard lock(mutex_);
        if (callbacks_.erase(prefix) == 0) {
            return NIXL_ERR_NOT_FOUND;
        }
        return NIXL_SUCCESS;
    }

    [[nodiscard]] bool
    empty() const {
        const std::lock_guard lock(mutex_);
        return callbacks_.empty();
    }

    /**
     * @brief  Run the callback whose prefix starts args.notifMessage, and move args into it.
     * @return NIXL_NOTIF_ROUTE_UNMATCHED leaves args unchanged.
     */
    [[nodiscard]] nixl_notif_route_t
    call(nixlNotifCallbackArgs &args) const {
        const std::lock_guard lock(mutex_);
        const nixl_notif_callback_t *callback = findCallback(args.notifMessage);
        if (callback == nullptr) {
            return nixl_notif_route_t::NIXL_NOTIF_ROUTE_UNMATCHED;
        }

        (*callback)(std::move(args));
        return nixl_notif_route_t::NIXL_NOTIF_ROUTE_CALLBACK;
    }

private:
    [[nodiscard]] const nixl_notif_callback_t *
    findCallback(const std::string &message) const {
        const auto greater = callbacks_.upper_bound(message);
        if (greater == callbacks_.begin()) {
            return nullptr;
        }
        const auto &[prefix, callback] = *std::prev(greater);
        return message.starts_with(prefix) ? &callback : nullptr;
    }

    [[nodiscard]] bool
    overlaps(const std::string &prefix) const {
        const auto next = callbacks_.lower_bound(prefix);
        if (next != callbacks_.end() && next->first.starts_with(prefix)) {
            return true;
        }
        return next != callbacks_.begin() && prefix.starts_with(std::prev(next)->first);
    }

    mutable std::mutex mutex_;
    std::map<std::string, nixl_notif_callback_t> callbacks_;
};

#endif
