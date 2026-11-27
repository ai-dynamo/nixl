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
#include "notif_dispatcher.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <utility>

#include "common/nixl_log.h"

namespace {
// Runs on the noexcept receive path, so a failure to log must not escape.
void
logCallbackFailure(std::string_view remote_agent, const char *reason) noexcept {
    try {
        NIXL_ERROR << "Notification callback failed on a notification from '" << remote_agent
                   << "', which was dropped: " << reason;
    }
    catch (...) {
    }
}
} // namespace

nixl_status_t
nixlNotifDispatcher::add(std::string_view prefix, std::shared_ptr<nixlNotifCallback> callback) {
    if (!callback) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // No delimiter is required, but at most one nonempty prefix can match.
    const auto overlaps = [prefix](const Entry &existing) {
        return prefix.starts_with(existing.prefix) || existing.prefix.starts_with(prefix);
    };
    // Declared before the lock, so a rejected or failed insertion can only destroy the
    // callback after the lock is released.
    Entry entry{std::string(prefix), std::move(callback)};

    const std::unique_lock lock(mutex_);
    if (prefix.empty()) {
        if (default_) {
            return NIXL_ERR_INVALID_PARAM;
        }
        default_ = std::move(entry.callback);
        return NIXL_SUCCESS;
    }
    if (std::any_of(entries_.begin(), entries_.end(), overlaps)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    entries_.push_back(std::move(entry));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlNotifDispatcher::remove(std::string_view prefix) noexcept {
    // Released after the lock, so a callback destroyed here cannot deadlock on it.
    std::shared_ptr<nixlNotifCallback> removed;
    const std::unique_lock lock(mutex_);
    if (prefix.empty()) {
        if (!default_) {
            return NIXL_ERR_NOT_FOUND;
        }
        removed = std::move(default_);
        return NIXL_SUCCESS;
    }
    const auto it = std::find_if(entries_.begin(), entries_.end(), [prefix](const Entry &entry) {
        return entry.prefix == prefix;
    });
    if (it == entries_.end()) {
        return NIXL_ERR_NOT_FOUND;
    }
    removed = std::move(it->callback);
    entries_.erase(it);
    return NIXL_SUCCESS;
}

bool
nixlNotifDispatcher::dispatch(std::string_view remote_agent,
                              std::string_view message) const noexcept {
    const std::shared_lock lock(mutex_);
    const auto matches = [message](const Entry &entry) {
        return message.starts_with(entry.prefix);
    };
    const auto it = std::find_if(entries_.begin(), entries_.end(), matches);
    nixlNotifCallback *const callback = it != entries_.end() ? it->callback.get() : default_.get();
    if (!callback) {
        return false;
    }
    // A user exception must never unwind through the transport's receive handler. The
    // notification matched, so it does not go to getNotifs() either.
    try {
        callback->received(remote_agent, message);
    }
    catch (const std::exception &e) {
        logCallbackFailure(remote_agent, e.what());
    }
    catch (...) {
        logCallbackFailure(remote_agent, "unknown exception");
    }
    return true;
}

bool
nixlNotifDispatcher::hasRegistrations() const noexcept {
    const std::shared_lock lock(mutex_);
    return default_ || !entries_.empty();
}
