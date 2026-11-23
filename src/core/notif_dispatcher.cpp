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
#include <mutex>
#include <utility>

nixlNotifRegistration::~nixlNotifRegistration() noexcept {
    reset();
}

nixlNotifRegistration::nixlNotifRegistration(nixlNotifRegistration &&other) noexcept
    : dispatcher_(std::move(other.dispatcher_)),
      id_(std::exchange(other.id_, 0)) {}

nixlNotifRegistration &
nixlNotifRegistration::operator=(nixlNotifRegistration &&other) noexcept {
    if (this != &other) {
        reset();
        dispatcher_ = std::move(other.dispatcher_);
        id_ = std::exchange(other.id_, 0);
    }
    return *this;
}

void
nixlNotifRegistration::reset() noexcept {
    // Destroyed last, so user destructors never keep the dispatcher alive after the agent.
    nixl_notif_callback_t removed;
    const auto id = std::exchange(id_, 0);
    if (const auto dispatcher = std::exchange(dispatcher_, {}).lock(); dispatcher && id != 0) {
        removed = dispatcher->erase(id);
    }
}

nixlNotifRegistration::
operator bool() const noexcept {
    if (id_ == 0) {
        return false;
    }
    const auto dispatcher = dispatcher_.lock();
    return dispatcher && !dispatcher->isShutDown();
}

nixl_status_t
nixlNotifDispatcher::add(std::string_view prefix,
                         nixl_notif_callback_t callback,
                         nixlNotifRegistration &token) {
    // A token whose agent has shut down tests false and is simply overwritten.
    if (!callback || token) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // No delimiter is required, but at most one nonempty prefix can match.
    const auto overlaps = [prefix](const Entry &existing) {
        return prefix.starts_with(existing.prefix) || existing.prefix.starts_with(prefix);
    };
    // Declared before the lock, so a rejected or failed insertion destroys the callback, and
    // any registration it owns, after the lock is released.
    Entry entry{0, std::string(prefix), std::move(callback)};

    const std::unique_lock lock(mutex_);
    if (isShutDown()) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (prefix.empty() ? default_.has_value() :
                         std::any_of(entries_.begin(), entries_.end(), overlaps)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto id = entry.id = ++lastId_;
    if (prefix.empty()) {
        default_ = std::move(entry);
    } else {
        entries_.push_back(std::move(entry));
    }
    token.dispatcher_ = weak_from_this();
    token.id_ = id;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlNotifDispatcher::remove(nixlNotifRegistration &token) noexcept {
    if (token.id_ != 0 && token.dispatcher_.lock().get() != this) {
        return NIXL_ERR_INVALID_PARAM;
    }
    token.reset();
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
    const Entry *entry = it != entries_.end() ? &*it : (default_ ? &*default_ : nullptr);
    if (!entry) {
        return false;
    }
    try {
        return entry->callback(remote_agent, message);
    }
    catch (...) {
        // A user exception must never unwind through the transport's receive handler.
        // A false return preserves the original message for getNotifs().
        return false;
    }
}

bool
nixlNotifDispatcher::hasRegistrations() const noexcept {
    const std::shared_lock lock(mutex_);
    return default_ || !entries_.empty();
}

nixl_notif_callback_t
nixlNotifDispatcher::erase(std::uint64_t id) noexcept {
    const std::unique_lock lock(mutex_);
    if (default_ && default_->id == id) {
        auto removed = std::move(default_->callback);
        default_.reset();
        return removed;
    }
    const auto it = std::find_if(
        entries_.begin(), entries_.end(), [id](const Entry &entry) { return entry.id == id; });
    if (it == entries_.end()) {
        return {};
    }
    auto removed = std::move(it->callback);
    entries_.erase(it);
    return removed;
}

void
nixlNotifDispatcher::shutdown() noexcept {
    // Destroyed after the lock is released.
    std::vector<Entry> entries;
    std::optional<Entry> default_entry;
    const std::unique_lock lock(mutex_);
    entries.swap(entries_);
    default_entry.swap(default_);
    shutDown_.store(true, std::memory_order_release);
}
