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
#ifndef NIXL_SRC_CORE_NOTIF_DISPATCHER_H
#define NIXL_SRC_CORE_NOTIF_DISPATCHER_H

#include <atomic>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include "backend/backend_aux.h"
#include "nixl_notif.h"

// Registry of an agent's notification callbacks, which its backends use as their sink.
// Callbacks run under the shared lock and every change takes the exclusive lock, so a
// change waits for running callbacks. Registration is expected around setup and teardown,
// so concurrent changes are safe but not optimized.
class nixlNotifDispatcher final : public nixlBackendNotifSink,
                                  public std::enable_shared_from_this<nixlNotifDispatcher> {
public:
    [[nodiscard]] nixl_status_t
    add(std::string_view prefix, nixl_notif_callback_t callback, nixlNotifRegistration &token);

    [[nodiscard]] nixl_status_t
    remove(nixlNotifRegistration &token) noexcept;

    [[nodiscard]] bool
    dispatch(std::string_view remote_agent, std::string_view message) const noexcept override;

    [[nodiscard]] bool
    hasRegistrations() const noexcept;

    // Release all callbacks before agent teardown, after running ones finish.
    void
    shutdown() noexcept;

private:
    friend class nixlNotifRegistration;

    [[nodiscard]] bool
    isShutDown() const noexcept {
        return shutDown_.load(std::memory_order_acquire);
    }

    struct Entry {
        std::uint64_t id;
        std::string prefix;
        nixl_notif_callback_t callback;
    };

    // Returns the removed callback for the caller to destroy outside the lock.
    [[nodiscard]] nixl_notif_callback_t
    erase(std::uint64_t id) noexcept;

    mutable std::shared_mutex mutex_;
    std::vector<Entry> entries_; // Nonempty, disjoint prefixes.
    std::optional<Entry> default_; // The empty prefix.
    std::uint64_t lastId_ = 0;
    std::atomic<bool> shutDown_{false};
};

#endif
