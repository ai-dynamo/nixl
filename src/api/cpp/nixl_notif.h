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
#ifndef NIXL_SRC_API_CPP_NIXL_NOTIF_H
#define NIXL_SRC_API_CPP_NIXL_NOTIF_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>

class nixlNotifDispatcher;

/**
 * A notification receiver invoked on the backend progress path. Message and sender views
 * are valid for the invocation. Return true to consume the notification, or false to
 * leave it available through getNotifs(). An exception is treated as a refusal.
 *
 * Keep callbacks short and hand work off to another thread. A callback must not call
 * NIXL APIs or wait for another thread to do so, including registration token operations.
 * A callback can run concurrently on several backend threads.
 */
using nixl_notif_callback_t =
    std::function<bool(std::string_view remote_agent, std::string_view message)>;

/** Move-only registration owner. A token may outlive the agent that issued it. */
class nixlNotifRegistration {
public:
    nixlNotifRegistration() noexcept = default;
    ~nixlNotifRegistration() noexcept;
    nixlNotifRegistration(nixlNotifRegistration &&other) noexcept;
    nixlNotifRegistration &
    operator=(nixlNotifRegistration &&other) noexcept;
    nixlNotifRegistration(const nixlNotifRegistration &) = delete;
    nixlNotifRegistration &
    operator=(const nixlNotifRegistration &) = delete;

    /**
     * Deregister after any running callbacks of the agent finish. Does nothing for an
     * empty token or once the agent is destroyed. Synchronize access to the token itself
     * across threads.
     */
    void
    reset() noexcept;

    /**
     * Whether the token holds a registration with an active notification dispatcher.
     * The token tests false and becomes reusable once its notification dispatcher shuts down.
     */
    [[nodiscard]] explicit
    operator bool() const noexcept;

private:
    friend class nixlNotifDispatcher;

    std::weak_ptr<nixlNotifDispatcher> dispatcher_;
    std::uint64_t id_ = 0; // Zero for an empty token.
};

#endif
