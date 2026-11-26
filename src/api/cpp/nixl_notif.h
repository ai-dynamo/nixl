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

#include <functional>
#include <string_view>

/**
 * A notification receiver invoked on the receiving backend's progress path. It runs on the
 * backend's progress thread when there is one, and otherwise on the application's thread
 * inside a NIXL call that drives progress, such as getNotifs() or a transfer status check.
 * It receives every notification that matches its prefix. Message and sender views are
 * valid for the invocation. An exception is logged as an error, and the notification is
 * dropped.
 *
 * A callback must not call any NIXL agent function, including registering or deregistering
 * callbacks, and must not wait for another thread that does. It can run while the agent
 * holds internal locks, so such a call can deadlock. A callback must also hand off any
 * work that is not trivial, or that can block or take long, such as I/O, waiting on locks
 * or heavy processing, to another thread. Otherwise it stalls the backend's progress and
 * delays other notifications and transfers. A callback can run concurrently on several
 * backend threads.
 */
using nixl_notif_callback_t =
    std::function<void(std::string_view remote_agent, std::string_view message)>;

#endif
