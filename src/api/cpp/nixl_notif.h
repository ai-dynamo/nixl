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
 * A notification receiver invoked on the backend progress path. It receives every
 * notification that matches its prefix. Message and sender views are valid for the
 * invocation. An exception is logged as an error, and the notification is dropped.
 *
 * Keep callbacks short and hand work off to another thread. A callback must not call
 * NIXL APIs or wait for another thread to do so. A callback can run concurrently on
 * several backend threads.
 */
using nixl_notif_callback_t =
    std::function<void(std::string_view remote_agent, std::string_view message)>;

#endif
