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
/**
 * @file nixl_monitor.h
 * @brief Completion monitor for NIXL transfer requests
 */
#ifndef NIXL_SRC_API_CPP_NIXL_MONITOR_H
#define NIXL_SRC_API_CPP_NIXL_MONITOR_H

#include <cstddef>
#include <memory>
#include <vector>

#include "nixl_types.h"

class nixlAgent;

/**
 * @class nixlCompletionMonitor
 * @brief Waits for the completion of transfer requests without polling each of them.
 *        A request is attached to a monitor when it is posted with nixl_opt_args_t::monitor,
 *        and detached when a wait returns it, when it is posted again or when it is released.
 *        A monitor never releases a request.
 *
 *        A monitor must not be used concurrently: one thread posts, waits for and releases
 *        its requests. It must outlive the requests posted with it until they are released.
 *
 *        Monitors are created by create() only, other implementations are not supported.
 */
class nixlCompletionMonitor {
public:
    /**
     * @param  agent         Agent that posts the requests attached to this monitor
     */
    [[nodiscard]] static std::unique_ptr<nixlCompletionMonitor>
    create(const nixlAgent &agent);

    virtual ~nixlCompletionMonitor() = default;

    /**
     * @brief  Event file descriptor for poll, epoll or an event loop. It becomes readable
     *         when an attached request completes, and the waits reset it.
     */
    [[nodiscard]] virtual int
    fd() const noexcept = 0;

    /**
     * @brief  Number of attached requests, which no wait has returned yet
     */
    [[nodiscard]] virtual size_t
    size() const noexcept = 0;

    /**
     * @brief  Wait until at least one attached request completes, successfully or not
     *
     * @param  completions   [out] All requests completed so far are appended with their final
     *                       status and detached from the monitor
     * @param  timeout_ms    Time limit in milliseconds: negative waits without a limit and
     *                       0 returns what is available without blocking
     * @return nixl_status_t NIXL_SUCCESS if a request completed or no request is attached,
     *                       NIXL_IN_PROG if the time limit expired, or an error code
     */
    virtual nixl_status_t
    waitAny(std::vector<nixlXferCompletion> &completions, int timeout_ms = -1) = 0;

    /**
     * @brief  Wait until all attached requests complete, successfully or not
     *
     * @param  completions   [out] The completed requests are appended with their final status
     *                       and detached from the monitor, also when the time limit expires
     * @param  timeout_ms    Time limit in milliseconds: negative waits without a limit and
     *                       0 returns what is available without blocking
     * @return nixl_status_t NIXL_SUCCESS if no request remains attached, NIXL_IN_PROG if the
     *                       time limit expired, or an error code
     */
    virtual nixl_status_t
    waitAll(std::vector<nixlXferCompletion> &completions, int timeout_ms = -1) = 0;
};

#endif // NIXL_SRC_API_CPP_NIXL_MONITOR_H
