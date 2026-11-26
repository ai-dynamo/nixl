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
#include "nixl_monitor_impl.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <memory>
#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include "nixl.h"
#include "transfer_request.h"
#include "common/nixl_log.h"
#include "common/nixl_time.h"

std::unique_ptr<nixlCompletionMonitor>
nixlCompletionMonitor::create(const nixlAgent &agent) {
    return std::make_unique<nixlCompletionMonitorImpl>(agent);
}

nixlCompletionMonitorImpl::~nixlCompletionMonitorImpl() {
    for (nixlXferReqH *req : handles_) {
        req->monitor_ = nullptr;
    }
    close(fd_);
}

nixl_status_t
nixlCompletionMonitorImpl::wait(std::vector<nixlXferCompletion> &completions,
                                std::chrono::milliseconds timeout,
                                bool any) {
    constexpr std::chrono::milliseconds max_timeout(std::numeric_limits<int>::max());
    const int timeout_ms = static_cast<int>(
        std::clamp(timeout, std::chrono::milliseconds::zero(), max_timeout).count());

    const nixlTime::ms_t start_ms = nixlTime::getMs();
    const size_t num_completions = completions.size();
    int remaining_ms = 0;

    do {
        pollfd pfd = {fd_, POLLIN, 0};
        const int ret = poll(&pfd, 1, remaining_ms);
        if ((ret < 0) && (errno != EINTR)) {
            NIXL_PERROR << "Failed to poll monitor eventfd";
            return NIXL_ERR_UNKNOWN;
        }

        if (ret > 0) {
            // Reset the eventfd before the scan: a completion signaled during the scan
            // keeps it readable, so the next poll returns at once
            eventfd_t value;
            (void)eventfd_read(fd_, &value);
        }

        std::erase_if(handles_, [&](nixlXferReqH *req) {
            const nixl_status_t status = agent_.getXferStatus(req);
            if (status == NIXL_IN_PROG) {
                return false;
            }
            req->monitor_ = nullptr;
            completions.push_back({req, status});
            return true;
        });

        if (handles_.empty() || (any && (completions.size() > num_completions))) {
            return NIXL_SUCCESS;
        }

        remaining_ms = timeout_ms - static_cast<int>(nixlTime::getMs() - start_ms);
    } while (remaining_ms > 0);

    return NIXL_IN_PROG;
}
