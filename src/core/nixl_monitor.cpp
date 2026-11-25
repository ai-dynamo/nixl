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
#include "nixl_monitor.h"

#include <cerrno>
#include <limits>
#include <stdexcept>
#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include "nixl.h"
#include "transfer_request.h"
#include "common/nixl_log.h"
#include "common/nixl_time.h"

nixlCompletionMonitor::nixlCompletionMonitor(const nixlAgent &agent)
    : agent_(agent),
      fd_(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
    if (fd_ < 0) {
        throw std::runtime_error("Failed to create monitor eventfd");
    }
}

nixlCompletionMonitor::~nixlCompletionMonitor() {
    for (nixlXferReqH *req : handles_) {
        req->monitor_ = nullptr;
    }
    close(fd_);
}

void
nixlCompletionMonitor::attach(nixlXferReqH *req) {
    handles_.push_back(req);
}

void
nixlCompletionMonitor::detach(nixlXferReqH *req) {
    std::erase(handles_, req);
}

void
nixlCompletionMonitor::completed(const nixlXferCompletion &) {
    if (eventfd_write(fd_, 1) < 0) {
        NIXL_PERROR << "Failed to write to monitor eventfd";
    }
}

nixl_status_t
nixlCompletionMonitor::waitAny(std::vector<nixlXferCompletion> &completions, int timeout_ms) {
    return wait(completions, timeout_ms, true);
}

nixl_status_t
nixlCompletionMonitor::waitAll(std::vector<nixlXferCompletion> &completions, int timeout_ms) {
    return wait(completions, timeout_ms, false);
}

nixl_status_t
nixlCompletionMonitor::wait(std::vector<nixlXferCompletion> &completions,
                            int timeout_ms,
                            bool any) {
    if (timeout_ms < 0) {
        timeout_ms = std::numeric_limits<int>::max();
    }

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
