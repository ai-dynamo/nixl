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
#ifndef NIXL_SRC_CORE_NIXL_MONITOR_IMPL_H
#define NIXL_SRC_CORE_NIXL_MONITOR_IMPL_H

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <sys/eventfd.h>

#include "nixl_monitor.h"
#include "common/nixl_log.h"

class nixlCompletionMonitorImpl final : public nixlCompletionMonitor, public nixlXferCallback {
public:
    explicit nixlCompletionMonitorImpl(const nixlAgent &agent)
        : agent_(agent),
          fd_(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
        if (fd_ < 0) {
            throw std::runtime_error("Failed to create monitor eventfd");
        }
    }

    ~nixlCompletionMonitorImpl() override;

    [[nodiscard]] int
    fd() const noexcept override {
        return fd_;
    }

    [[nodiscard]] size_t
    size() const noexcept override {
        return handles_.size();
    }

    nixl_status_t
    waitAny(std::vector<nixlXferCompletion> &completions, int timeout_ms) override {
        return wait(completions, timeout_ms, true);
    }

    nixl_status_t
    waitAll(std::vector<nixlXferCompletion> &completions, int timeout_ms) override {
        return wait(completions, timeout_ms, false);
    }

    void
    attach(nixlXferReqH *req) {
        handles_.push_back(req);
    }

    void
    detach(nixlXferReqH *req) {
        std::erase(handles_, req);
    }

    void
    completed(const nixlXferCompletion &) override {
        if (eventfd_write(fd_, 1) < 0) {
            NIXL_PERROR << "Failed to write to monitor eventfd";
        }
    }

private:
    nixl_status_t
    wait(std::vector<nixlXferCompletion> &completions, int timeout_ms, bool any);

    const nixlAgent &agent_;
    const int fd_;
    std::vector<nixlXferReqH *> handles_;
};

#endif // NIXL_SRC_CORE_NIXL_MONITOR_IMPL_H
