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

#ifndef NIXL_SRC_PLUGINS_UCX_AM_SEND_H
#define NIXL_SRC_PLUGINS_UCX_AM_SEND_H

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace nixl::ucx {

// AM sends in flight whose requests their callbacks release. An observer can
// outlive the endpoint or worker that counts the sends.
class amSendStats {
public:
    [[nodiscard]] size_t
    pending() const noexcept {
        return pending_;
    }

    void
    add() noexcept {
        ++pending_;
    }

    void
    remove() noexcept {
        --pending_;
    }

private:
    std::atomic<size_t> pending_{0};
};

// Counts one AM send as pending on its endpoint and on its worker while it
// exists.
class amPendingStats {
public:
    amPendingStats(std::shared_ptr<amSendStats> ep_stats,
                   std::shared_ptr<amSendStats> worker_stats) noexcept
        : epStats_(std::move(ep_stats)),
          workerStats_(std::move(worker_stats)) {
        epStats_->add();
        workerStats_->add();
    }

    ~amPendingStats() {
        epStats_->remove();
        workerStats_->remove();
    }

    amPendingStats(const amPendingStats &) = delete;
    amPendingStats &
    operator=(const amPendingStats &) = delete;

private:
    const std::shared_ptr<amSendStats> epStats_;
    const std::shared_ptr<amSendStats> workerStats_;
};

// Payload of an AM send in flight. A counted send is pending while its context
// exists, and its callback releases the request.
class amSendCtx {
public:
    explicit amSendCtx(std::string &&data) noexcept : payload_(std::move(data)) {}

    amSendCtx(std::string &&data,
              std::shared_ptr<amSendStats> ep_stats,
              std::shared_ptr<amSendStats> worker_stats) noexcept
        : payload_(std::move(data)),
          pendingStats_(std::in_place, std::move(ep_stats), std::move(worker_stats)) {}

    [[nodiscard]] const std::string &
    payload() const noexcept {
        return payload_;
    }

    [[nodiscard]] bool
    counted() const noexcept {
        return pendingStats_.has_value();
    }

private:
    const std::string payload_;
    const std::optional<amPendingStats> pendingStats_;
};

} // namespace nixl::ucx

#endif
