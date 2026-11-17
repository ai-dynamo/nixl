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

#ifndef NIXL_SRC_UTILS_UCX_AM_SEND_H
#define NIXL_SRC_UTILS_UCX_AM_SEND_H

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace nixl::ucx {

// AM sends in flight whose requests their callbacks release. An observer can
// outlive the endpoint or worker that counts the sends.
class amSendStats {
public:
    [[nodiscard]] size_t
    pending() const noexcept {
        return pending_.load(std::memory_order_acquire);
    }

    void
    add() noexcept {
        pending_.fetch_add(1, std::memory_order_release);
    }

    void
    remove() noexcept {
        pending_.fetch_sub(1, std::memory_order_release);
    }

private:
    std::atomic<size_t> pending_{0};
};

// Payload of an AM send in flight. With counters, the send counts as pending
// while the context exists, and the callback releases its request.
struct amSendCtx {
    amSendCtx(std::string &&data,
              std::shared_ptr<amSendStats> ep_stats,
              std::shared_ptr<amSendStats> worker_stats) noexcept
        : payload(std::move(data)),
          epStats(std::move(ep_stats)),
          workerStats(std::move(worker_stats)) {
        if (counted()) {
            epStats->add();
            workerStats->add();
        }
    }

    ~amSendCtx() {
        if (counted()) {
            epStats->remove();
            workerStats->remove();
        }
    }

    amSendCtx(const amSendCtx &) = delete;
    amSendCtx &
    operator=(const amSendCtx &) = delete;

    [[nodiscard]] bool
    counted() const noexcept {
        return epStats != nullptr;
    }

    const std::string payload;
    const std::shared_ptr<amSendStats> epStats;
    const std::shared_ptr<amSendStats> workerStats;
};

} // namespace nixl::ucx

#endif
