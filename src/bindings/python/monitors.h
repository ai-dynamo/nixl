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
#ifndef NIXL_SRC_BINDINGS_PYTHON_MONITORS_H
#define NIXL_SRC_BINDINGS_PYTHON_MONITORS_H

#include <sys/eventfd.h>
#include <poll.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "common/scoped_fd.h"
#include "nixl.h"

namespace nixl::python {

class MonitorClosed : public std::runtime_error {
public:
    MonitorClosed() : std::runtime_error("NIXL monitor is closed") {}
};

struct Notification {
    std::string local;
    std::string remote;
    std::string message;
};

// Native queue of one monitor. Receive and progress threads push into it and never call
// Python or log. Like the agent's own notification queue, it is unbounded and is emptied
// only by draining it. Its eventfd is readable exactly while notifications are queued or
// the inbox is closed. Only an external write to the borrowed descriptor breaks that, and
// the next wait repairs it.
class NotificationInbox {
public:
    using Clock = std::chrono::steady_clock;
    // No deadline waits indefinitely.
    using Deadline = std::optional<Clock::time_point>;

    NotificationInbox() : fd_(eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {
        if (!fd_.valid()) {
            throw std::system_error(errno, std::generic_category(), "eventfd");
        }
    }

    void
    push(std::string_view local, std::string_view remote, std::string_view message) {
        const std::lock_guard lock(mutex_);
        if (closed_) {
            return;
        }
        const bool was_empty = ready_.empty();
        ready_.push_back({std::string(local), std::string(remote), std::string(message)});
        if (was_empty) {
            signal();
        }
    }

    // Returns every queued notification once at least one is available, or an empty batch
    // at the deadline. Returns nullopt when a signal interrupts the wait, so the caller can
    // run signal handlers and wait again with the same deadline.
    [[nodiscard]] std::optional<std::vector<Notification>>
    waitAny(const Deadline &deadline) {
        std::unique_lock lock(mutex_);
        while (!closed_ && ready_.empty()) {
            int timeout_ms = -1;
            if (deadline) {
                const auto now = Clock::now();
                if (now >= *deadline) {
                    break;
                }
                timeout_ms = static_cast<int>(
                    std::chrono::ceil<std::chrono::milliseconds>(*deadline - now).count());
            }
            pollfd poll_fd{fd(), POLLIN, 0};
            lock.unlock();
            const int ret = poll(&poll_fd, 1, timeout_ms);
            const int error = errno;
            lock.lock();
            if (ret < 0) {
                if (error == EINTR) {
                    return std::nullopt;
                }
                throw std::system_error(error, std::generic_category(), "poll monitor");
            }
            if (poll_fd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                throw std::runtime_error("monitor eventfd is no longer usable");
            }
            if (ret > 0 && !closed_ && ready_.empty()) {
                clearSignal(); // A stale wakeup would otherwise make poll spin.
            }
        }
        if (closed_) {
            throw MonitorClosed();
        }
        // The queue is drained below, so the descriptor must stop being readable.
        clearSignal();
        return std::exchange(ready_, {});
    }

    void
    close() noexcept {
        const std::lock_guard lock(mutex_);
        if (!closed_) {
            closed_ = true;
            ready_.clear();
            signal();
        }
    }

    [[nodiscard]] int
    fd() const noexcept {
        return fd_.get();
    }

private:
    // Called with mutex_ held. The counter stays tiny, so a write cannot fail, and a failed
    // read (EAGAIN) means the descriptor was already clear.
    void
    signal() noexcept {
        (void)eventfd_write(fd(), 1);
    }

    void
    clearSignal() noexcept {
        eventfd_t count;
        (void)eventfd_read(fd(), &count);
    }

    const scopedFd fd_;
    mutable std::mutex mutex_;
    std::vector<Notification> ready_;
    bool closed_ = false;
};

// Queues one subscription's notifications, tagged with the subscribed agent's name.
class InboxCallback final : public nixlNotifCallback {
public:
    InboxCallback(std::shared_ptr<NotificationInbox> inbox, std::string local)
        : inbox_(std::move(inbox)),
          local_(std::move(local)) {}

    void
    received(std::string_view remote_agent, std::string_view message) override {
        inbox_->push(local_, remote_agent, message);
    }

private:
    const std::shared_ptr<NotificationInbox> inbox_;
    const std::string local_;
};

// A Python notification monitor: its queue plus the agent subscriptions that fill it.
// The monitor, its callbacks and any running wait share the queue. The Python wrapper
// keeps each subscribed agent alive until its subscriptions are removed. close()
// deregisters, which waits for running callbacks, then closes the queue and releases the
// monitor's share, so the eventfd closes as soon as no wait still uses it.
class NotificationMonitor {
public:
    NotificationMonitor() : inbox_(std::make_shared<NotificationInbox>()) {}

    ~NotificationMonitor() {
        close();
    }

    nixl_status_t
    subscribe(nixlAgent &agent, std::string local, std::string_view prefix) {
        const std::lock_guard lock(mutex_);
        if (!inbox_) {
            throw MonitorClosed();
        }
        // Allocate first, so recording a successful registration cannot fail.
        std::string owned_prefix(prefix);
        subscriptions_.reserve(subscriptions_.size() + 1);
        auto deliver = std::make_shared<InboxCallback>(inbox_, std::move(local));
        const auto status = agent.registerNotifCallback(prefix, std::move(deliver));
        if (status == NIXL_SUCCESS) {
            subscriptions_.push_back({&agent, std::move(owned_prefix)});
        }
        return status;
    }

    [[nodiscard]] bool
    unsubscribe(nixlAgent &agent, std::string_view prefix) {
        const std::lock_guard lock(mutex_);
        const auto it = std::find_if(
            subscriptions_.begin(), subscriptions_.end(), [&](const Subscription &entry) {
                return entry.agent == &agent && entry.prefix == prefix;
            });
        if (it == subscriptions_.end()) {
            return false;
        }
        (void)agent.deregisterNotifCallback(prefix);
        subscriptions_.erase(it);
        return true;
    }

    void
    close() noexcept {
        const std::lock_guard lock(mutex_);
        if (inbox_) {
            // Deregister first, so no callback can run against the closed queue.
            for (const auto &entry : subscriptions_) {
                (void)entry.agent->deregisterNotifCallback(entry.prefix);
            }
            subscriptions_.clear();
            inbox_->close(); // Wakes a waiter.
            inbox_.reset();
        }
    }

    // A wait keeps the returned queue, and its eventfd, alive until it returns.
    [[nodiscard]] std::shared_ptr<NotificationInbox>
    inbox() const {
        const std::lock_guard lock(mutex_);
        if (!inbox_) {
            throw MonitorClosed();
        }
        return inbox_;
    }

    [[nodiscard]] int
    fd() const {
        return inbox()->fd();
    }

private:
    struct Subscription {
        nixlAgent *agent;
        std::string prefix;
    };

    mutable std::mutex mutex_; // Guards inbox_ and subscriptions_.
    std::shared_ptr<NotificationInbox> inbox_;
    std::vector<Subscription> subscriptions_;
};

} // namespace nixl::python

#endif // NIXL_SRC_BINDINGS_PYTHON_MONITORS_H
