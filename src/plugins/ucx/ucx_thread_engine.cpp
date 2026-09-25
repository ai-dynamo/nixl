/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucx_thread_engine.h"

#include "common/nixl_log.h"
#include "common/nixl_time.h"
#include "common/scoped_fd.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>
#include <poll.h>
#include <unistd.h>

#include "ucx_utils.h"

namespace {

class nixlUcxSharedThread final : public nixlUcxThread {
public:
    nixlUcxSharedThread(const nixlUcxEngine *engine,
                        std::vector<nixlUcxWorker *> workers,
                        nixlTime::us_t delay)
        : nixlUcxThread(engine, std::move(workers)) {
        int control_pipe[2];
        if (pipe(control_pipe) < 0) {
            throw std::runtime_error("Couldn't create progress thread control pipe");
        }
        controlPipe_[0] = nixl::scopedFd(control_pipe[0]);
        controlPipe_[1] = nixl::scopedFd(control_pipe[1]);

        // TODO: We need delay to manual periodic wakeup/polling as a temporary
        // workaround for UCX bug (poll wouldn't wake up some fds in particular
        // circumstances)

        // This will ensure that the resulting delay is at least 1ms and fits into int in order for
        // it to be compatible with poll()
        const int delay_us =
            static_cast<int>(std::min<nixlTime::us_t>(delay, std::numeric_limits<int>::max()));
        delay_ = std::chrono::ceil<std::chrono::milliseconds>(std::chrono::microseconds(delay_us));

        pollFds_.reserve(getWorkers().size() + 1);
        for (const auto *worker : getWorkers()) {
            pollFds_.push_back({worker->getEfd(), POLLIN, 0});
        }
        pollFds_.push_back({controlPipe_[0].get(), POLLIN, 0});
        thread_ = startThread();
    }

protected:
    void
    run(std::stop_token token) override {
        NIXL_DEBUG << "shared " << *this << " running";
        // A stop request wakes the poll below through the control pipe
        const std::stop_callback wake(token, [this]() {
            const char signal = 'X';
            if (write(controlPipe_[1].get(), &signal, sizeof(signal)) < 0) {
                NIXL_PERROR << "write to progress thread control pipe failed";
            }
        });

        // Set timeout event so that the main loop would progress all workers on first iteration
        bool timeout = true;
        while (!token.stop_requested()) {
            for (size_t i = 0; i < pollFds_.size() - 1; i++) {
                if (!(pollFds_[i].revents & POLLIN) && !timeout) {
                    continue;
                }
                pollFds_[i].revents = 0;
                nixlUcxWorker *worker = getWorkers()[i];
                do {
                    worker->progressLoop();
                } while (worker->arm() == NIXL_IN_PROG);
            }
            timeout = false;

            int ret;
            while ((ret = poll(pollFds_.data(), pollFds_.size(), delay_.count())) < 0) {
                NIXL_PTRACE << "Call to poll() was interrupted, retrying";
            }

            if (!ret) {
                timeout = true;
            } else if (pollFds_.back().revents & POLLIN) {
                pollFds_.back().revents = 0;

                char signal;
                int ret = read(pollFds_.back().fd, &signal, sizeof(signal));
                if (ret < 0) {
                    NIXL_PERROR << "read() on control pipe failed";
                }
            }
        }

        NIXL_DEBUG << "shared " << *this << " exiting";
    }

private:
    std::chrono::milliseconds delay_;
    nixl::scopedFd controlPipe_[2];
    std::vector<pollfd> pollFds_;
    std::jthread thread_;
};

} // namespace

nixlUcxThreadEngine::nixlUcxThreadEngine(const nixlBackendInitParams &init_params,
                                         size_t num_dedicated_workers)
    : nixlUcxEngine(init_params, num_dedicated_workers) {
    if (!init_params.enableProgTh) {
        return;
    }

    if (!nixlUcxMtLevelIsSupported(nixl::ucx::mt_mode_t::WORKER)) {
        throw std::invalid_argument("UCX library does not support multi-threading");
    }

    std::vector<nixlUcxWorker *> workers;
    workers.reserve(getSharedWorkers().size());
    for (const auto &worker : getSharedWorkers()) {
        workers.push_back(worker.get());
    }
    thread_ =
        std::make_unique<nixlUcxSharedThread>(this, std::move(workers), init_params.pthrDelay);
}

void
nixlUcxThreadEngine::appendNotif(std::string &&remote_name, std::string &&msg) {
    const std::lock_guard lock(notifMutex_);
    notifList_.emplace_back(std::move(remote_name), std::move(msg));
}

nixl_status_t
nixlUcxThreadEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!thread_) {
        progressLoop();
    }

    const std::lock_guard lock(notifMutex_);
    notifList_.swap(notif_list);
    return NIXL_SUCCESS;
}
