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

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>
#include <poll.h>
#include <unistd.h>

#include "ucx_utils.h"

namespace {

class nixlUcxSharedThread : public nixlUcxThread {
public:
    nixlUcxSharedThread(const nixlUcxEngine *engine, size_t num_workers, nixlTime::us_t delay)
        : nixlUcxThread(engine, num_workers) {
        if (pipe(controlPipe_) < 0) {
            throw std::runtime_error("Couldn't create progress thread control pipe");
        }
        // TODO: We need delay to manual periodic wakeup/polling as a temporary
        // workaround for UCX bug (poll wouldn't wake up some fds in particular
        // circumstances)

        // This will ensure that the resulting delay is at least 1ms and fits into int in order for
        // it to be compatible with poll()
        const int delay_us =
            static_cast<int>(std::min<nixlTime::us_t>(delay, std::numeric_limits<int>::max()));
        delay_ = std::chrono::ceil<std::chrono::milliseconds>(std::chrono::microseconds(delay_us));

        pollFds_.resize(num_workers + 1);
        pollFds_.back() = {controlPipe_[0], POLLIN, 0};
    }

    ~nixlUcxSharedThread() {
        close(controlPipe_[0]);
        close(controlPipe_[1]);
    }

    void
    join() override {
        const char signal = 'X';
        int ret = write(controlPipe_[1], &signal, sizeof(signal));
        if (ret < 0) {
            NIXL_PERROR << "write to progress thread control pipe failed";
        }
        nixlUcxThread::join();
    }

    void
    addWorker(nixlUcxWorker *worker) override {
        pollFds_[getWorkers().size()] = {worker->getEfd(), POLLIN, 0};
        nixlUcxThread::addWorker(worker);
    }

protected:
    void
    run() override {
        NIXL_DEBUG << "shared " << *this << " running";
        // Set timeout event so that the main loop would progress all workers on first iteration
        bool timeout = true;
        bool pthr_stop = false;
        while (!pthr_stop) {
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

                pthr_stop = true;
            }
        }

        NIXL_DEBUG << "shared " << *this << " exiting";
    }

private:
    std::chrono::milliseconds delay_;
    int controlPipe_[2];
    std::vector<pollfd> pollFds_;
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

    const size_t shared_count = getSharedWorkers().size();
    thread_ = std::make_unique<nixlUcxSharedThread>(this, shared_count, init_params.pthrDelay);
    for (size_t i = 0; i < shared_count; i++) {
        thread_->addWorker(getSharedWorkers()[i].get());
    }
    thread_->start();
}

nixlUcxThreadEngine::~nixlUcxThreadEngine() {
    if (thread_) {
        thread_->join();
    }
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
