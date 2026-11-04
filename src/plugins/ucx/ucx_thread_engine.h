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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_THREAD_ENGINE_H
#define NIXL_SRC_PLUGINS_UCX_UCX_THREAD_ENGINE_H

#include <mutex>
#include <ostream>
#include <stop_token>
#include <thread>
#include <utility>

#include "absl/strings/str_join.h"

#include "ucx_backend.h"
#include "ucx_utils.h"

/*
 * This class encapsulates a thread that polls one or multiple UCX workers
 */
class nixlUcxThread {
public:
    nixlUcxThread(const nixlUcxEngine *engine, std::vector<nixlUcxWorker *> workers)
        : engine_(engine),
          workers_(std::move(workers)) {}

    virtual ~nixlUcxThread() = default;

    const std::vector<nixlUcxWorker *> &
    getWorkers() const {
        return workers_;
    }

    static nixlUcxThread *&
    tlsThread() {
        static thread_local nixlUcxThread *tls = nullptr;
        return tls;
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxThread &thread) {
        const auto id_formatter = [](std::string *out, const nixlUcxWorker *worker) {
            out->append(std::to_string(worker->getId()));
        };
        return os << "thread " << &thread << "{engine: " << thread.engine_ << ", worker_ids: ["
                  << absl::StrJoin(thread.workers_, ",", id_formatter) << "]}";
    }

protected:
    /**
     * @brief Thread body, returns once a stop is requested on @p token
     */
    virtual void
    run(std::stop_token token) = 0;

    [[nodiscard]] std::jthread
    startThread() {
        return std::jthread([this](std::stop_token token) {
            tlsThread() = this;
            run(token);
        });
    }

private:
    const nixlUcxEngine *engine_;
    std::vector<nixlUcxWorker *> workers_;
};

/**
 * Engine with an optional single progress thread that progresses all shared
 * workers. The thread is started only when there are shared workers; with none
 * shared workers no progress thread is created and progress is synchronous.
 */
class nixlUcxThreadEngine : public nixlUcxEngine {
public:
    nixlUcxThreadEngine(const nixlBackendInitParams &init_params, size_t num_dedicated_workers = 0);

    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;

protected:
    void
    appendNotif(std::string &&remote_name, std::string &&msg) override;

private:
    std::mutex notifMutex_;
    std::unique_ptr<nixlUcxThread> thread_;
};

#endif // NIXL_SRC_PLUGINS_UCX_UCX_THREAD_ENGINE_H
