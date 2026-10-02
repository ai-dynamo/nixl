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
#ifndef NIXL_SRC_PLUGINS_KV_KV_REQ_HANDLE_H
#define NIXL_SRC_PLUGINS_KV_KV_REQ_HANDLE_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include "backend/backend_aux.h"

/**
 * @brief Shared async request handle for KV-family backend plugins.
 *
 * Tracks in-flight descriptors via an atomic pending counter and a condition
 * variable.  The C-style callback nixlKVXferCallback() decrements the counter
 * and signals the CV when all operations complete, making it compatible with
 * both C++ std::async and C vendor callback APIs.
 *
 * Lifecycle contract (same as the NIXL async transfer protocol):
 *   postXfer  — sets pending = n, dispatches n async ops, returns immediately
 *   checkXfer — reads pending / first_error atomically (non-blocking)
 *   releaseReqH — waits on CV until pending == 0, then deletes the handle
 */
class nixlKVReqH : public nixlBackendReqH {
public:
    /// Number of async operations still in flight.
    std::atomic<int> pending{0};

    /// First non-zero error code returned by any operation (set via CAS).
    std::atomic<int> first_error{0};

    /// CV + mutex used by releaseReqH to block until pending reaches zero.
    std::mutex mu;
    std::condition_variable cv;

    /// Per-descriptor state kept alive for the duration of the async call.
    struct DescState {
        std::string key;
        std::vector<struct iovec> ioVec;
    };
    std::vector<DescState> descStates;

    nixlKVReqH() = default;
    ~nixlKVReqH() override = default;
};

/**
 * @brief Standard C-style completion callback for KV async operations.
 *
 * Backends pass this (or a thin wrapper) as the completion callback to their
 * vendor API.  @p rc is the vendor return code (0 = success), @p arg is the
 * nixlKVReqH pointer cast to void*.
 *
 * The callback:
 *   1. Records the first non-zero rc via compare-exchange.
 *   2. Decrements pending.
 *   3. Notifies the CV when the last operation completes (pending reaches 0).
 */
inline void
nixlKVXferCallback(int rc, void *arg) {
    auto *req = static_cast<nixlKVReqH *>(arg);
    if (rc != 0) {
        int expected = 0;
        req->first_error.compare_exchange_strong(expected, rc, std::memory_order_relaxed);
    }
    if (req->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        // Last in-flight operation completed — wake up releaseReqH.
        std::lock_guard<std::mutex> lk(req->mu);
        req->cv.notify_all();
    }
}

#endif // NIXL_SRC_PLUGINS_KV_KV_REQ_HANDLE_H
