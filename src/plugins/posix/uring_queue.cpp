/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "uring_queue.h"
#include <liburing.h>
#include <array>
#include <vector>
#include <cstring>
#include <stdexcept>
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "common/nixl_log.h"

namespace {
    // Log completion percentage at regular intervals (every log_percent_step percent)
    void logOnPercentStep(unsigned int completed, unsigned int total) {
        constexpr unsigned int default_log_percent_step = 10;
        static_assert (default_log_percent_step >= 1 && default_log_percent_step <= 100,
                       "log_percent_step must be in [1, 100]");
        unsigned int log_percent_step = total < 10 ? 1 : default_log_percent_step;

        if (total == 0) {
            NIXL_ERROR << "Tried to log completion percentage with total == 0";
            return;
        }
        // Only log at each percentage step
        if (completed % (total / log_percent_step) == 0) {
            NIXL_DEBUG << absl::StrFormat("Queue progress: %.1f%% complete",
                                          (completed * 100.0 / total));
        }
    }

    std::string stringifyUringFeatures(unsigned int features) {
        static const std::unordered_map<unsigned int, std::string> feature_map = {
            {IORING_FEAT_SQPOLL_NONFIXED, "SQPOLL"},
            {IORING_FEAT_FAST_POLL, "IOPOLL"}
        };

        std::vector<std::string> enabled;
        for (unsigned int bits = features; bits; bits &= (bits - 1)) { // step through each set bit
            unsigned int bit = bits & -bits; // isolate lowest set bit
            auto it = feature_map.find(bit);
            if (it != feature_map.end()) {
                enabled.push_back(it->second);
            }
        }
        return enabled.empty() ? "none" : absl::StrJoin(enabled, ", ");
    }
}

nixl_status_t UringQueue::init(int entries, const io_uring_params& params) {
    // Initialize with basic setup - need a mutable copy since the API modifies the params
    io_uring_params mutable_params = params;
    if (io_uring_queue_init_params(entries, &uring, &mutable_params) < 0) {
        throw std::runtime_error(absl::StrFormat("Failed to initialize io_uring instance: %s", nixl_strerror(errno)));
    }

    // Log the features supported by this io_uring instance
    NIXL_INFO << absl::StrFormat("io_uring features: %s", stringifyUringFeatures(mutable_params.features));

    return NIXL_SUCCESS;
}

UringQueue::UringQueue(int num_entries, const io_uring_params& params, nixl_xfer_op_t operation)
    : num_entries(num_entries)
    , num_completed(0)
    , prep_op(operation == NIXL_READ ?
        reinterpret_cast<io_uring_prep_func_t>(io_uring_prep_read) :
        reinterpret_cast<io_uring_prep_func_t>(io_uring_prep_write))
{
    if (num_entries <= 0) {
        throw std::invalid_argument("Invalid number of entries for UringQueue");
    }

    init(num_entries, params);
}

UringQueue::~UringQueue() {
    io_uring_queue_exit(&uring);
}

nixl_status_t
UringQueue::submit (const nixl_meta_dlist_t &local, const nixl_meta_dlist_t &remote) {
    for (auto [local_it, remote_it] = std::make_pair (local.begin(), remote.begin());
         local_it != local.end() && remote_it != remote.end();
         ++local_it, ++remote_it) {
        int fd = remote_it->devId;
        void *buf = reinterpret_cast<void *> (local_it->addr);
        size_t len = local_it->len;
        off_t offset = remote_it->addr;

        struct io_uring_sqe *sqe = io_uring_get_sqe (&uring);
        if (!sqe) {
            NIXL_ERROR << "Failed to get io_uring submission queue entry";
            return NIXL_ERR_BACKEND;
        }
        prep_op (sqe, fd, buf, len, offset);
    }

    int ret = io_uring_submit(&uring);
    if (ret != num_entries) {
        if (ret < 0) {
            NIXL_ERROR << absl::StrFormat("io_uring submit failed: %s", nixl_strerror(-ret));
        } else {
            NIXL_ERROR << absl::StrFormat("io_uring submit failed. Partial submission: %d/%d", num_entries, ret);
        }
        return NIXL_ERR_BACKEND;
    }
    num_completed = 0;
    return NIXL_IN_PROG;
}

nixl_status_t UringQueue::checkCompleted() {
    if (num_completed == num_entries) {
        return NIXL_SUCCESS;
    }

    // Process all available completions
    struct io_uring_cqe* cqe;
    unsigned head;
    unsigned count = 0;

    // Get completion events
    io_uring_for_each_cqe(&uring, head, cqe) {
        int res = cqe->res;
        if (res < 0) {
            NIXL_ERROR << absl::StrFormat("IO operation failed: %s", nixl_strerror(-res));
            return NIXL_ERR_BACKEND;
        }
        count++;
    }

    // Mark all seen
    io_uring_cq_advance(&uring, count);
    num_completed += count;

    logOnPercentStep(num_completed, num_entries);

    return (num_completed == num_entries) ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t UringQueue::prepIO(int fd, void* buf, size_t len, off_t offset) {
    return NIXL_SUCCESS;
}
