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

UringQueue::UringQueue(int num_entries, const io_uring_params &params, nixl_xfer_op_t operation)
    : num_entries(num_entries),
      num_completed(0),
      descriptors(num_entries),
      prep_op(operation == NIXL_READ ?
                  reinterpret_cast<io_uring_prep_func_t>(io_uring_prep_read) :
                  reinterpret_cast<io_uring_prep_func_t>(io_uring_prep_write)) {
    if (num_entries <= 0) {
        throw std::invalid_argument("Invalid number of entries for UringQueue");
    }

    init(num_entries, params);
}

UringQueue::~UringQueue() {
    io_uring_queue_exit(&uring);
}

nixl_status_t
UringQueue::submitBatch(int start_idx, int count, int &submitted_count) {
    // Prepare SQEs for the batch
    for (int i = 0; i < count; i++) {
        int idx = start_idx + i;
        auto &desc = descriptors[idx];

        struct io_uring_sqe *sqe = io_uring_get_sqe (&uring);
        if (!sqe) {
            NIXL_ERROR << "Failed to get io_uring submission queue entry";
            submitted_count = 0;
            return NIXL_ERR_BACKEND;
        }
        prep_op(sqe, desc.fd, desc.buf, desc.len, desc.offset);
    }

    // Submit the batch
    int ret = io_uring_submit(&uring);
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat("io_uring submit failed: %s", nixl_strerror(-ret));
        submitted_count = 0;
        return NIXL_ERR_BACKEND;
    }

    // io_uring_submit can return partial submissions
    submitted_count = ret;
    if (ret != count) {
        NIXL_ERROR << absl::StrFormat("io_uring submit partial: %d/%d", ret, count);
    }

    return NIXL_SUCCESS;
}

nixl_status_t UringQueue::checkCompleted() {
    // Check if all IOs are submitted and completed
    if (num_ios_submitted_total >= num_ios_to_submit && num_completed == num_ios_to_submit) {
        num_ios_submitted_total = 0;
        num_ios_to_submit = 0;
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
    num_ios_outstanding -= count;

    logOnPercentStep(num_completed, num_ios_to_submit);

    return NIXL_IN_PROG; // Continue until all IOs are submitted and completed
}

nixl_status_t UringQueue::prepIO(int fd, void* buf, size_t len, off_t offset) {
    if (num_ios_to_submit >= num_entries) {
        NIXL_ERROR << "No available io_uring entries";
        return NIXL_ERR_BACKEND;
    }

    descriptors[num_ios_to_submit] = {fd, buf, len, offset};
    num_ios_to_submit++;
    return NIXL_SUCCESS;
}
