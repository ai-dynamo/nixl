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
#include "common/status.h"
#include "common/nixl_log.h"

namespace {
    static constexpr unsigned int max_posix_ring_size_log = 10;
    static constexpr unsigned int max_posix_ring_size = 1 << max_posix_ring_size_log;
}

nixl_status_t UringQueue::init(int entries, const io_uring_params& params, bool read_op) {
    memset(&uring, 0, sizeof(uring));

    // Initialize with basic setup - need a mutable copy since the API modifies the params
    io_uring_params mutable_params = params;
    int ret = io_uring_queue_init_params(entries, &uring, &mutable_params);

    if (ret < 0) {
        NIXL_ERROR << "Failed to initialize io_uring instance: " << strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    // Log the features supported by this io_uring instance
    NIXL_INFO << "io_uring features:"
              << " SQPOLL=" << ((mutable_params.features & IORING_FEAT_SQPOLL_NONFIXED) ? "yes" : "no")
              << " IOPOLL=" << ((mutable_params.features & IORING_FEAT_FAST_POLL) ? "yes" : "no");

    return NIXL_SUCCESS;
}

UringQueue::UringQueue(int num_entries, const io_uring_params& params, bool is_read)
    : num_entries(num_entries)
    , num_completed(0)
    , is_read(is_read)
    , prep_op(is_read ?
        reinterpret_cast<io_uring_prep_func_t>(io_uring_prep_read) :
        reinterpret_cast<io_uring_prep_func_t>(io_uring_prep_write))
{
    if (num_entries <= 0) {
        throw std::invalid_argument("Invalid number of entries for UringQueue");
    }

    nixl_status_t status = init(num_entries, params, is_read);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to initialize UringQueue");
    }
}

UringQueue::~UringQueue() {
    io_uring_queue_exit(&uring);
}

nixl_status_t UringQueue::submit() {
    int ret = io_uring_submit(&uring);
    if (ret != num_entries) {
        NIXL_ERROR << "io_uring submit failed: " << strerror(-ret);
        return NIXL_ERR_BACKEND;
    }
    return NIXL_IN_PROG;  // Changed to IN_PROG since we need to wait for completion
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
            NIXL_ERROR << "IO operation failed: " << strerror(-res);
            return NIXL_ERR_BACKEND;
        }
        count++;
    }

    // Mark all seen
    io_uring_cq_advance(&uring, count);
    num_completed += count;

    // Log progress periodically
    if (num_completed % (num_entries / 10) == 0) {
        NIXL_DEBUG << "Queue progress: "
                   <<  (num_completed * 100.0 / num_entries) << "% complete";
    }

    return (num_completed == num_entries) ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t UringQueue::prepareIO(int fd, void* buf, size_t len, off_t offset) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&uring);
    if (!sqe) {
        NIXL_ERROR << "Failed to get io_uring submission queue entry";
        return NIXL_ERR_BACKEND;
    }

    prep_op(sqe, fd, buf, len, offset);
    return NIXL_SUCCESS;
}
