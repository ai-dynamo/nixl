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
    num_entries = entries;
    num_completed = 0;
    num_submitted = 0;
    is_read = read_op;
    use_fixed_files = false;
    use_fixed_buffers = false;
    memset(&uring, 0, sizeof(uring));

    // Start with basic parameters, no special flags
    io_uring_params local_params = params;

    // Initialize with basic setup
    int ret = io_uring_queue_init_params(entries, &uring, &local_params);
    if (ret < 0) {
        NIXL_ERROR << "Failed to initialize io_uring instance: " << strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    // Log the features supported by this io_uring instance
    NIXL_INFO << "io_uring features:"
              << " SQPOLL=" << ((local_params.features & IORING_FEAT_SQPOLL_NONFIXED) ? "yes" : "no")
              << " IOPOLL=" << ((local_params.features & IORING_FEAT_FAST_POLL) ? "yes" : "no");

    return NIXL_SUCCESS;
}

UringQueue::UringQueue(int num_entries, const io_uring_params& params, bool is_read) {
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
    if (ret < 0) {
        NIXL_ERROR << "io_uring submit failed: " << strerror(-ret);
        return NIXL_ERR_BACKEND;
    }
    num_submitted += ret;
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
        NIXL_INFO << "Queue progress: "
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

    if (is_read) {
        io_uring_prep_read(sqe, fd, buf, len, offset);
    } else {
        io_uring_prep_write(sqe, fd, buf, len, offset);
    }

    // Don't use IOSQE_FIXED_FILE since we're not using registered files
    sqe->flags = 0;

    return NIXL_SUCCESS;
}
