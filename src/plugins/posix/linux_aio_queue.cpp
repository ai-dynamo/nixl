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

#include "linux_aio_queue.h"
#include "posix_backend.h"
#include <errno.h>
#include "common/nixl_log.h"
#include <string.h>
#include <time.h>
#include <stdexcept>

linuxAioQueue::linuxAioQueue(int num_entries, nixl_xfer_op_t operation)
    : io_ctx(io_context_t()),
      ios(num_entries),
      num_entries(num_entries),
      operation(operation) {
    if (num_entries <= 0) {
        throw std::runtime_error("Invalid number of entries for AIO queue");
    }

    if (operation != NIXL_READ && operation != NIXL_WRITE) {
        throw std::runtime_error("Invalid operation for AIO queue");
    }

    int res = io_queue_init(num_entries, &io_ctx);
    if (res) {
        throw std::runtime_error("io_queue_init (" + std::to_string(num_entries) +
                                 ") failed with " + std::to_string(res));
    }

    ios_to_submit.assign(num_entries, nullptr);
}

linuxAioQueue::~linuxAioQueue() {
    io_queue_release(io_ctx);
}

nixl_status_t
linuxAioQueue::submitBatch(int start_idx, int count, int &submitted_count) {
    // Submit the batch starting from start_idx
    int ret = io_submit(io_ctx, count, &ios_to_submit[start_idx]);
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat("linux_aio submit failed: %s", nixl_strerror(-ret));
        submitted_count = 0;
        return NIXL_ERR_BACKEND;
    }

    // io_submit can return partial submissions
    submitted_count = ret;
    if (ret != count) {
        NIXL_ERROR << absl::StrFormat("linux_aio submit partial: %d/%d", ret, count);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
linuxAioQueue::checkCompleted() {
    struct io_event events[32];
    int rc;
    struct timespec timeout = {0, 0};

    rc = io_getevents(io_ctx, 0, 32, events, &timeout);
    if (rc < 0) {
        NIXL_ERROR << "io_getevents error: " << rc;
        return NIXL_ERR_BACKEND;
    }

    for (int i = 0; i < rc; i++) {
        if (events[i].res < 0) {
            NIXL_ERROR << "AIO operation failed: " << events[i].res;
            return NIXL_ERR_BACKEND;
        }
    }

    num_ios_outstanding -= rc;
    if (!num_ios_outstanding) {
        // No outstanding IOs, check if we need to submit more
        if (num_ios_submitted_total < num_ios_to_submit) {
            // More IOs to submit, signal to caller to submit more
            return NIXL_IN_PROG;
        }
        // All IOs submitted and completed
        num_ios_submitted_total = 0;
        return NIXL_SUCCESS;
    }

    return NIXL_IN_PROG; // Continue until all IOs are submitted and completed
}

nixl_status_t
linuxAioQueue::prepIO(int fd, void *buf, size_t len, off_t offset) {
    if (num_ios_to_submit == num_entries) {
        NIXL_ERROR << "No available IOs";
        return NIXL_ERR_BACKEND;
    }

    // Check if file descriptor is valid
    if (fd < 0) {
        NIXL_ERROR << "Invalid file descriptor provided to prepareIO";
        return NIXL_ERR_BACKEND;
    }

    // Check buffer and length
    if (!buf || len == 0) {
        NIXL_ERROR << "Invalid buffer or length provided to prepareIO";
        return NIXL_ERR_BACKEND;
    }

    int idx = num_ios_to_submit;
    auto io = &ios[idx];

    if (operation == NIXL_READ) {
        io_prep_pread(io, fd, buf, len, offset);
    } else {
        io_prep_pwrite(io, fd, buf, len, offset);
    }

    ios_to_submit[idx] = io;
    num_ios_to_submit++;

    return NIXL_SUCCESS;
}
