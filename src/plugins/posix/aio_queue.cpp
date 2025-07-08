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

#include "aio_queue.h"
#include "posix_backend.h"
#include <errno.h>
#include "common/nixl_log.h"
#include <string.h>
#include <time.h>
#include <stdexcept>

aioQueue::aioQueue(int num_entries, nixl_xfer_op_t operation)
    : aiocbs(num_entries),
      num_entries(num_entries),
      completed(num_entries),
      num_completed(0),
      num_submitted(0),
      operation(operation) {
    if (num_entries <= 0) {
        throw std::runtime_error("Invalid number of entries for AIO queue");
    }
    for (auto& aiocb : aiocbs) {
        memset(&aiocb, 0, sizeof(struct aiocb));
    }
}

aioQueue::~aioQueue() {
    // There should not be any in-flight I/Os at destruction time
    if (num_submitted > num_completed) {
        NIXL_ERROR << "Programming error: Destroying aioQueue with " << (num_submitted - num_completed) << " in-flight I/Os";
    }

    // Cancel any remaining I/Os
    for (auto& aiocb : aiocbs) {
        if (aiocb.aio_fildes != 0) {
            aio_cancel(aiocb.aio_fildes, &aiocb);
        }
    }
}

nixl_status_t
aioQueue::submit (const nixl_meta_dlist_t &, const nixl_meta_dlist_t &) {
    num_submitted = 0;
    // Submit all I/Os at once
    for (auto& aiocb : aiocbs) {
        if (aiocb.aio_fildes == 0 || aiocb.aio_nbytes == 0) continue;
        // Check if file descriptor is valid
        if (aiocb.aio_fildes < 0) {
            NIXL_ERROR << "Invalid file descriptor in AIO request";
            return NIXL_ERR_BACKEND;
        }

        int ret;
        if (operation == NIXL_READ) {
            ret = aio_read(&aiocb);
        } else {
            ret = aio_write(&aiocb);
        }

        if (ret < 0) {
            if (errno == EAGAIN) {
                // If we hit the kernel limit, cancel all submitted I/Os and return error
                NIXL_ERROR << "AIO submit failed: kernel queue full";
                for (auto& cb : aiocbs) {
                    if (cb.aio_fildes != 0) {
                        aio_cancel(cb.aio_fildes, &cb);
                    }
                }
                return NIXL_ERR_BACKEND;
            }
            NIXL_PERROR << "AIO submit failed";
            return NIXL_ERR_BACKEND;
        }

        num_submitted++;
    }

    completed.assign(num_entries, false);
    num_completed = 0;
    return NIXL_IN_PROG;
}

nixl_status_t aioQueue::checkCompleted() {
    if (num_completed == num_entries)
        return NIXL_SUCCESS;

    // Check all submitted I/Os
    for (int i = 0; i < num_entries; i++) {
        if (completed[i] || aiocbs[i].aio_fildes == 0 || aiocbs[i].aio_nbytes == 0)
            continue; // Skip completed I/Os

        int status = aio_error(&aiocbs[i]);
        if (status == 0) {  // Operation completed
            ssize_t ret = aio_return(&aiocbs[i]);
            if (ret < 0 || ret != static_cast<ssize_t>(aiocbs[i].aio_nbytes)) {
                NIXL_PERROR << "AIO operation failed or incomplete";
                return NIXL_ERR_BACKEND;
            }
            num_completed++;
            completed[i] = true;
        } else if (status == EINPROGRESS) {
            return NIXL_IN_PROG;  // At least one operation still in progress
        } else {
            NIXL_PERROR << "AIO error";
            return NIXL_ERR_BACKEND;
        }
    }

    return (num_completed == num_submitted) ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t aioQueue::prepIO(int fd, void* buf, size_t len, off_t offset) {
    // Find an unused control block
    for (auto& aiocb : aiocbs) {
        if (aiocb.aio_fildes == 0) {
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

            aiocb.aio_fildes = fd;
            aiocb.aio_buf = buf;
            aiocb.aio_nbytes = len;
            aiocb.aio_offset = offset;
            return NIXL_SUCCESS;
        }
    }
    NIXL_ERROR << "No available AIO control blocks";
    return NIXL_ERR_BACKEND;
}
