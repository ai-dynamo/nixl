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

#ifndef LINUXAIO_QUEUE_H
#define LINUXAIO_QUEUE_H

#include <vector>
#include <libaio.h>
#include "posix_queue.h"

// Forward declare Error class
class nixlPosixBackendReqH;

class linuxAioQueue : public nixlPosixQueue {
private:
    io_context_t io_ctx; // I/O context
    std::vector<struct iocb> ios; // Array of I/Os
    int num_entries; // Total number of entries expected
    std::vector<struct iocb *> ios_to_submit; // Array of I/Os to submit
    int num_ios_to_submit; // Total number of entries to submit
    std::vector<bool> completed; // Track completed I/Os
    int num_completed; // Number of completed operations
    nixl_xfer_op_t operation; // Whether this is a read operation

    // Delete copy and move operations
    linuxAioQueue(const linuxAioQueue &) = delete;
    linuxAioQueue &
    operator=(const linuxAioQueue &) = delete;
    linuxAioQueue(linuxAioQueue &&) = delete;
    linuxAioQueue &
    operator=(linuxAioQueue &&) = delete;

public:
    linuxAioQueue(int num_entries, nixl_xfer_op_t operation);
    ~linuxAioQueue();
    nixl_status_t
    submit(const nixl_meta_dlist_t &, const nixl_meta_dlist_t &) override;
    nixl_status_t
    checkCompleted() override;
    nixl_status_t
    prepIO(int fd, void *buf, size_t len, off_t offset) override;
};

#endif // LINUXAIO_QUEUE_H
