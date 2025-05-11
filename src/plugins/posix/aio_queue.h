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

#ifndef AIO_QUEUE_H
#define AIO_QUEUE_H

#include <vector>
#include <aio.h>
#include "async_queue.h"
#include "common/status.h"

// Forward declare Error class
class nixlPosixBackendReqH;

class aioQueue : public nixlPosixQueue {
    private:
        std::vector<struct aiocb> aiocbs;  // Array of AIO control blocks
        int num_entries;                   // Total number of entries expected
        int num_completed;                 // Number of completed operations
        int num_submitted;                 // Track number of submitted I/Os
        bool is_read;                      // Whether this is a read operation

        // Delete copy and move operations
        aioQueue(const aioQueue&) = delete;
        aioQueue& operator=(const aioQueue&) = delete;
        aioQueue(aioQueue&&) = delete;
        aioQueue& operator=(aioQueue&&) = delete;

    public:
        aioQueue(int num_entries, bool is_read);
        ~aioQueue();
        nixl_status_t submit() override;
        nixl_status_t checkCompleted() override;
        nixl_status_t prepareIO(int fd, void* buf, size_t len, off_t offset) override;
};

#endif // AIO_QUEUE_H
