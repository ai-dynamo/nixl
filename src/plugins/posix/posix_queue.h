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

#ifndef POSIX_QUEUE_H
#define POSIX_QUEUE_H

#include "nixl_types.h"
#include "backend/backend_aux.h"
#include <sys/types.h>

// Abstract base class for async I/O operations
class nixlPosixQueue {
    public:
        virtual ~nixlPosixQueue() = default;
        virtual nixl_status_t
        submit (const nixl_meta_dlist_t &local, const nixl_meta_dlist_t &remote) = 0;
        virtual nixl_status_t checkCompleted() = 0;
        virtual nixl_status_t prepIO(int fd, void* buf, size_t len, off_t offset) = 0;

        enum class queue_t {
            AIO,
            URING,
            POSIXAIO,
            UNSUPPORTED,
        };
};

#endif // POSIX_QUEUE_H
