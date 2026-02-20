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

#include <memory>
#include "queue_factory_impl.h"
#include "posix_queue.h"
#include "posix_backend.h"

#ifdef HAVE_POSIXAIO
#include "aio_queue.h"
#endif

#ifdef HAVE_LIBURING
#include "uring_queue.h"
#endif

#ifdef HAVE_LINUXAIO
#include "linux_aio_queue.h"
#endif

// Public functions implementation
std::unique_ptr<nixlPosixQueue>
QueueFactory::createPosixAioQueue(int num_entries, nixl_xfer_op_t operation) {
#ifdef HAVE_POSIXAIO
    return std::make_unique<aioQueue>(num_entries, operation);
#else
    throw nixlPosixBackendReqH::exception(
        "Attempting to create POSIX AIO queue when support is not compiled in",
        NIXL_ERR_NOT_SUPPORTED);
#endif
}

std::unique_ptr<nixlPosixQueue> QueueFactory::createUringQueue(int num_entries, nixl_xfer_op_t operation) {
#ifdef HAVE_LIBURING
    // Initialize io_uring parameters with basic configuration
    // Start with basic parameters, no special flags
    // We can add optimizations like SQPOLL later
    struct io_uring_params params = {};
    return std::make_unique<class UringQueue>(num_entries, params, operation);
#else
    throw nixlPosixBackendReqH::exception(
        "Attempting to create io_uring queue when support is not compiled in",
        NIXL_ERR_NOT_SUPPORTED);
#endif
}

std::unique_ptr<nixlPosixQueue>
QueueFactory::createLinuxAioQueue(int num_entries, nixl_xfer_op_t operation) {
#ifdef HAVE_LINUXAIO
    return std::make_unique<linuxAioQueue>(num_entries, operation);
#else
    throw nixlPosixBackendReqH::exception(
        "Attempting to create linux_aio queue when support is not compiled in",
        NIXL_ERR_NOT_SUPPORTED);
#endif
}

bool
QueueFactory::isPosixAioAvailable() {
#ifdef HAVE_POSIXAIO
    return true;
#else
    return false;
#endif
}

bool QueueFactory::isUringAvailable() {
#ifdef HAVE_LIBURING
    return true;
#else
    return false;
#endif
}

bool
QueueFactory::isLinuxAioAvailable() {
#ifdef HAVE_LINUXAIO
    return true;
#else
    return false;
#endif
}
