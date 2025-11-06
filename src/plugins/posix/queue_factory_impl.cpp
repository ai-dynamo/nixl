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

#include <memory>
#include "queue_factory_impl.h"
#include "posix_queue.h"
#include "posix_backend.h"
#include "aio_queue.h"

#ifdef HAVE_LIBURING
#include "uring_queue.h"
#endif

#ifdef HAVE_LINUXAIO
#include "linux_aio_queue.h"
#endif

// Anonymous namespace for internal template implementations for functions that use the optional liburing
namespace {
    struct uringEnabled {};
    struct uringDisabled {};

#ifdef HAVE_LIBURING
    using uringMode = uringEnabled;
#else
    using uringMode = uringDisabled;
#endif

    template <typename Mode, typename Enable = void>
    struct funcImpl;

    template <typename Mode>
    struct funcImpl<Mode, std::enable_if_t<std::is_same<Mode, uringEnabled>::value>> {
        static std::unique_ptr<nixlPosixQueue> createUringQueue(int num_entries, nixl_xfer_op_t operation) {
            // Initialize io_uring parameters with basic configuration
            // Start with basic parameters, no special flags
            // We can add optimizations like SQPOLL later
            struct io_uring_params params = {};
            return std::make_unique<class UringQueue>(num_entries, params, operation);
        }

        static bool isUringAvailable() {
            return true;
        }
    };

    template <typename Mode>
    struct funcImpl<Mode, std::enable_if_t<std::is_same<Mode, uringDisabled>::value>> {
        static std::unique_ptr<nixlPosixQueue> createUringQueue(int num_entries, nixl_xfer_op_t operation) {
            (void)num_entries;
            (void)operation;
            throw nixlPosixBackendReqH::exception("Attempting to create io_uring queue when support is not compiled in",
                                                  NIXL_ERR_NOT_SUPPORTED);
        }

        static bool isUringAvailable() {
            return false;
        }
    };
}

// Public functions implementation
std::unique_ptr<nixlPosixQueue>
QueueFactory::createPosixAioQueue(int num_entries, nixl_xfer_op_t operation) {
    return std::make_unique<aioQueue>(num_entries, operation);
}

std::unique_ptr<nixlPosixQueue> QueueFactory::createUringQueue(int num_entries, nixl_xfer_op_t operation) {
    return funcImpl<uringMode>::createUringQueue(num_entries, operation);
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

bool QueueFactory::isUringAvailable() {
    return funcImpl<uringMode>::isUringAvailable();
}

bool
QueueFactory::isLinuxAioAvailable() {
#ifdef HAVE_LINUXAIO
    return true;
#else
    return false;
#endif
}
