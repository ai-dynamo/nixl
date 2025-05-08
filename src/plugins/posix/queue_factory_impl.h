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

#ifndef QUEUE_FACTORY_IMPL_H
#define QUEUE_FACTORY_IMPL_H

#include "async_queue.h"
#include <memory>

// Backend-specific includes
#ifdef HAVE_LIBAIO
#include "aio_queue.h"
#endif

#ifdef HAVE_LIBURING
#include "uring_queue.h"
#endif

class QueueFactory {
    public:
        // Backend-specific queue creation
#ifdef HAVE_LIBAIO
        static std::unique_ptr<nixlPosixQueue> createAioQueue(int num_entries, bool is_read) {
            return std::make_unique<aioQueue>(num_entries, is_read);
        }
#else
        static std::unique_ptr<nixlPosixQueue> createAioQueue(int num_entries, bool is_read) {
            (void)num_entries; // Avoid unused parameter warning
            (void)is_read;
            return nullptr;
        }
#endif // HAVE_LIBAIO

#ifdef HAVE_LIBURING
        static std::unique_ptr<nixlPosixQueue> createUringQueue(int num_entries, bool is_read, const io_uring_params* params) {
            if (!params) {
                return nullptr;
            }
            return std::make_unique<UringQueue>(num_entries, *params, is_read);
        }
#else
        static std::unique_ptr<nixlPosixQueue> createUringQueue(int num_entries, bool is_read, const void* params) {
            (void)num_entries; // Avoid unused parameter warning
            (void)is_read;
            (void)params;
            return nullptr;
        }
#endif // HAVE_LIBURING

        // Backend availability checks
        static bool isAioAvailable() {
#ifdef HAVE_LIBAIO
            return createAioQueue(1, true) != nullptr;
#else
            return false;
#endif // HAVE_LIBAIO
        }

        static bool isUringAvailable() {
#ifdef HAVE_LIBURING
            io_uring_params params = {};
            params.cq_entries = 1;  // Match the sq_entries (1) to avoid 2x overhead
            return createUringQueue(1, true, &params) != nullptr;
#else
            return false;
#endif // HAVE_LIBURING
        }

    private:
        // Make constructor private to prevent instantiation
        QueueFactory() = delete;
};

#endif // QUEUE_FACTORY_IMPL_H
