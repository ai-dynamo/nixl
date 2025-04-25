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
#ifndef SYNC_H
#define SYNC_H
#include "common/util.h"
#include "nixl_params.h"
#include <mutex>

class nixlLock {
    public:
        nixlLock(const nixl_thread_sync_t sync_mode): syncMode(sync_mode)
        {}

        void lock() {
            if (syncMode == nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT) {
                m.lock();
            }
        }

        void unlock() {
            if (syncMode == nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT) {
                m.unlock();
            }
        }

    private:
        nixl_thread_sync_t syncMode;
        std::mutex m;
};

#define NIXL_LOCK_GUARD(lock) const std::lock_guard<nixlLock> UNIQUE_NAME(lock_guard) (lock)

#endif /* SYNC_H */
