/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_CORE_TRACING_LOG_ONCE_H
#define NIXL_SRC_CORE_TRACING_LOG_ONCE_H

#include <atomic>

namespace nixl::trace {

template<typename logFn>
void
logOnce(std::atomic<bool> &logged, logFn &&log) noexcept {
    if (logged.load(std::memory_order_relaxed) ||
        logged.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    try {
        log();
    }
    catch (...) {
    }
}

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_LOG_ONCE_H
