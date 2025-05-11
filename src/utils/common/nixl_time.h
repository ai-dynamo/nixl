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
#ifndef _NIXL_TIME_H
#define _NIXL_TIME_H

#include <chrono>

namespace nixlTime {

    using namespace std::chrono;

    using ns_t = uint64_t;
    using us_t = uint64_t;
    using ms_t = uint64_t;
    using sec_t = uint64_t;

    static inline ns_t getNs() {
        return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    }

    static inline us_t getUs() {
        return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    }

    static inline ms_t getMs() {
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    }

    static inline sec_t getSec() {
        return duration_cast<seconds>(steady_clock::now().time_since_epoch()).count();
    }

}

#endif
