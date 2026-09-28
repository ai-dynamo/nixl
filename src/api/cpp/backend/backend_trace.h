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
#ifndef NIXL_SRC_API_CPP_BACKEND_BACKEND_TRACE_H
#define NIXL_SRC_API_CPP_BACKEND_BACKEND_TRACE_H

#include <cstdint>
#include <span>
#include <string_view>

#include "common/nixl_time.h"

enum class nixl_trace_stage_t : uint8_t {
    SUBMIT,
    WIRE_SUBMITTED,
    WIRE_COMPLETED,
    NOTIF_SENT,
    NOTIF_RECEIVED,
    REMOTE_OBSERVED,
    STAGE,
};

struct nixlBackendTraceAttr {
    std::string_view key;
    std::string_view value;
};

class nixlBackendTraceSink {
public:
    virtual ~nixlBackendTraceSink() = default;

    virtual void
    recordPhase(nixl_trace_stage_t stage,
                std::string_view label,
                nixlTime::us_t timestamp,
                std::span<const nixlBackendTraceAttr> attrs) = 0;
};

#endif // NIXL_SRC_API_CPP_BACKEND_BACKEND_TRACE_H
