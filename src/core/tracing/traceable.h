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
#ifndef NIXL_SRC_CORE_TRACING_TRACEABLE_H
#define NIXL_SRC_CORE_TRACING_TRACEABLE_H

#include <cstdint>

#include "backend/backend_aux.h"
#include "tracing/trace_context.h"

// NOTE: This file contains only external tracing interface, stuff used
// directly by NIXL modules and nothing more

namespace nixl::trace {

enum class Event : std::uint8_t {
    CREATED,
    POSTED,
    SUBMITTED,
    WIRE_SUBMITTED,
    WIRE_COMPLETED,
    NOTIF_SENT,
    NOTIF_RECEIVED,
    COMPLETED,
    FAILED,
    RELEASED,
    COUNT,
};

class Tracer;

class Traceable {
public:
    explicit Traceable(const Tracer *tracer = nullptr) noexcept
        : tracer_(tracer),
          id_(generateRequestId(tracer)) {}

    static Traceable *
    from(const nixlBackendOptionalArgs *opt_args) noexcept {
        return ((opt_args != nullptr) && (opt_args->tracing != nullptr)) ? opt_args->tracing :
                                                                           &none();
    }

    bool
    isEnabled() const noexcept {
        return tracer_ != nullptr;
    }

    void
    trace(Event event) noexcept {
        if (isEnabled()) {
            record(event);
        }
    }

private:
    static Traceable &
    none() noexcept {
        static Traceable inert;
        return inert;
    }

    void
    record(Event event) noexcept;

    const Tracer *tracer_;
    std::uint64_t id_;
};

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_TRACEABLE_H
