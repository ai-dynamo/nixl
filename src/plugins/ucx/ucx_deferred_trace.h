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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_DEFERRED_TRACE_H
#define NIXL_SRC_PLUGINS_UCX_UCX_DEFERRED_TRACE_H

#include "nixl_types.h"
#include "tracing/backend_trace.h"

namespace nixl::ucx {

struct DeferredWireCompleted {
    nixlBackendPostTrace trace;
    nixl_xfer_op_t operation = NIXL_WRITE;
    nixlTime::us_t inflightSince = 0;
    nixlTime::us_t phaseAt = 0;
    nixlTime::us_t completedAt = 0;
};

void
deferWireCompleted(const DeferredWireCompleted &completion) noexcept;

void
recordDeferredTraces() noexcept;

} // namespace nixl::ucx

#endif // NIXL_SRC_PLUGINS_UCX_UCX_DEFERRED_TRACE_H
