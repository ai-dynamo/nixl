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

#include <utility>

#include "tracing/core/trace.h"
#include "tracing/core/trace_sink.h"

namespace {

thread_local const nixl::trace::TraceIdentity *current_identity = nullptr;

} // namespace

const nixl::trace::TraceIdentity *
nixl::trace::CompositeTracer::swapCurrentIdentity(const TraceIdentity *identity) noexcept {
    return std::exchange(current_identity, identity);
}

const nixl::trace::TraceIdentity *
nixl::trace::TracerPhaseSink::currentIdentity() const noexcept {
    return current_identity;
}
