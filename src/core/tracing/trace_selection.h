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
#ifndef NIXL_SRC_CORE_TRACING_TRACE_SELECTION_H
#define NIXL_SRC_CORE_TRACING_TRACE_SELECTION_H

/*
 * Trace backend-selection policy (agent-wiring layer).
 *
 * This is deliberately separate from the backend-agnostic facade core
 * (trace.h / tracer.cpp): it encodes the NVTX-specific "auto-enable under
 * Nsight Systems" rule, so nsys detection never leaks into makeTracer().
 */

#include <optional>
#include <string>
#include <vector>

namespace nixl::trace {

/**
 * @brief True when the process is running under Nsight Systems.
 *
 * nsys injects NVTX_INJECTION64_PATH into the environment of the process it
 * profiles (it points at nsys's NVTX injection library). Its presence therefore
 * means "this process is running under nsys" -- not merely that nsys is
 * installed on the machine.
 */
[[nodiscard]] bool
runningUnderNsys();

/**
 * @brief Decide which trace backends to activate.
 *
 * Running under nsys auto-enables NVTX *in addition to* any explicitly requested
 * backends -- it never suppresses them. The only hard override is a set-but-empty
 * value, which forces tracing off:
 *   - @p explicit_spec set to a non-empty list -> that list, plus "nvtx" when
 *     @p under_nsys (deduplicated).
 *   - @p explicit_spec set but empty -> {} (explicit off; beats nsys auto-enable).
 *   - @p explicit_spec unset -> {"nvtx"} when @p under_nsys, else {}.
 *
 * @param explicit_spec Raw NIXL_TRACE_BACKENDS value; std::nullopt when unset.
 * @param under_nsys    Whether the process runs under nsys (see runningUnderNsys).
 */
[[nodiscard]] std::vector<std::string>
resolveTraceBackends(const std::optional<std::string> &explicit_spec, bool under_nsys);

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_TRACE_SELECTION_H
