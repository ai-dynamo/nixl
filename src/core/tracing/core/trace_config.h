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
#ifndef NIXL_SRC_CORE_TRACING_CORE_TRACE_CONFIG_H
#define NIXL_SRC_CORE_TRACING_CORE_TRACE_CONFIG_H

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tracing/api/trace_backend.h"

namespace nixl::trace {

inline constexpr std::string_view traceSampleRatioVar = "NIXL_TRACE_SAMPLE_RATIO";
inline constexpr std::string_view otelTracesSampleRatioVar = "OTEL_TRACES_SAMPLE_RATIO";
inline constexpr double defaultSampleRatio = 1.0;

/**
 * @brief Resolve the sampling ratio: traceSampleRatioVar when set, else
 *        otelTracesSampleRatioVar, which is Dynamo's convention rather than an
 *        OpenTelemetry one (the SDK spec defines OTEL_TRACES_SAMPLER and
 *        OTEL_TRACES_SAMPLER_ARG) but carries the same head-sampling meaning,
 *        so a NIXL running inside Dynamo inherits it. When neither is set,
 *        every request is sampled, as in Dynamo. Setting
 *        traceSampleRatioVar empty is an explicit "off" that beats the
 *        fallback, matching how NIXL_TRACE_BACKENDS is treated.
 * @return The resolved ratio, or defaultSampleRatio (every request) when
 *         neither is set or otelTracesSampleRatioVar is empty or unusable.
 * @throws std::invalid_argument when traceSampleRatioVar holds anything other
 *         than 0 or a number in [2^-56, 1] — OpenTelemetry's range, 2^-56 being
 *         one trace in every value 56 bits can express — so a caller who asked
 *         for sampling is not silently given none. A bad value in the shared
 *         OpenTelemetry variable only warns: NIXL does not own it and must not
 *         fail construction on it.
 */
[[nodiscard]] double
resolveTraceSampleRatio();

[[nodiscard]] std::vector<std::unique_ptr<TraceBackend>>
loadTraceBackends(const std::vector<std::string> &names, std::string_view agent_name);

} // namespace nixl::trace

#endif
