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

#include "tracing/api/tracing.h"

#include "common/configuration.h"
#include "common/nixl_log.h"
#include "common/str_util.h"

nixl::trace::TracerPtr
nixl::trace::makeAgentTracer(const std::string &name) {
    const auto backends = nixl::config::getValueOptional<std::string>("NIXL_TRACE_BACKENDS");
    const bool requested = backends ? !nixl::str::splitStrippedSet(*backends).empty() :
                                      nixl::config::checkExistence("NVTX_INJECTION64_PATH");
    if (requested) {
        NIXL_WARN << "nixl::trace: agent '" << name
                  << "' was asked to trace (NIXL_TRACE_BACKENDS or Nsight Systems), but this "
                     "NIXL is built with -Dwith_trace=false, so nothing is traced";
    }
    return {};
}
