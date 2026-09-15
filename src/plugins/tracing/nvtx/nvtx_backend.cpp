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

#include "nvtx_backend.h"

#include "common/nixl_log.h"
#include "nvtx_trace_backend.h"

namespace nixl::trace {

std::unique_ptr<TraceBackend>
createNvtxBackend(const nixlTraceBackendInitParams &init_params) {
    try {
        return std::make_unique<nvtx_internal::NvtxTraceBackend>(init_params.agentName);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create NVTX trace backend: " << e.what();
        return nullptr;
    }
}

} // namespace nixl::trace
