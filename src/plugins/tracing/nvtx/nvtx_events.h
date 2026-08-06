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
#ifndef NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_EVENTS_H
#define NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_EVENTS_H

#include <string>
#include <string_view>
#include <vector>

#include <nvtx3/nvToolsExt.h>

#include "tracing/trace.h"

namespace nixl::trace::nvtx_internal {

/** @brief Register fixed `nixl::*` span/mark names on @p domain (hot-path labels). */
[[nodiscard]] std::vector<nvtxStringHandle_t>
registerSpanNames(nvtxDomainHandle_t domain);

/** @brief Build NVTX event attributes for a span/mark name and operation kind.
 *         Uses a pre-registered string handle; an unknown name is registered on
 *         @p domain on demand, so the event always carries a registered label. */
[[nodiscard]] nvtxEventAttributes_t
eventForName(std::string_view name,
             Kind kind,
             nvtxDomainHandle_t domain,
             const std::vector<nvtxStringHandle_t> &registered_handles);

} // namespace nixl::trace::nvtx_internal

#endif // NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_EVENTS_H
