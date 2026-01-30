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

#ifndef NIXL_SRC_UTILS_UCX_MEM_LIST_H
#define NIXL_SRC_UTILS_UCX_MEM_LIST_H

#include "backend/backend_aux.h"

class nixlUcxWorker;

namespace nixl::ucx {
[[nodiscard]] void *
createMemList(const nixl_remote_meta_dlist_t &, size_t, nixlUcxWorker &);

[[nodiscard]] void *
createMemList(const nixl_meta_dlist_t &, const nixlUcxWorker &);

void
releaseMemList(void *mvh) noexcept;
} // namespace nixl::ucx

#endif
