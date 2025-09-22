/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_UTILS_UCX_GPU_XFER_REQ_H_H
#define NIXL_SRC_UTILS_UCX_GPU_XFER_REQ_H_H

#include <vector>

#include "nixl_types.h"

class nixlUcxEp;
class nixlUcxMem;

namespace nixl::ucx {
class rkey;

nixlGpuXferReqH
createGpuXferReq(const nixlUcxEp &,
                 const std::vector<nixlUcxMem> &,
                 const std::vector<const nixl::ucx::rkey *> &);

void
releaseGpuXferReq(nixlGpuXferReqH gpu_req) noexcept;
} // namespace nixl::ucx

#endif
