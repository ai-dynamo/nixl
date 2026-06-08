/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "worker/nixl/nixl_mem_region.h"

#include <utility>

NixlMemRegion::NixlMemRegion(nixlAgent &agent,
                             nixlBackendH *backend,
                             nixl_mem_t seg_type,
                             std::vector<xferBenchIOV> iovs,
                             std::function<void(xferBenchIOV &)> cleanup)
    : agent_(agent),
      backend_(backend),
      seg_type_(seg_type),
      iovs_(std::move(iovs)),
      cleanup_(std::move(cleanup)) {
    if (backend_) {
        cached_opt_args_.backends.push_back(backend_);
    }
}

NixlMemRegion::~NixlMemRegion() {
    release();
}

NixlMemRegion::NixlMemRegion(NixlMemRegion &&o) noexcept
    : agent_(o.agent_),
      backend_(o.backend_),
      seg_type_(o.seg_type_),
      iovs_(std::move(o.iovs_)),
      cleanup_(std::move(o.cleanup_)),
      cached_opt_args_(std::move(o.cached_opt_args_)) {
    // Empty iovs_ is the "nothing to release" sentinel; guarantee the
    // moved-from region won't deregister on destruction.
    o.iovs_.clear();
}

void
NixlMemRegion::release() {
    if (iovs_.empty()) {
        return;
    }
    const nixl_reg_dlist_t desc_list = iovListToNixlRegDlist(iovs_, seg_type_);
    CHECK_NIXL_ERROR(agent_.deregisterMem(desc_list, &cached_opt_args_), "deregisterMem failed");
    for (auto &iov : iovs_) {
        if (cleanup_) {
            cleanup_(iov);
        }
    }
    iovs_.clear();
}
