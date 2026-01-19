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

#ifndef NIXL_SRC_UTILS_UCX_DEVICE_MEM_LIST_H
#define NIXL_SRC_UTILS_UCX_DEVICE_MEM_LIST_H

#include <memory>
#include <vector>

extern "C" {
#include <ucp/api/ucp_def.h>
}

class nixlUcxMem;
class nixlUcxWorker;
class nixlUcxEp;

namespace nixl::ucx {
class rkey;

struct remoteMem {
    remoteMem(const nixlUcxEp &ep, uint64_t addr, const rkey &rkey)
        : ep_(ep),
          addr_(addr),
          rkey_(rkey) {}

    const nixlUcxEp &ep_;
    uint64_t addr_;
    const rkey &rkey_;
};

class memList {
public:
    memList(const std::vector<nixlUcxMem> &, const nixlUcxWorker &);
    memList(const std::vector<std::unique_ptr<remoteMem>> &, nixlUcxWorker &);

    [[nodiscard]] void *
    get() const noexcept {
        return memList_.get();
    }

private:
    std::unique_ptr<void, void (*)(void *)> memList_;
};
} // namespace nixl::ucx

#endif
