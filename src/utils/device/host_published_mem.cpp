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
#include "device/host_published_mem.h"

#include <cstring>
#include <utility>

#include "common/nixl_log.h"
#include "device/device_ops.h"
#include "nixl_log.h"

namespace nixl {
namespace {

    class mappedHostPublishedMem final : public hostPublishedDeviceMem {
    public:
        mappedHostPublishedMem(mappedHostMem mem, size_t size) noexcept
            : hostPublishedDeviceMem(mem.devicePointer(), size, backing::mapped_host),
              mem_(std::move(mem)) {}

    private:
        nixl_status_t
        doPublish(size_t offset, uint64_t value) noexcept override {
            __atomic_store_n(
                mem_.hostPointer<uint64_t>() + offset / sizeof(uint64_t), value, __ATOMIC_RELEASE);
            return NIXL_SUCCESS;
        }

        mappedHostMem mem_;
    };

} // namespace

nixl_status_t
deviceOps::allocHostPublishedMem(size_t size,
                                 std::unique_ptr<hostPublishedDeviceMem> &out) noexcept {
    if (size == 0) {
        NIXL_ERROR << "Host-published allocation requires nonzero size";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (doAllocHostPublishedMem(size, out) == NIXL_SUCCESS) {
        return NIXL_SUCCESS;
    }

    mappedHostMem mem;
    const nixl_status_t status = allocMappedHostMem(size, mem);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    std::memset(mem.hostPointer(), 0, size);
    out = std::make_unique<mappedHostPublishedMem>(std::move(mem), size);
    return NIXL_SUCCESS;
}

} // namespace nixl
