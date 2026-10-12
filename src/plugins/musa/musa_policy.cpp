// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "musa_policy.h"

#include <stdexcept>
#include <utility>

namespace nixl::musa {
MemoryPolicy::MemoryPolicy(std::shared_ptr<const Runtime> runtime) : runtime_(std::move(runtime)) {
    if (!runtime_) {
        throw std::invalid_argument("MUSA_UCX requires a runtime");
    }
    const int count = runtime_->deviceCount();
    if (count <= 0) {
        throw std::runtime_error("MUSA_UCX requires at least one visible MUSA device");
    }
    deviceCount_ = static_cast<uint64_t>(count);
}

nixl_status_t
MemoryPolicy::validateBeforeMap(const nixlBlobDesc &desc, nixl_mem_t type) const {
    const auto status = nixl::ucx::validateMemoryDescriptor(desc, type);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    if (type == DRAM_SEG) {
        return NIXL_SUCCESS;
    }
    if (desc.devId >= deviceCount_) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto attributes = runtime_->pointerAttributes(desc.addr);
    if (!attributes || !attributes->device_memory || attributes->device < 0 ||
        static_cast<uint64_t>(attributes->device) != desc.devId) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return NIXL_SUCCESS;
}
} // namespace nixl::musa
