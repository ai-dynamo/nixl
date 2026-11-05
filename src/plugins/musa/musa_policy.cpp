// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "musa_policy.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace nixl::musa {
nixl::ucx::MemoryRegistrationPolicy
makeMemoryPolicy(std::shared_ptr<const Runtime> runtime) {
    if (!runtime) {
        throw std::invalid_argument("MUSA_UCX requires a runtime");
    }
    const int count = runtime->deviceCount();
    if (count <= 0) {
        throw std::runtime_error("MUSA_UCX requires at least one visible MUSA device");
    }
    return {"musa",
            [runtime = std::move(runtime), device_count = static_cast<uint64_t>(count)](
                const nixlBlobDesc &desc, nixl_mem_t type) {
                if (type == DRAM_SEG) {
                    return NIXL_SUCCESS;
                }
                if (desc.devId >= device_count) {
                    return NIXL_ERR_INVALID_PARAM;
                }
                const auto attributes = runtime->pointerAttributes(desc.addr);
                if (!attributes || !attributes->device_memory || attributes->device < 0 ||
                    static_cast<uint64_t>(attributes->device) != desc.devId) {
                    return NIXL_ERR_INVALID_PARAM;
                }
                return NIXL_SUCCESS;
            }};
}
} // namespace nixl::musa
