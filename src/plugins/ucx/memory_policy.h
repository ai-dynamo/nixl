// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_UCX_MEMORY_POLICY_H
#define NIXL_SRC_PLUGINS_UCX_MEMORY_POLICY_H

#include <nixl_descriptors.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace nixl::ucx {

[[nodiscard]] inline bool
validMemoryRange(uintptr_t address, size_t length) noexcept {
    return address != 0 && length != 0 && length <= std::numeric_limits<uintptr_t>::max() - address;
}

[[nodiscard]] inline nixl_status_t
validateMemoryDescriptor(const nixlBlobDesc &desc, nixl_mem_t type) noexcept {
    if (type != DRAM_SEG && type != VRAM_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    return validMemoryRange(desc.addr, desc.len) ? NIXL_SUCCESS : NIXL_ERR_INVALID_PARAM;
}

// An empty policy preserves stock UCX behavior. Named policies enforce the common
// descriptor checks before an optional provider-runtime validation callback.
struct MemoryRegistrationPolicy {
    std::string device_memory_type;
    std::function<nixl_status_t(const nixlBlobDesc &, nixl_mem_t)> validate;

    [[nodiscard]] bool
    isStrict() const noexcept {
        return !device_memory_type.empty();
    }

    [[nodiscard]] nixl_status_t
    validateBeforeMap(const nixlBlobDesc &desc, nixl_mem_t type) const {
        if (!isStrict()) {
            return NIXL_SUCCESS;
        }
        const auto status = validateMemoryDescriptor(desc, type);
        return status == NIXL_SUCCESS && validate ? validate(desc, type) : status;
    }
};

// Enum values are resolved from the linked provider's names, never invented for a vendor.
[[nodiscard]] inline std::optional<unsigned>
findDeviceMemoryType(std::span<const char *const> names,
                     std::string_view requested,
                     uint64_t supported,
                     unsigned host_type) noexcept {
    if (requested.empty() || requested == "unknown") {
        return std::nullopt;
    }
    for (size_t i = 0; i < names.size() && i < std::numeric_limits<uint64_t>::digits; ++i) {
        if (i != host_type && names[i] && requested == names[i] &&
            (supported & (uint64_t{1} << i))) {
            return static_cast<unsigned>(i);
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline bool
registeredMemoryTypeMatches(nixl_mem_t requested,
                            unsigned actual,
                            unsigned host_type,
                            unsigned device_type) noexcept {
    return (requested == DRAM_SEG && actual == host_type) ||
        (requested == VRAM_SEG && actual == device_type);
}

} // namespace nixl::ucx

#endif
