// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "musa_runtime.h"
#include "../ucx/memory_policy.h"

namespace nixl::musa {
class MemoryPolicy final : public nixl::ucx::MemoryRegistrationPolicy {
public:
    explicit MemoryPolicy(std::shared_ptr<const Runtime> runtime);

    std::string_view
    deviceMemoryType() const noexcept override {
        return "musa";
    }

    nixl_status_t
    validateBeforeMap(const nixlBlobDesc &desc, nixl_mem_t type) const override;

private:
    const std::shared_ptr<const Runtime> runtime_;
    uint64_t device_count_;
};
} // namespace nixl::musa
