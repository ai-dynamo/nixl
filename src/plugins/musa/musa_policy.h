// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_MUSA_MUSA_POLICY_H
#define NIXL_SRC_PLUGINS_MUSA_MUSA_POLICY_H

#include "musa_runtime.h"
#include "../ucx/memory_policy.h"

#include <memory>

namespace nixl::musa {
// The callback owns the runtime for as long as any UCX engine uses the policy.
[[nodiscard]] nixl::ucx::MemoryRegistrationPolicy
makeMemoryPolicy(std::shared_ptr<const Runtime> runtime);
} // namespace nixl::musa

#endif
