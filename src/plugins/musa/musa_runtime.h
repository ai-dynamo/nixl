// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_MUSA_MUSA_RUNTIME_H
#define NIXL_SRC_PLUGINS_MUSA_MUSA_RUNTIME_H

#include <cstdint>
#include <memory>
#include <optional>

namespace nixl::musa {

struct PointerAttributes {
    bool device_memory;
    int device;
};

class Runtime {
public:
    virtual ~Runtime() = default;
    [[nodiscard]] virtual int
    deviceCount() const = 0;
    [[nodiscard]] virtual std::optional<PointerAttributes>
    pointerAttributes(uintptr_t address) const = 0;
};

[[nodiscard]] std::shared_ptr<const Runtime>
makeRuntime();

} // namespace nixl::musa

#endif
