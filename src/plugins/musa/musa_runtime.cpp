// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "musa_runtime.h"

#include <musa_runtime_api.h>
#include <stdexcept>
#include <string>

namespace nixl::musa {
namespace {

    void
    check(musaError_t result, const char *operation) {
        if (result != musaSuccess) {
            throw std::runtime_error(std::string(operation) + ": " + musaGetErrorString(result) +
                                     " (" + std::to_string(static_cast<int>(result)) + ")");
        }
    }

    class SdkRuntime final : public Runtime {
    public:
        int
        deviceCount() const override {
            int count = 0;
            check(musaGetDeviceCount(&count), "musaGetDeviceCount");
            return count;
        }

        std::optional<PointerAttributes>
        pointerAttributes(uintptr_t address) const override {
            musaPointerAttributes attributes{};
            const auto status =
                musaPointerGetAttributes(&attributes, reinterpret_cast<const void *>(address));
            if (status == musaErrorInvalidValue) {
                return std::nullopt;
            }
            check(status, "musaPointerGetAttributes");
            return PointerAttributes{attributes.type == musaMemoryTypeDevice, attributes.device};
        }
    };
} // namespace

std::shared_ptr<const Runtime>
makeRuntime() {
    return std::make_shared<SdkRuntime>();
}
} // namespace nixl::musa
