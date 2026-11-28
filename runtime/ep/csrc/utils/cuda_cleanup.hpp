// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include "nixl_log.h"

namespace nixl::runtime::ep {

inline void warn_cuda_cleanup(const char* operation, cudaError_t status) noexcept {
    if (status != cudaSuccess) {
        // PyTorch's allocator checks this thread's last CUDA error when freeing tensors.
        (void)cudaGetLastError();
        NIXL_WARN << "NIXL EP cleanup (" << operation << ") failed: " << cudaGetErrorString(status);
    }
}

}  // namespace nixl::runtime::ep
