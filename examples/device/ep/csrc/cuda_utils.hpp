/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

namespace nixl_ep {

/** Log a non-fatal warning if a CUDA runtime API call failed (e.g. during teardown). */
inline void
warn_cuda_api(cudaError_t status, const char *context, const char *operation) noexcept {
    if (status != cudaSuccess) {
        std::cerr << "WARNING: " << context << " failed to " << operation << ": "
                  << cudaGetErrorString(status) << '\n';
    }
}

/** Log a non-fatal warning if a CUDA driver API call failed (e.g. during teardown). */
inline void
warn_cu_api(CUresult status, const char *context, const char *operation) noexcept {
    if (status != CUDA_SUCCESS) {
        const char *msg = nullptr;
        if (cuGetErrorString(status, &msg) != CUDA_SUCCESS || msg == nullptr) {
            msg = "unknown CUDA driver error";
        }
        std::cerr << "WARNING: " << context << " failed to " << operation << ": " << msg << '\n';
    }
}

} // namespace nixl_ep
