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

#pragma once

#include <cuda.h>
#include <cstddef>
#include <cstdint>

namespace nixl_ep {

class vmm_region {
public:
    explicit vmm_region(size_t size);

    ~vmm_region();

    vmm_region(const vmm_region &) = delete;
    vmm_region &
    operator=(const vmm_region &) = delete;
    vmm_region(vmm_region &&) = delete;
    vmm_region &
    operator=(vmm_region &&) = delete;

    [[nodiscard]] void *
    ptr() const noexcept {
        return reinterpret_cast<void *>(static_cast<std::uintptr_t>(ptr_));
    }

private:
    static void
    warn_cu_api(CUresult status, const char *context, const char *operation) noexcept;

    void
    release() noexcept;

    CUdeviceptr ptr_ = 0;
    size_t size_ = 0;
    CUmemGenericAllocationHandle handle_ = 0;
    bool is_cuda_malloc_ = false;
    bool vmm_addr_reserved_ = false;
    bool vmm_mapped_ = false;
};

} // namespace nixl_ep
