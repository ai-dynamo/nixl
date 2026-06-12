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
#include <memory>
#include <optional>

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

    void
    release_physical_preserve_va();

    void
    remap_physical_at_preserved_va();

    [[nodiscard]] bool
    can_preserve_va() const noexcept {
        return !is_cuda_malloc_ && ptr_;
    }

    [[nodiscard]] bool
    is_mapped() const noexcept {
        return is_cuda_malloc_ || vmm_mapped_;
    }

    [[nodiscard]] CUmemFabricHandle
    export_fabric_handle() const;

    [[nodiscard]] static std::unique_ptr<vmm_region>
    import_fabric_handle(const CUmemFabricHandle &fabric_handle, size_t size);

private:
    vmm_region() = default;

    struct cuda_alloc_ctx;

    [[nodiscard]] static const cuda_alloc_ctx &
    get_cuda_alloc_ctx();

    void
    map_handle(std::optional<CUdeviceptr> fixed_addr);

    void
    reserve_and_map(std::optional<CUdeviceptr> fixed_addr);

    void
    release() noexcept;

    CUdeviceptr ptr_ = 0;
    size_t size_ = 0;
    CUmemGenericAllocationHandle handle_ = 0;
    bool is_cuda_malloc_ = false;
    bool vmm_mapped_ = false;
};

} // namespace nixl_ep
