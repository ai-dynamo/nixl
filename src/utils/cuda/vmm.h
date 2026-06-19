/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>

namespace nixl::cuda {

bool is_fabric_vmm_supported(int device_id);

class fabric_vmm_region {
public:
    explicit fabric_vmm_region(std::size_t size,
                               int device_id = -1,
                               bool require_fabric = false,
                               std::uintptr_t address_hint = 0);
    ~fabric_vmm_region();

    fabric_vmm_region(const fabric_vmm_region&) = delete;
    fabric_vmm_region& operator=(const fabric_vmm_region&) = delete;
    fabric_vmm_region(fabric_vmm_region&&) = delete;
    fabric_vmm_region& operator=(fabric_vmm_region&&) = delete;

    [[nodiscard]] void* ptr() const noexcept {
        return reinterpret_cast<void*>(static_cast<std::uintptr_t>(ptr_));
    }

    [[nodiscard]] std::size_t requested_size() const noexcept {
        return requested_size_;
    }

    [[nodiscard]] std::size_t mapped_size() const noexcept {
        return mapped_size_;
    }

    [[nodiscard]] bool is_fabric() const noexcept {
        return is_fabric_;
    }

    void set_current_context() const;
    void release() noexcept;

private:
    CUdeviceptr ptr_ = 0;
    std::size_t requested_size_ = 0;
    std::size_t mapped_size_ = 0;
    CUmemGenericAllocationHandle handle_ = 0;
    CUdevice device_ = 0;
    CUcontext context_ = nullptr;
    bool is_fabric_ = false;
    bool mapped_ = false;
    bool primary_context_retained_ = false;
};

} // namespace nixl::cuda
