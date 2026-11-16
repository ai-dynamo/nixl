/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_CUDA_DEVICE_GUARD_H
#define NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_CUDA_DEVICE_GUARD_H

#include <cuda_runtime.h>
#include <cstdint>

namespace nixl::doca {
// Establish the engine's runtime context even when a new thread already reports
// the same default device. Restore the caller's runtime device on scope exit.
class cudaDeviceGuard {
public:
    explicit cudaDeviceGuard(uint32_t device) : device_(static_cast<int>(device)) {
        status_ = cudaGetDevice(&previous_device_);
        if (status_ == cudaSuccess) {
            status_ = cudaSetDevice(device_);
            restore_ = status_ == cudaSuccess && previous_device_ != device_;
        }
    }

    ~cudaDeviceGuard() {
        if (restore_) {
            cudaSetDevice(previous_device_);
        }
    }

    cudaDeviceGuard(const cudaDeviceGuard &) = delete;
    cudaDeviceGuard &
    operator=(const cudaDeviceGuard &) = delete;

    [[nodiscard]] cudaError_t
    status() const {
        return status_;
    }

private:
    int device_;
    int previous_device_ = 0;
    bool restore_ = false;
    cudaError_t status_ = cudaSuccess;
};
} // namespace nixl::doca

#endif
