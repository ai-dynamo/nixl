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
#ifndef NIXL_SRC_UTILS_DEVICE_DEVICE_OPS_H
#define NIXL_SRC_UTILS_DEVICE_DEVICE_OPS_H

#include <cstddef>
#include <memory>
#include <utility>

#include <nixl_types.h>

namespace nixl {

class deviceOps;
class mappedHostMem;

struct mappedHostMemDeleter {
    deviceOps *ops = nullptr;

    void
    operator()(void *ptr) const noexcept;
};

struct deviceMemDeleter {
    deviceOps *ops = nullptr;

    void
    operator()(void *ptr) const noexcept;
};

// Owns device storage; typed access requires a cast of get().
using deviceMem = std::unique_ptr<void, deviceMemDeleter>;

#define NIXL_DEVICE_OPS_EXPORT __attribute__((visibility("default")))

/**
 * Device memory-ops interface. All host-side interaction with the GPU memory
 * runtime goes through this class so that no other host code needs a
 * cuda_runtime.h include. CUDA is the current platform implementation; HIP
 * can provide another. Allocations are returned as owning RAII handles.
 * Raw alloc/free remain protected; release() and construction with the
 * original deleter cross the C-handle boundary. The class is not
 * internally synchronized. Transfers are issued on the default stream and are
 * ordered there, but not all are complete on return; synchronize() is the
 * barrier. Active device state is thread-local: copies, memset, and
 * synchronize use the caller's current device; freeing works from any.
 * Allocation hooks modify their output pointers only on success.
 * Zero-size allocations, copies and memset return NIXL_ERR_INVALID_PARAM,
 * even without a device runtime, and leave outputs unchanged. Use
 * getActiveDevice() to check runtime availability.
 */
class deviceOps {
public:
    enum class copyDirection { HostToDevice, DeviceToHost };

    virtual ~deviceOps() = default;

    [[nodiscard]] nixl_status_t
    allocDeviceMem(size_t size, deviceMem &out) noexcept;

    /**
     * Allocate pinned host memory that is mapped into the device address
     * space; the handle exposes both the host pointer and its
     * device-visible alias.
     */
    [[nodiscard]] nixl_status_t
    allocMappedHostMem(size_t size, mappedHostMem &out) noexcept;

    /** H2D: src is reusable on return; D2H: dst holds the data on return. */
    [[nodiscard]] virtual nixl_status_t
    copy(void *dst, const void *src, size_t size, copyDirection direction) noexcept = 0;

    /** Enqueued, not complete on return; ordered against later default-stream work. */
    [[nodiscard]] virtual nixl_status_t
    memsetDeviceMem(void *ptr, int value, size_t size) noexcept = 0;

    /** Block until all outstanding work on the active device completes. */
    [[nodiscard]] virtual nixl_status_t
    synchronize() noexcept = 0;

    [[nodiscard]] virtual nixl_status_t
    getActiveDevice(int &device_id) noexcept = 0;

    [[nodiscard]] virtual nixl_status_t
    setActiveDevice(int device_id) noexcept = 0;

protected:
    [[nodiscard]] virtual nixl_status_t
    doAllocDeviceMem(void *&ptr, size_t size) noexcept = 0;

    virtual void
    doFreeDeviceMem(void *ptr) noexcept = 0;

    [[nodiscard]] virtual nixl_status_t
    doAllocMappedHostMem(void *&host_ptr, void *&dev_ptr, size_t size) noexcept = 0;

    virtual void
    doFreeMappedHostMem(void *host_ptr) noexcept = 0;

private:
    friend struct deviceMemDeleter;
    friend struct mappedHostMemDeleter;
};

inline void
deviceMemDeleter::operator()(void *ptr) const noexcept {
    ops->doFreeDeviceMem(ptr);
}

inline void
mappedHostMemDeleter::operator()(void *ptr) const noexcept {
    ops->doFreeMappedHostMem(ptr);
}

/** Owns pinned host memory and exposes its non-owning device alias. */
class mappedHostMem {
public:
    mappedHostMem() = default;

    template<typename T = void>
    [[nodiscard]] T *
    hostPointer() const noexcept {
        return static_cast<T *>(hostPtr_.get());
    }

    template<typename T = void>
    [[nodiscard]] T *
    devicePointer() const noexcept {
        return hostPtr_ ? static_cast<T *>(devPtr_) : nullptr;
    }

    explicit
    operator bool() const noexcept {
        return static_cast<bool>(hostPtr_);
    }

    void
    reset() noexcept {
        hostPtr_.reset();
    }

private:
    friend class deviceOps;

    mappedHostMem(deviceOps *ops, void *host_ptr, void *dev_ptr) noexcept
        : hostPtr_(host_ptr, mappedHostMemDeleter{ops}),
          devPtr_(dev_ptr) {}

    std::unique_ptr<void, mappedHostMemDeleter> hostPtr_;
    void *devPtr_ = nullptr;
};

/** Process-wide device operations for the device runtime available to this process. */
[[nodiscard]] deviceOps &
getDeviceOps() noexcept;

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_DEVICE_OPS_H
