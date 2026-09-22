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
#include <utility>

#include <nixl_types.h>

namespace nixl {

class deviceMem;
class mappedHostMem;

#define NIXL_DEVICE_OPS_EXPORT __attribute__((visibility("default")))

/**
 * Device memory-ops interface. All host-side interaction with the GPU memory
 * runtime goes through this class so that no other host code needs a
 * cuda_runtime.h include. CUDA is the current platform implementation; HIP
 * can provide another. Allocations are returned as owning RAII handles.
 * Raw alloc/free remain protected; deviceMem::release()/adopt() is the
 * C-handle escape hatch. The class is not
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

    /** src is reusable on return; the device-side write may still be pending. */
    [[nodiscard]] virtual nixl_status_t
    copyHostToDevice(void *dst, const void *src, size_t size) noexcept = 0;

    /** Blocking: dst holds the data on return. */
    [[nodiscard]] virtual nixl_status_t
    copyDeviceToHost(void *dst, const void *src, size_t size) noexcept = 0;

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
    friend class deviceMem;
    friend class mappedHostMem;
};

/**
 * Owning, move-only handle to device memory. The platform free recovers the
 * owning device, so the destroying thread may be on any device.
 */
class deviceMem {
public:
    deviceMem() = default;

    ~deviceMem() {
        reset();
    }

    deviceMem(deviceMem &&other) noexcept
        : ops_(other.ops_),
          ptr_(std::exchange(other.ptr_, nullptr)) {}

    deviceMem &
    operator=(deviceMem &&other) noexcept {
        if (this != &other) {
            reset();
            ops_ = other.ops_;
            ptr_ = std::exchange(other.ptr_, nullptr);
        }
        return *this;
    }

    deviceMem(const deviceMem &) = delete;
    deviceMem &
    operator=(const deviceMem &) = delete;

    template<typename T = void>
    [[nodiscard]] T *
    devicePointer() const noexcept {
        return static_cast<T *>(ptr_);
    }

    explicit
    operator bool() const noexcept {
        return ptr_ != nullptr;
    }

    void
    reset() noexcept {
        if (ptr_ == nullptr) {
            return;
        }
        ops_->doFreeDeviceMem(ptr_);
        ptr_ = nullptr;
    }

    /** Give up ownership; reclaim with adopt(). */
    [[nodiscard]] void *
    release() noexcept {
        return std::exchange(ptr_, nullptr);
    }

    /**
     * Take ownership of a pointer previously returned by release(). Null
     * yields an empty handle. The pointer must be one allocDeviceMem produced.
     */
    [[nodiscard]] static deviceMem
    adopt(deviceOps &ops, void *ptr) noexcept {
        return deviceMem(&ops, ptr);
    }

private:
    friend class deviceOps;

    deviceMem(deviceOps *ops, void *ptr) noexcept : ops_(ops), ptr_(ptr) {}

    deviceOps *ops_ = nullptr;
    void *ptr_ = nullptr;
};

/**
 * Owning, move-only handle to pinned host memory mapped into the device
 * address space. Exposes the host pointer and its device-visible alias.
 */
class mappedHostMem {
public:
    mappedHostMem() = default;

    ~mappedHostMem() {
        reset();
    }

    mappedHostMem(mappedHostMem &&other) noexcept
        : ops_(other.ops_),
          hostPtr_(std::exchange(other.hostPtr_, nullptr)),
          devPtr_(std::exchange(other.devPtr_, nullptr)) {}

    mappedHostMem &
    operator=(mappedHostMem &&other) noexcept {
        if (this != &other) {
            reset();
            ops_ = other.ops_;
            hostPtr_ = std::exchange(other.hostPtr_, nullptr);
            devPtr_ = std::exchange(other.devPtr_, nullptr);
        }
        return *this;
    }

    mappedHostMem(const mappedHostMem &) = delete;
    mappedHostMem &
    operator=(const mappedHostMem &) = delete;

    template<typename T = void>
    [[nodiscard]] T *
    hostPointer() const noexcept {
        return static_cast<T *>(hostPtr_);
    }

    template<typename T = void>
    [[nodiscard]] T *
    devicePointer() const noexcept {
        return static_cast<T *>(devPtr_);
    }

    explicit
    operator bool() const noexcept {
        return hostPtr_ != nullptr;
    }

    void
    reset() noexcept {
        if (hostPtr_ == nullptr) {
            return;
        }
        ops_->doFreeMappedHostMem(hostPtr_);
        hostPtr_ = nullptr;
        devPtr_ = nullptr;
    }

private:
    friend class deviceOps;

    mappedHostMem(deviceOps *ops, void *host_ptr, void *dev_ptr) noexcept
        : ops_(ops),
          hostPtr_(host_ptr),
          devPtr_(dev_ptr) {}

    deviceOps *ops_ = nullptr;
    void *hostPtr_ = nullptr;
    void *devPtr_ = nullptr;
};

inline nixl_status_t
deviceOps::allocDeviceMem(size_t size, deviceMem &out) noexcept {
    if (size == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    void *ptr;
    const nixl_status_t status = doAllocDeviceMem(ptr, size);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    out = deviceMem(this, ptr);
    return NIXL_SUCCESS;
}

inline nixl_status_t
deviceOps::allocMappedHostMem(size_t size, mappedHostMem &out) noexcept {
    if (size == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    void *host_ptr;
    void *dev_ptr;
    const nixl_status_t status = doAllocMappedHostMem(host_ptr, dev_ptr, size);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    out = mappedHostMem(this, host_ptr, dev_ptr);
    return NIXL_SUCCESS;
}

/** Process-wide device operations for the device runtime available to this process. */
[[nodiscard]] deviceOps &
getDeviceOps() noexcept;

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_DEVICE_OPS_H
