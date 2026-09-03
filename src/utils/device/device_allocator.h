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
#ifndef NIXL_SRC_UTILS_DEVICE_DEVICE_ALLOCATOR_H
#define NIXL_SRC_UTILS_DEVICE_DEVICE_ALLOCATOR_H

#include <cstddef>
#include <utility>

#include <nixl_types.h>

class nixlDeviceMem;
class nixlMappedHostMem;

/**
 * Device memory-ops interface. All host-side interaction with the GPU memory
 * runtime goes through this class so that no other host code needs a
 * cuda_runtime.h include. CUDA is the current platform implementation; HIP
 * can provide another. Allocations are returned as owning RAII handles, while
 * raw alloc/free remain protected implementation hooks. The class is not
 * internally synchronized. Active device state is thread-local: copies,
 * memset, and synchronize use the caller's current device, while
 * device-memory free is the platform hook's job and must land on the
 * pointer's owning device.
 */
class nixlDeviceAllocator {
public:
    virtual ~nixlDeviceAllocator() = default;

    [[nodiscard]] nixl_status_t
    allocDeviceMem(size_t size, nixlDeviceMem &out) noexcept;

    /**
     * Allocate pinned host memory that is mapped into the device address
     * space; the handle exposes both the host pointer and its
     * device-visible alias.
     */
    [[nodiscard]] nixl_status_t
    allocMappedHostMem(size_t size, nixlMappedHostMem &out) noexcept;

    /**
     * Free device memory. Used for pointers whose ownership left RAII scope
     * via nixlDeviceMem::release() (for example, nixlMemViewH handles crossing
     * the public API boundary). The platform implementation is responsible for
     * freeing on the pointer's owning device, whatever device is current.
     * Prefer the RAII handles everywhere else.
     */
    void
    freeDeviceMem(void *ptr) noexcept {
        doFreeDeviceMem(ptr);
    }

    [[nodiscard]] virtual nixl_status_t
    copyHostToDevice(void *dst, const void *src, size_t size) noexcept = 0;

    [[nodiscard]] virtual nixl_status_t
    copyDeviceToHost(void *dst, const void *src, size_t size) noexcept = 0;

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
    doAllocDeviceMem(void **ptr, size_t size) noexcept = 0;

    virtual void
    doFreeDeviceMem(void *ptr) noexcept = 0;

    [[nodiscard]] virtual nixl_status_t
    doAllocMappedHostMem(void **host_ptr, void **dev_ptr, size_t size) noexcept = 0;

    virtual void
    doFreeMappedHostMem(void *host_ptr) noexcept = 0;

private:
    friend class nixlDeviceMem;
    friend class nixlMappedHostMem;
};

/**
 * Owning, move-only handle to device (HBM) memory. The handle stores only the
 * allocator, the pointer, and the size. The platform implementation of
 * doFreeDeviceMem() must free on the pointer's owning device, whatever device
 * is active in the destroying thread.
 */
class nixlDeviceMem {
public:
    nixlDeviceMem() = default;

    ~nixlDeviceMem() {
        reset();
    }

    nixlDeviceMem(nixlDeviceMem &&other) noexcept {
        *this = std::move(other);
    }

    nixlDeviceMem &
    operator=(nixlDeviceMem &&other) noexcept {
        if (this != &other) {
            reset();
            allocator_ = std::exchange(other.allocator_, nullptr);
            ptr_ = std::exchange(other.ptr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    nixlDeviceMem(const nixlDeviceMem &) = delete;
    nixlDeviceMem &
    operator=(const nixlDeviceMem &) = delete;

    [[nodiscard]] void *
    get() const noexcept {
        return ptr_;
    }

    template<class T>
    [[nodiscard]] T *
    as() const noexcept {
        return static_cast<T *>(ptr_);
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return size_;
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
        allocator_->doFreeDeviceMem(ptr_);
        allocator_ = nullptr;
        ptr_ = nullptr;
        size_ = 0;
    }

    /** Give up ownership; the pointer must later go to freeDeviceMem(). */
    [[nodiscard]] void *
    release() noexcept {
        allocator_ = nullptr;
        size_ = 0;
        return std::exchange(ptr_, nullptr);
    }

private:
    friend class nixlDeviceAllocator;

    nixlDeviceMem(nixlDeviceAllocator *allocator, void *ptr, size_t size) noexcept
        : allocator_(allocator),
          ptr_(ptr),
          size_(size) {}

    nixlDeviceAllocator *allocator_ = nullptr;
    void *ptr_ = nullptr;
    size_t size_ = 0;
};

/**
 * Owning, move-only handle to pinned host memory mapped into the device
 * address space. Exposes the host pointer and its device-visible alias.
 */
class nixlMappedHostMem {
public:
    nixlMappedHostMem() = default;

    ~nixlMappedHostMem() {
        reset();
    }

    nixlMappedHostMem(nixlMappedHostMem &&other) noexcept {
        *this = std::move(other);
    }

    nixlMappedHostMem &
    operator=(nixlMappedHostMem &&other) noexcept {
        if (this != &other) {
            reset();
            allocator_ = std::exchange(other.allocator_, nullptr);
            host_ptr_ = std::exchange(other.host_ptr_, nullptr);
            dev_ptr_ = std::exchange(other.dev_ptr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    nixlMappedHostMem(const nixlMappedHostMem &) = delete;
    nixlMappedHostMem &
    operator=(const nixlMappedHostMem &) = delete;

    [[nodiscard]] void *
    hostPtr() const noexcept {
        return host_ptr_;
    }

    [[nodiscard]] void *
    devPtr() const noexcept {
        return dev_ptr_;
    }

    template<class T>
    [[nodiscard]] T *
    asHost() const noexcept {
        return static_cast<T *>(host_ptr_);
    }

    template<class T>
    [[nodiscard]] T *
    asDev() const noexcept {
        return static_cast<T *>(dev_ptr_);
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return size_;
    }

    explicit
    operator bool() const noexcept {
        return host_ptr_ != nullptr;
    }

    void
    reset() noexcept {
        if (host_ptr_ == nullptr) {
            return;
        }
        allocator_->doFreeMappedHostMem(host_ptr_);
        allocator_ = nullptr;
        host_ptr_ = nullptr;
        dev_ptr_ = nullptr;
        size_ = 0;
    }

private:
    friend class nixlDeviceAllocator;

    nixlMappedHostMem(nixlDeviceAllocator *allocator,
                      void *host_ptr,
                      void *dev_ptr,
                      size_t size) noexcept
        : allocator_(allocator),
          host_ptr_(host_ptr),
          dev_ptr_(dev_ptr),
          size_(size) {}

    nixlDeviceAllocator *allocator_ = nullptr;
    void *host_ptr_ = nullptr;
    void *dev_ptr_ = nullptr;
    size_t size_ = 0;
};

inline nixl_status_t
nixlDeviceAllocator::allocDeviceMem(size_t size, nixlDeviceMem &out) noexcept {
    // `out` is only touched on success; a failed allocation leaves the
    // caller's existing buffer intact.
    void *ptr = nullptr;
    const nixl_status_t status = doAllocDeviceMem(&ptr, size);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    out = nixlDeviceMem(this, ptr, size);
    return NIXL_SUCCESS;
}

inline nixl_status_t
nixlDeviceAllocator::allocMappedHostMem(size_t size, nixlMappedHostMem &out) noexcept {
    // `out` is only touched on success; a failed allocation leaves the
    // caller's existing buffer intact.
    void *host_ptr = nullptr;
    void *dev_ptr = nullptr;
    const nixl_status_t status = doAllocMappedHostMem(&host_ptr, &dev_ptr, size);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    out = nixlMappedHostMem(this, host_ptr, dev_ptr, size);
    return NIXL_SUCCESS;
}

/** Process-wide allocator for the platform this library was built for. */
[[nodiscard]] nixlDeviceAllocator &
nixlGetDeviceAllocator() noexcept;

#endif // NIXL_SRC_UTILS_DEVICE_DEVICE_ALLOCATOR_H
