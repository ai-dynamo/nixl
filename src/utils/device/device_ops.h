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

/**
 * @brief Owns device storage.
 * @note Preserve its deleter when rewrapping a released pointer.
 */
using deviceMem = std::unique_ptr<void, deviceMemDeleter>;

#define NIXL_DEVICE_OPS_EXPORT __attribute__((visibility("default")))

/**
 * @brief Device memory operations provided by a runtime-loaded backend.
 * @note Active device selection is thread-local.
 */
class deviceOps {
public:
    enum class copyDirection { HostToDevice, DeviceToHost };

    virtual ~deviceOps() = default;

    /**
     * @brief Allocate device memory on the active device.
     * @param[out] out Owning handle; unchanged on failure.
     * @retval NIXL_ERR_INVALID_PARAM size is zero.
     * @note This instance must outlive the returned handle.
     */
    [[nodiscard]] nixl_status_t
    allocDeviceMem(size_t size, deviceMem &out) noexcept;

    /**
     * @brief Allocate pinned host memory with a device-visible alias.
     * @param[out] out Owning handle; unchanged on failure.
     * @retval NIXL_ERR_INVALID_PARAM size is zero.
     * @note This instance must outlive the returned handle.
     */
    [[nodiscard]] nixl_status_t
    allocMappedHostMem(size_t size, mappedHostMem &out) noexcept;

    /**
     * @brief Copy between host and device memory using the default stream.
     * @note On success, H2D source storage is reusable; D2H destination data is ready.
     *       H2D device completion may still be pending.
     * @retval NIXL_ERR_INVALID_PARAM Zero size, null pointer or invalid direction;
     *         no copy is performed.
     */
    [[nodiscard]] virtual nixl_status_t
    copy(void *dst, const void *src, size_t size, copyDirection direction) noexcept = 0;

    /**
     * @brief Fill device memory using the default stream.
     * @note Completion is not guaranteed on return.
     * @retval NIXL_ERR_INVALID_PARAM Zero size or null pointer; no write is performed.
     */
    [[nodiscard]] virtual nixl_status_t
    memsetDeviceMem(void *ptr, int value, size_t size) noexcept = 0;

    /** @brief Block until all outstanding work on the active device completes. */
    [[nodiscard]] virtual nixl_status_t
    synchronize() noexcept = 0;

    /** @brief Get the calling thread's active device. */
    [[nodiscard]] virtual nixl_status_t
    getActiveDevice(int &device_id) noexcept = 0;

    /** @brief Set the calling thread's active device. */
    [[nodiscard]] virtual nixl_status_t
    setActiveDevice(int device_id) noexcept = 0;

protected:
    /** @brief Allocate device storage, assigning ptr only on success. */
    [[nodiscard]] virtual nixl_status_t
    doAllocDeviceMem(void *&ptr, size_t size) noexcept = 0;

    /** @brief Free device storage regardless of the calling thread's active device. */
    virtual void
    doFreeDeviceMem(void *ptr) noexcept = 0;

    /** @brief Allocate mapped host storage, assigning both pointers only on success. */
    [[nodiscard]] virtual nixl_status_t
    doAllocMappedHostMem(void *&host_ptr, void *&dev_ptr, size_t size) noexcept = 0;

    /** @brief Free the host allocation, not its device alias; no device switch required. */
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

/** @brief Owns pinned host memory and exposes its non-owning device alias. */
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

/**
 * @brief Get the process-wide device operations.
 * @return Borrowed pointer, or nullptr if unavailable.
 * @note The first result is cached, including failure.
 */
[[nodiscard]] deviceOps *
getDeviceOps() noexcept;

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_DEVICE_OPS_H
