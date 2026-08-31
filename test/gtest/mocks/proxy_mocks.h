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
#ifndef NIXL_TEST_GTEST_MOCKS_PROXY_MOCKS_H
#define NIXL_TEST_GTEST_MOCKS_PROXY_MOCKS_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend_aux.h"
#include "device/device_ops.h"
#include "device/proxy/proxy_backend_ops.h"

namespace gtest {
namespace proxy_mocks {

    /** Host-backed allocations with distinct device aliases to catch incorrect publication. */
    class MockDeviceOps : public nixl::deviceOps {
    public:
        /** Fail once after N allocation/H2D calls; -1 disables injection. */
        int fail_after = -1;

        nixl_status_t
        copy(void *dst, const void *src, size_t size, copyDirection direction) noexcept override {
            if (size == 0 || dst == nullptr || src == nullptr) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (direction == copyDirection::HostToDevice) {
                if (fail_after >= 0 && fail_after-- == 0) {
                    return NIXL_ERR_BACKEND;
                }
                std::memcpy(resolve(dst), src, size);
            } else if (direction == copyDirection::DeviceToHost) {
                std::memcpy(dst, resolve(src), size);
            } else {
                return NIXL_ERR_INVALID_PARAM;
            }
            return NIXL_SUCCESS;
        }

        nixl_status_t
        memsetDeviceMem(void *ptr, int value, size_t size) noexcept override {
            if (size == 0 || ptr == nullptr) {
                return NIXL_ERR_INVALID_PARAM;
            }
            std::memset(resolve(ptr), value, size);
            return NIXL_SUCCESS;
        }

        nixl_status_t
        synchronize() noexcept override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        getActiveDevice(int &device_id) noexcept override {
            device_id = 0;
            return NIXL_SUCCESS;
        }

        nixl_status_t
        setActiveDevice(int) noexcept override {
            return NIXL_SUCCESS;
        }

        /** Translate a live mapped device alias; reject host and freed pointers. */
        template<class T>
        T *
        hostAlias(const T *device_alias) const {
            const uintptr_t raw = reinterpret_cast<uintptr_t>(device_alias);
            if ((raw & kDeviceAliasTag) == 0) {
                return nullptr;
            }
            const uintptr_t host = raw & ~kDeviceAliasTag;

            const std::lock_guard<std::mutex> lock(mutex_);
            auto it = mapped_.upper_bound(host);
            if (it == mapped_.begin()) {
                return nullptr;
            }
            --it;
            if (host >= it->first + it->second) {
                return nullptr;
            }
            return reinterpret_cast<T *>(host);
        }

        size_t
        liveAllocations() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return live_.size();
        }

        bool
        wasFreed(const void *ptr) const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return freed_.count(ptr) != 0;
        }

    protected:
        nixl_status_t
        doAllocDeviceMem(void *&ptr, size_t size) noexcept override {
            if (fail_after >= 0 && fail_after-- == 0) {
                return NIXL_ERR_BACKEND;
            }
            void *allocation = allocate(size);
            if (allocation == nullptr) {
                return NIXL_ERR_BACKEND;
            }
            ptr = allocation;
            return NIXL_SUCCESS;
        }

        void
        doFreeDeviceMem(void *ptr) noexcept override {
            release(ptr);
        }

        nixl_status_t
        doAllocMappedHostMem(void *&host_ptr, void *&dev_ptr, size_t size) noexcept override {
            void *allocation = allocate(size);
            if (allocation == nullptr) {
                return NIXL_ERR_BACKEND;
            }
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                mapped_[reinterpret_cast<uintptr_t>(allocation)] = size;
            }
            host_ptr = allocation;
            dev_ptr =
                reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(allocation) | kDeviceAliasTag);
            return NIXL_SUCCESS;
        }

        void
        doFreeMappedHostMem(void *host_ptr) noexcept override {
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                mapped_.erase(reinterpret_cast<uintptr_t>(host_ptr));
            }
            release(host_ptr);
        }

    private:
        /** Never set in a user-space address, and non-canonical, so a dereference faults. */
        static constexpr uintptr_t kDeviceAliasTag = uintptr_t{1} << 62;

        /** Device-memory pointers are used as they are; a device alias is untagged. */
        static void *
        resolve(const void *ptr) noexcept {
            return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) & ~kDeviceAliasTag);
        }

        void *
        allocate(size_t size) noexcept {
            // Match control-buffer alignment; zeroing makes tests deterministic.
            const size_t rounded = std::max<size_t>(64, (size + 63) & ~size_t{63});
            void *allocation = std::aligned_alloc(64, rounded);
            if (allocation == nullptr) {
                return nullptr;
            }
            std::memset(allocation, 0, rounded);
            const std::lock_guard<std::mutex> lock(mutex_);
            live_.insert(allocation);
            freed_.erase(allocation);
            return allocation;
        }

        void
        release(void *ptr) noexcept {
            if (ptr == nullptr) {
                return;
            }
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                live_.erase(ptr);
                freed_.insert(ptr);
            }
            std::free(ptr);
        }

        mutable std::mutex mutex_;
        std::set<const void *> live_;
        std::set<const void *> freed_;
        /** Host base to size of every live mapped allocation. */
        std::map<uintptr_t, size_t> mapped_;
    };

    /** The registry stores backend metadata by pointer and never looks inside. */
    class DummyBackendMD : public nixlBackendMD {
    public:
        DummyBackendMD() : nixlBackendMD(false) {}
    };

    inline nixl_meta_dlist_t
    makeLocalDlist(uintptr_t addr, size_t len, uint64_t dev_id, nixlBackendMD *md) {
        nixl_meta_dlist_t dlist(DRAM_SEG);
        dlist.addDesc(nixlMetaDesc(addr, len, dev_id, md));
        return dlist;
    }

    inline nixlRemoteMetaDesc
    makeRemoteDesc(const std::string &agent,
                   uintptr_t addr,
                   size_t len,
                   uint64_t dev_id,
                   nixlBackendMD *md) {
        nixlRemoteMetaDesc desc(agent);
        desc.addr = addr;
        desc.len = len;
        desc.devId = dev_id;
        desc.metadataP = md;
        return desc;
    }


} // namespace proxy_mocks
} // namespace gtest

#endif // NIXL_TEST_GTEST_MOCKS_PROXY_MOCKS_H
