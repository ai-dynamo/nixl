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
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend_aux.h"
#include "device/device_ops.h"
#include "device/proxy/proxy_transport.h"

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

    /** The memory-view manager stores backend metadata by pointer and never looks inside. */
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

    class MockTransport;

    /**
     * Records submissions and holds requests until the test completes them. It outlives the
     * runtimes a test creates; each runtime gets a fresh transport() that reports here.
     */
    class MockBackend {
    public:
        /** Optional per-test overrides; a set hook replaces the recording behavior. */
        struct Hooks {
            std::function<nixl_status_t(const nixl::proxyConfig &)> init;
            std::function<nixl_status_t(const nixl::proxyBackendSubmission &,
                                        nixl::proxyBackendRequest &)>
                submit;
            std::function<nixl_status_t(uint32_t, uint32_t)> quiesce;
            std::function<nixl_status_t(const nixl_remote_meta_dlist_t &, std::vector<void *> &)>
                resolve_direct_ptrs;
        };

        /** A transport bound to this recorder; set hooks on it before create(). */
        std::unique_ptr<MockTransport>
        transport();

        void
        complete(uint64_t token, nixl_status_t status = NIXL_SUCCESS) {
            const std::lock_guard<std::mutex> lock(mutex_);
            entries_.at(token - 1).status = status;
        }

        /** Also completes requests submitted later, for teardown. */
        void
        completeEverything() {
            const std::lock_guard<std::mutex> lock(mutex_);
            complete_on_check_ = true;
        }

        void
        failNextSubmit(nixl_status_t status) {
            const std::lock_guard<std::mutex> lock(mutex_);
            next_submit_status_ = status;
        }

        size_t
        submissionCount() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return entries_.size();
        }

        std::vector<nixl::proxyBackendSubmission>
        submissions() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            std::vector<nixl::proxyBackendSubmission> result;
            for (const auto &entry : entries_) {
                result.push_back(entry.submission);
            }
            return result;
        }

        static uint64_t
        token(size_t index) {
            return index + 1;
        }

        size_t
        quiesceCalls() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return quiesce_calls_;
        }

        size_t
        shutdownCalls() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return shutdown_calls_;
        }

        // The recording behavior behind MockTransport.

        nixl_status_t
        recordSubmit(const nixl::proxyBackendSubmission &submission,
                     nixl::proxyBackendRequest &request) {
            const std::lock_guard<std::mutex> lock(mutex_);
            entries_.push_back({submission});
            const auto status = std::exchange(next_submit_status_, NIXL_IN_PROG);
            request = status == NIXL_IN_PROG ? nixl::proxyBackendRequest{entries_.size()} :
                                               nixl::proxyBackendRequest{};
            return status;
        }

        nixl_status_t
        check(const nixl::proxyBackendRequest &request) const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return complete_on_check_ ? NIXL_SUCCESS : entries_.at(request.token - 1).status;
        }

        nixl_status_t
        recordQuiesce() {
            const std::lock_guard<std::mutex> lock(mutex_);
            ++quiesce_calls_;
            return NIXL_SUCCESS;
        }

        nixl_status_t
        recordShutdown() {
            const std::lock_guard<std::mutex> lock(mutex_);
            ++shutdown_calls_;
            return NIXL_SUCCESS;
        }

    private:
        struct Entry {
            nixl::proxyBackendSubmission submission;
            nixl_status_t status = NIXL_IN_PROG;
        };

        mutable std::mutex mutex_;
        std::vector<Entry> entries_;
        nixl_status_t next_submit_status_ = NIXL_IN_PROG;
        bool complete_on_check_ = false;
        size_t quiesce_calls_ = 0;
        size_t shutdown_calls_ = 0;
    };

    /** Each operation runs its hook when set, else MockBackend's recording behavior. */
    class MockTransport final : public nixl::proxyTransport {
    public:
        explicit MockTransport(MockBackend &backend) : backend_(backend) {}

        MockBackend::Hooks hooks;

        nixl_status_t
        init(const nixl::proxyConfig &config) override {
            return hooks.init ? hooks.init(config) : NIXL_SUCCESS;
        }

        nixl_status_t
        submit(const nixl::proxyBackendSubmission &submission,
               nixl::proxyBackendRequest &request) override {
            return hooks.submit ? hooks.submit(submission, request) :
                                  backend_.recordSubmit(submission, request);
        }

        nixl_status_t
        checkCompletion(uint32_t, uint32_t, const nixl::proxyBackendRequest &request) override {
            return backend_.check(request);
        }

        void
        progress(uint32_t, uint32_t) noexcept override {}

        /** An overriding hook bypasses quiesceCalls(). */
        nixl_status_t
        quiesce(uint32_t channel, uint32_t peer) override {
            return hooks.quiesce ? hooks.quiesce(channel, peer) : backend_.recordQuiesce();
        }

        nixl_status_t
        shutdown() override {
            return backend_.recordShutdown();
        }

        nixl_status_t
        resolveDirectPtrs(const nixl_remote_meta_dlist_t &dlist,
                          std::vector<void *> &direct_ptrs) override {
            return hooks.resolve_direct_ptrs ?
                hooks.resolve_direct_ptrs(dlist, direct_ptrs) :
                nixl::proxyTransport::resolveDirectPtrs(dlist, direct_ptrs);
        }

    private:
        MockBackend &backend_;
    };

    inline std::unique_ptr<MockTransport>
    MockBackend::transport() {
        return std::make_unique<MockTransport>(*this);
    }

} // namespace proxy_mocks
} // namespace gtest

#endif // NIXL_TEST_GTEST_MOCKS_PROXY_MOCKS_H
