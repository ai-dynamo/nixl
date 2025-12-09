/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef __TRANSFER_REQUEST_H_
#define __TRANSFER_REQUEST_H_

#include <string>
#include <array>
#include <utility>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cstddef>

#include "nixl_types.h"
#include "backend_engine.h"
#include "telemetry.h"

enum nixl_telemetry_stat_status_t {
    NIXL_TELEMETRY_POST = 0,
    NIXL_TELEMETRY_POST_AND_FINISH = 1,
    NIXL_TELEMETRY_FINISH = 2
};

// Contains pointers to corresponding backend engine and its handler, and populated
// and verified DescLists, and other state and metadata needed for a NIXL transfer
class nixlXferReqH {
private:
    nixlBackendEngine *engine = nullptr;
    nixlBackendReqH *backendHandle = nullptr;

    std::shared_ptr<nixl_meta_dlist_t> initiatorDescs;
    std::shared_ptr<nixl_meta_dlist_t> targetDescs;

    std::string remoteAgent;
    nixl_blob_t notifMsg;
    bool hasNotif = false;

    nixl_xfer_op_t backendOp;
    nixl_status_t status;

    nixl_xfer_telem_t telemetry;

public:
    inline nixlXferReqH() {}

    inline ~nixlXferReqH() {
        // shared_ptr handles cleanup automatically
        if (backendHandle != nullptr) engine->releaseReqH(backendHandle);
    }

    void
    updateRequestStats(std::unique_ptr<nixlTelemetry> &telemetry,
                       nixl_telemetry_stat_status_t stat_status);

    friend class nixlAgent;
};

class nixlDlistH {
private:
    size_t capacity_;
    size_t size_;

    std::string remoteAgent;
    bool isLocal;

    using DescPair = std::pair<nixlBackendEngine *, std::shared_ptr<nixl_meta_dlist_t>>;

    // Helper to get typed pointer to the storage
    inline DescPair *
    getDescs() {
        return reinterpret_cast<DescPair *>(descs_storage);
    }

    inline const DescPair *
    getDescs() const {
        return reinterpret_cast<const DescPair *>(descs_storage);
    }

    // Helper method to find an entry by backend
    inline DescPair *
    find(nixlBackendEngine *backend) {
        auto descs = getDescs();
        return std::find_if(
            descs, descs + size_, [backend](const auto &p) { return p.first == backend; });
    }

    inline const DescPair *
    find(nixlBackendEngine *backend) const {
        auto descs = getDescs();
        return std::find_if(
            descs, descs + size_, [backend](const auto &p) { return p.first == backend; });
    }

public:
    // Custom operator new to allocate space for flexible array member
    static void *
    operator new(size_t base_size, size_t num_backends) {
        using DescPair = std::pair<nixlBackendEngine *, std::shared_ptr<nixl_meta_dlist_t>>;
        size_t total = base_size + num_backends * sizeof(DescPair);
        return ::operator new(total);
    }

    static void
    operator delete(void *p) noexcept {
        ::operator delete(p);
    }

    // Placement delete for exception safety
    static void
    operator delete(void *p, size_t) noexcept {
        ::operator delete(p);
    }

    explicit nixlDlistH(size_t capacity) : capacity_(capacity), size_(0) {}

    ~nixlDlistH() {
        // Manually destroy each pair in the flexible array member
        auto descs = getDescs();
        for (size_t i = 0; i < size_; ++i) {
            descs[i].~DescPair();
        }
    }

    // Accessor methods to encapsulate internal data structure
    inline size_t
    count(nixlBackendEngine *backend) const {
        return find(backend) != (getDescs() + size_) ? 1 : 0;
    }

    inline std::shared_ptr<nixl_meta_dlist_t>
    at(nixlBackendEngine *backend) {
        auto it = find(backend);
        auto descs = getDescs();
        if (it == descs + size_) throw std::out_of_range("Backend not found in descs");
        return it->second;
    }

    inline std::shared_ptr<nixl_meta_dlist_t>
    at(nixlBackendEngine *backend) const {
        auto it = find(backend);
        auto descs = getDescs();
        if (it == descs + size_) throw std::out_of_range("Backend not found in descs");
        return it->second;
    }

    inline std::shared_ptr<nixl_meta_dlist_t> &
    operator[](nixlBackendEngine *backend) {
        auto descs = getDescs();
        auto it = find(backend);
        if (it != descs + size_) return it->second;
        if (size_ >= capacity_) throw std::out_of_range("nixlDlistH: Maximum capacity exceeded");
        // Use placement new to construct the pair in uninitialized memory
        new (&descs[size_]) DescPair(backend, nullptr);
        return descs[size_++].second;
    }

    inline void
    erase(nixlBackendEngine *backend) {
        auto descs = getDescs();
        auto it = find(backend);
        if (it != descs + size_) {
            // Shift elements left to fill the gap
            for (auto next = it + 1; next != descs + size_; ++next, ++it) {
                *it = *next;
            }
            --size_;
        }
    }

    // Iterators for range-based for loops
    inline auto
    begin() {
        return getDescs();
    }

    inline auto
    end() {
        return getDescs() + size_;
    }

    inline auto
    begin() const {
        return getDescs();
    }

    inline auto
    end() const {
        return getDescs() + size_;
    }

    friend class nixlAgent;

    // Flexible array member - must be last (using std::byte for storage)
    alignas(std::pair<nixlBackendEngine *, std::shared_ptr<nixl_meta_dlist_t>>) std::byte
        descs_storage[];
};

#endif
