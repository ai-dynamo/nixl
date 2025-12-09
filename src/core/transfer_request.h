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
#include <vector>
#include <utility>
#include <memory>
#include <algorithm>
#include <stdexcept>

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
    std::vector<std::pair<nixlBackendEngine *, std::shared_ptr<nixl_meta_dlist_t>>> descs;

    std::string remoteAgent;
    bool isLocal;

    // Helper method to find an entry by backend
    inline auto
    find(nixlBackendEngine *backend) {
        return std::find_if(
            descs.begin(), descs.end(), [backend](const auto &p) { return p.first == backend; });
    }

    inline auto
    find(nixlBackendEngine *backend) const {
        return std::find_if(
            descs.begin(), descs.end(), [backend](const auto &p) { return p.first == backend; });
    }

public:
    inline nixlDlistH() {}

    inline ~nixlDlistH() {
        // shared_ptr handles cleanup automatically
    }

    // Accessor methods to encapsulate internal data structure
    inline size_t
    count(nixlBackendEngine *backend) const {
        return find(backend) != descs.end() ? 1 : 0;
    }

    inline std::shared_ptr<nixl_meta_dlist_t>
    at(nixlBackendEngine *backend) {
        auto it = find(backend);
        if (it == descs.end()) throw std::out_of_range("Backend not found in descs");
        return it->second;
    }

    inline std::shared_ptr<nixl_meta_dlist_t>
    at(nixlBackendEngine *backend) const {
        auto it = find(backend);
        if (it == descs.end()) throw std::out_of_range("Backend not found in descs");
        return it->second;
    }

    inline std::shared_ptr<nixl_meta_dlist_t>
    operator[](nixlBackendEngine *backend) {
        auto it = find(backend);
        if (it != descs.end()) return it->second;
        descs.emplace_back(backend, nullptr);
        return descs.back().second;
    }

    inline void
    erase(nixlBackendEngine *backend) {
        auto it = find(backend);
        if (it != descs.end()) descs.erase(it);
    }

    // Iterators for range-based for loops
    inline auto
    begin() {
        return descs.begin();
    }

    inline auto
    end() {
        return descs.end();
    }

    inline auto
    begin() const {
        return descs.begin();
    }

    inline auto
    end() const {
        return descs.end();
    }

    friend class nixlAgent;
};

#endif
