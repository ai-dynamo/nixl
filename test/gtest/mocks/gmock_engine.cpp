/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gmock_engine.h"

namespace mocks {

nixl_b_params_t custom_params{};
const nixlBackendInitParams init_params{.customParams = &custom_params};
const std::string gmock_engine_key = "gmock_engine_key";
char gmock_dummy_mvh;

namespace {

    nixlBackendInitParams
    normalizeInitParams(const nixlBackendInitParams *params) {
        nixlBackendInitParams normalized = params == nullptr ? nixlBackendInitParams{} : *params;
        if (normalized.customParams == nullptr) {
            normalized.customParams = &custom_params;
        }
        return normalized;
    }

} // namespace

GMockBackendEngine::GMockBackendEngine() : GMockBackendEngine(&init_params) {}

GMockBackendEngine::GMockBackendEngine(const nixlBackendInitParams *init_params)
    : GMockBackendEngine(normalizeInitParams(init_params)) {}

GMockBackendEngine::GMockBackendEngine(nixlBackendInitParams init_params)
    : nixlBackendEngine(&init_params) {
    setDefaults();
}

void
GMockBackendEngine::setDefaults() {
    using testing::Return;
    using testing::A;
    using testing::_;

    ON_CALL(*this, supportsRemote()).WillByDefault(Return(true));
    ON_CALL(*this, supportsLocal()).WillByDefault(Return(true));
    ON_CALL(*this, supportsNotif()).WillByDefault(Return(true));
    ON_CALL(*this, getSupportedMems()).WillByDefault(Return(nixl_mem_list_t{DRAM_SEG}));
    ON_CALL(*this, registerMem(_, _, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, deregisterMem(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, connect(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, disconnect(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, unloadMD(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, prepXfer(_, _, _, _, _, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, postXfer(_, _, _, _, _, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, checkXfer(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, releaseReqH(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, prepMemView(A<const nixl_remote_meta_dlist_t &>(), _, _))
        .WillByDefault(
            [](const nixl_remote_meta_dlist_t &, nixlMemViewH &mvh, const nixl_opt_b_args_t *) {
                mvh = &gmock_dummy_mvh;
                return NIXL_SUCCESS;
            });
    ON_CALL(*this, getPublicData(_, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, getConnInfo(_)).WillByDefault([&](std::string &str) {
        str = "mock_backend_plugin_conn_info";
        return NIXL_SUCCESS;
    });
    ON_CALL(*this, loadRemoteConnInfo(_, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, loadRemoteMD(_, _, _, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, loadLocalMD(_, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, getNotifs(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, genNotif(_, _)).WillByDefault(Return(NIXL_SUCCESS));
    setOptionalDefaults();
}

void
GMockBackendEngine::setOptionalDefaults() {
    using testing::A;
    using testing::Return;
    using testing::_;

    ON_CALL(*this, prepMemView(A<const nixl_meta_dlist_t &>(), _, _))
        .WillByDefault(Return(NIXL_ERR_NOT_SUPPORTED));
    ON_CALL(*this, queryMem(_, _)).WillByDefault(Return(NIXL_ERR_NOT_SUPPORTED));
    ON_CALL(*this, estimateXferCost(_, _, _, _, _, _, _, _, _))
        .WillByDefault(Return(NIXL_ERR_NOT_SUPPORTED));
}

void
GMockBackendEngine::SetToParams(nixl_b_params_t &params) const {
    params[gmock_engine_key] = std::to_string(reinterpret_cast<uintptr_t>(this));
}

GMockBackendEngine *
GMockBackendEngine::GetFromParams(nixl_b_params_t *params) {
    try {
        std::string gmock_engine_ptr_str = params->at(gmock_engine_key);
        return reinterpret_cast<GMockBackendEngine *>(std::stoul(gmock_engine_ptr_str));
    }
    catch (const std::exception &e) {
        std::cerr << "Error getting GMockBackendEngine from params: " << e.what() << std::endl;
        throw e;
    }
}

} // namespace mocks
