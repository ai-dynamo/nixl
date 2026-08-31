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

#include "mocks/error_injection/error_injection.h"

namespace mocks::error_injection {

namespace {

    /*
     * Stands in for a backend memory view. The agent only maps the handle back to
     * the engine that produced it, so any distinct non-null address will do.
     */
    char mem_view_handle;

} // namespace

mockErrorBackendEngine::mockErrorBackendEngine() {
    using testing::_;
    using testing::Return;

    ON_CALL(*this, prepMemView(testing::A<const nixl_meta_dlist_t &>(), _, _))
        .WillByDefault(
            [](const nixl_meta_dlist_t &, nixlMemViewH &view, const nixl_opt_b_args_t *) {
                view = &mem_view_handle;
                return NIXL_SUCCESS;
            });
    ON_CALL(*this, queryMem(_, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, estimateXferCost(_, _, _, _, _, _, _, _, _)).WillByDefault(Return(NIXL_SUCCESS));
}

mockAgent::mockAgent(const std::string &name) {
    agent_ = std::make_unique<nixlAgent>(name, nixlAgentConfig{});
}

nixlAgent &
mockAgent::agent() {
    return *agent_;
}

mockErrorBackendEngine &
mockAgent::gmock() {
    return gmock_;
}

nixl_status_t
mockAgent::createBackend(nixl_b_params_t &params, nixlBackendH *&backend) {
    gmock_.SetToParams(params);
    return agent_->createBackend(mock_backend_name, params, backend);
}

const char *
siteName(injection_site_t site) {
    switch (site) {
    case injection_site_t::NONE:
        return "(none)";
    case injection_site_t::REGISTER_MEM:
        return "registerMem";
    case injection_site_t::DEREGISTER_MEM:
        return "deregisterMem";
    case injection_site_t::LOAD_LOCAL_MD:
        return "loadLocalMD";
    case injection_site_t::GET_PUBLIC_DATA:
        return "getPublicData";
    case injection_site_t::GET_CONN_INFO:
        return "getConnInfo";
    case injection_site_t::LOAD_REMOTE_CONN_INFO:
        return "loadRemoteConnInfo";
    case injection_site_t::LOAD_REMOTE_MD:
        return "loadRemoteMD";
    case injection_site_t::CONNECT:
        return "connect";
    case injection_site_t::DISCONNECT:
        return "disconnect";
    case injection_site_t::UNLOAD_MD:
        return "unloadMD";
    case injection_site_t::PREP_XFER:
        return "prepXfer";
    case injection_site_t::POST_XFER:
        return "postXfer";
    case injection_site_t::CHECK_XFER:
        return "checkXfer";
    case injection_site_t::RELEASE_REQ:
        return "releaseReqH";
    case injection_site_t::PREP_REMOTE_MEM_VIEW:
        return "prepMemView(remote)";
    case injection_site_t::PREP_LOCAL_MEM_VIEW:
        return "prepMemView(local)";
    case injection_site_t::GET_NOTIFS:
        return "getNotifs";
    case injection_site_t::GEN_NOTIF:
        return "genNotif";
    case injection_site_t::QUERY_MEM:
        return "queryMem";
    case injection_site_t::ESTIMATE_XFER_COST:
        return "estimateXferCost";
    }
    return "(unknown)";
}

void
applyInjection(mockErrorBackendEngine &gmock,
               injection_site_t site,
               nixl_status_t status,
               unsigned &calls) {
    using testing::_;

    /* Polymorphic, so the one action fits every mocked signature below. */
    const auto injected =
        testing::DoAll(testing::InvokeWithoutArgs([&calls] { ++calls; }), testing::Return(status));

    switch (site) {
    case injection_site_t::NONE:
        break;
    case injection_site_t::REGISTER_MEM:
        ON_CALL(gmock, registerMem(_, _, _)).WillByDefault(injected);
        break;
    case injection_site_t::DEREGISTER_MEM:
        ON_CALL(gmock, deregisterMem(_)).WillByDefault(injected);
        break;
    case injection_site_t::LOAD_LOCAL_MD:
        ON_CALL(gmock, loadLocalMD(_, _)).WillByDefault(injected);
        break;
    case injection_site_t::GET_PUBLIC_DATA:
        ON_CALL(gmock, getPublicData(_, _)).WillByDefault(injected);
        break;
    case injection_site_t::GET_CONN_INFO:
        ON_CALL(gmock, getConnInfo(_)).WillByDefault(injected);
        break;
    case injection_site_t::LOAD_REMOTE_CONN_INFO:
        ON_CALL(gmock, loadRemoteConnInfo(_, _)).WillByDefault(injected);
        break;
    case injection_site_t::LOAD_REMOTE_MD:
        ON_CALL(gmock, loadRemoteMD(_, _, _, _)).WillByDefault(injected);
        break;
    case injection_site_t::CONNECT:
        ON_CALL(gmock, connect(_)).WillByDefault(injected);
        break;
    case injection_site_t::DISCONNECT:
        ON_CALL(gmock, disconnect(_)).WillByDefault(injected);
        break;
    case injection_site_t::UNLOAD_MD:
        ON_CALL(gmock, unloadMD(_)).WillByDefault(injected);
        break;
    case injection_site_t::PREP_XFER:
        ON_CALL(gmock, prepXfer(_, _, _, _, _, _)).WillByDefault(injected);
        break;
    case injection_site_t::POST_XFER:
        ON_CALL(gmock, postXfer(_, _, _, _, _, _)).WillByDefault(injected);
        break;
    case injection_site_t::CHECK_XFER:
        ON_CALL(gmock, checkXfer(_)).WillByDefault(injected);
        break;
    case injection_site_t::RELEASE_REQ:
        ON_CALL(gmock, releaseReqH(_)).WillByDefault(injected);
        break;
    case injection_site_t::PREP_REMOTE_MEM_VIEW:
        /*
         * The using declaration only un-hides the virtual, not the mock member
         * that ON_CALL expands to, so the base has to be named explicitly here
         * to reach the remote overload.
         */
        ON_CALL(static_cast<mocks::GMockBackendEngine &>(gmock), prepMemView(_, _, _))
            .WillByDefault(injected);
        break;
    case injection_site_t::PREP_LOCAL_MEM_VIEW:
        ON_CALL(gmock, prepMemView(testing::A<const nixl_meta_dlist_t &>(), _, _))
            .WillByDefault(injected);
        break;
    case injection_site_t::GET_NOTIFS:
        ON_CALL(gmock, getNotifs(_)).WillByDefault(injected);
        break;
    case injection_site_t::GEN_NOTIF:
        ON_CALL(gmock, genNotif(_, _)).WillByDefault(injected);
        break;
    case injection_site_t::QUERY_MEM:
        ON_CALL(gmock, queryMem(_, _)).WillByDefault(injected);
        break;
    case injection_site_t::ESTIMATE_XFER_COST:
        ON_CALL(gmock, estimateXferCost(_, _, _, _, _, _, _, _, _)).WillByDefault(injected);
        break;
    }
}

} // namespace mocks::error_injection
