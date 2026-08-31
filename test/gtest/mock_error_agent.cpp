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

/*
 * Exercises every status-returning MOCK_BACKEND API through its corresponding
 * northbound agent path. Prints the injected plugin status next to the status
 * the agent reported, including pass-through, collapse, cleanup, and stateful
 * transfer behavior. Pass --csv for spreadsheet-friendly output, --list for the
 * scenario names accepted as a filter argument.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "nixl.h"
#include "mocks/gmock_engine.h"

namespace {

/* Matches the shared_library name in test/gtest/mocks/meson.build. */
constexpr const char *mock_backend_name = "MOCK_BACKEND";
constexpr size_t buf_len = 256;
/* Fits NIXL_ERR_REMOTE_DISCONNECT, the widest status name at 26 characters. */
constexpr size_t column_width = 30;
constexpr size_t column_gap = 4;
/* "transformed to NIXL_ERR_REPOST_ACTIVE", only used to size the header rule. */
constexpr size_t longest_verdict = 37;

/*
 * Stands in for a backend memory view. The agent only maps the handle back to
 * the engine that produced it, so any distinct non-null address will do.
 */
char mem_view_handle;

/* Southbound call sites the mock engine can be told to fail at. */
enum class injection_site_t {
    NONE,
    REGISTER_MEM,
    DEREGISTER_MEM,
    LOAD_LOCAL_MD,
    GET_PUBLIC_DATA,
    GET_CONN_INFO,
    LOAD_REMOTE_CONN_INFO,
    LOAD_REMOTE_MD,
    CONNECT,
    DISCONNECT,
    UNLOAD_MD,
    PREP_XFER,
    POST_XFER,
    CHECK_XFER,
    RELEASE_REQ,
    PREP_REMOTE_MEM_VIEW,
    PREP_LOCAL_MEM_VIEW,
    GET_NOTIFS,
    GEN_NOTIF,
    QUERY_MEM,
    ESTIMATE_XFER_COST,
};

enum class action_t {
    COMPLETE,
    CREATE_BACKEND,
    REGISTER_MEM,
    DEREGISTER_MEM,
    LOAD_REMOTE_MD,
    MAKE_CONNECTION,
    INVALIDATE_REMOTE_MD,
    CREATE_XFER,
    POST_XFER,
    CHECK_XFER,
    RELEASE_XFER,
    PREP_REMOTE_MEM_VIEW,
    PREP_LOCAL_MEM_VIEW,
    GET_NOTIFS,
    GEN_NOTIF,
    QUERY_MEM,
    ESTIMATE_XFER_COST,
    POLL_TO_COMPLETION,
    REPOST_ACTIVE,
    STATUS_BEFORE_POST,
    TELEMETRY_DISABLED,
};

/*
 * What the agent is expected to do with the injected status. This has to be
 * stated per scenario rather than derived from the observed status: the sites
 * that collapse to NIXL_ERR_BACKEND are indistinguishable from a pass-through
 * when NIXL_ERR_BACKEND is the status being injected.
 */
enum class behavior_t {
    COMPLETED, /* nothing injected, the whole sequence succeeds */
    PASS_THROUGH, /* agent reports the plugin status unchanged */
    PASS_THROUGH_RECOVERABLE, /* pass-through, and the peer's metadata survives for retry */
    COLLAPSED, /* agent discards the plugin status in favor of NIXL_ERR_BACKEND */
    TRANSFORMED, /* agent replaces the plugin status with a different specific one */
    IGNORED, /* agent discards the plugin status and reports success */
    PROGRESSED, /* not an error: the transfer completes after polling */
    AGENT_GENERATED, /* agent rejects the call itself, the plugin is never asked */
};

/*
 * Adds the optional methods that the shared GMockBackendEngine does not mock.
 * MOCK_BACKEND stores the object as a base pointer, so virtual dispatch still
 * reaches these driver-only overrides.
 */
class mockErrorBackendEngine : public mocks::GMockBackendEngine {
public:
    mockErrorBackendEngine() {
        using testing::_;
        using testing::Return;

        ON_CALL(*this, prepMemView(testing::A<const nixl_meta_dlist_t &>(), _, _))
            .WillByDefault(
                [](const nixl_meta_dlist_t &, nixlMemViewH &view, const nixl_opt_b_args_t *) {
                    view = &mem_view_handle;
                    return NIXL_SUCCESS;
                });
        ON_CALL(*this, queryMem(_, _)).WillByDefault(Return(NIXL_SUCCESS));
        ON_CALL(*this, estimateXferCost(_, _, _, _, _, _, _, _, _))
            .WillByDefault(Return(NIXL_SUCCESS));
    }

    /* The MOCK_METHOD below would otherwise hide the base's remote overload. */
    using mocks::GMockBackendEngine::prepMemView;

    MOCK_METHOD(nixl_status_t,
                prepMemView,
                (const nixl_meta_dlist_t &dlist,
                 nixlMemViewH &view,
                 const nixl_opt_b_args_t *opt_args),
                (const, override));
    MOCK_METHOD(nixl_status_t,
                queryMem,
                (const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &responses),
                (const, override));
    MOCK_METHOD(nixl_status_t,
                estimateXferCost,
                (const nixl_xfer_op_t &operation,
                 const nixl_meta_dlist_t &local,
                 const nixl_meta_dlist_t &remote,
                 const std::string &remote_agent,
                 nixlBackendReqH *const &handle,
                 std::chrono::microseconds &duration,
                 std::chrono::microseconds &error_margin,
                 nixl_cost_t &method,
                 const nixl_opt_args_t *extra_params),
                (const, override));
};

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

/*
 * A later ON_CALL wins over the defaults set in the GMockBackendEngine c'tor.
 * Every injected action bumps `calls` so the caller can tell an error the agent
 * swallowed apart from a site the northbound path never reached at all, which
 * would otherwise look identical for the cleanup scenarios.
 */
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

/*
 * One agent plus the GMock engine backing its MOCK_BACKEND instance. The mock is
 * declared before the agent so that reverse-order member destruction tears the
 * agent down first: agent cleanup calls back into the engine.
 */
class mockAgent {
public:
    explicit mockAgent(const std::string &name) {
        agent_ = std::make_unique<nixlAgent>(name, nixlAgentConfig{});
    }

    mockAgent(const mockAgent &) = delete;
    mockAgent &
    operator=(const mockAgent &) = delete;

    nixlAgent &
    agent() {
        return *agent_;
    }

    mockErrorBackendEngine &
    gmock() {
        return gmock_;
    }

    nixl_status_t
    createBackend(nixl_b_params_t &params, nixlBackendH *&backend) {
        gmock_.SetToParams(params);
        return agent_->createBackend(mock_backend_name, params, backend);
    }

private:
    testing::NiceMock<mockErrorBackendEngine> gmock_;
    std::unique_ptr<nixlAgent> agent_;
};

struct observation {
    std::string stage;
    nixl_status_t status;
    /* Set when the scenario held up its end but the run did not, e.g. the
     * documented side effect was missing or the injected call never happened. */
    std::string failure;
};

struct scenario {
    const char *name;
    action_t action;
    injection_site_t site;
    nixl_status_t injected;
    nixl_status_t expected;
    behavior_t behavior;
};

/*
 * PASS_THROUGH_RECOVERABLE claims the agent left the peer's metadata in place,
 * which the returned status alone cannot show, so probe for it. Survival is the
 * contract rather than an oversight: the handle that saw the disconnect is
 * refused on repost from its own cached status, while the surviving registration
 * and connection info keep makeConnection callable, so a caller can rebuild
 * fresh handles without another metadata exchange.
 */
std::string
checkSideEffect(const scenario &s,
                nixlAgent &agent,
                const std::string &remote_name,
                const nixl_xfer_dlist_t &remote_descs) {
    if (s.behavior != behavior_t::PASS_THROUGH_RECOVERABLE) {
        return {};
    }

    const nixl_status_t status = agent.checkRemoteMD(remote_name, remote_descs);
    if (status == NIXL_SUCCESS) {
        return {};
    }
    /* Kept free of commas so that --csv output stays parsable. */
    return "remote metadata no longer resolvable: " +
        std::string(nixlEnumStrings::statusStr(status));
}

/*
 * Runs enough of the northbound sequence to reach the scenario's selected API.
 * Each scenario gets fresh agents so one injected failure cannot affect another.
 */
observation
runScenarioBody(const scenario &s, unsigned &calls) {
    using testing::_;
    using testing::Return;

    mockAgent local("Agent001");
    mockAgent remote("Agent002");

    nixl_b_params_t local_params, remote_params;
    nixlBackendH *local_backend = nullptr;
    nixlBackendH *remote_backend = nullptr;

    if (s.action == action_t::CREATE_BACKEND) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
    }

    nixl_status_t status = local.createBackend(local_params, local_backend);
    if (status != NIXL_SUCCESS) {
        return {"createBackend", status};
    }
    status = remote.createBackend(remote_params, remote_backend);
    if (status != NIXL_SUCCESS) {
        return {"createBackend (peer)", status};
    }

    std::vector<char> local_buf(buf_len, 0xbb);
    std::vector<char> remote_buf(buf_len, 0);

    nixlBlobDesc local_desc(reinterpret_cast<uintptr_t>(local_buf.data()), buf_len, 0);
    nixlBlobDesc remote_desc(reinterpret_cast<uintptr_t>(remote_buf.data()), buf_len, 0);

    nixl_reg_dlist_t local_reg(DRAM_SEG), remote_reg(DRAM_SEG);
    local_reg.addDesc(local_desc);
    remote_reg.addDesc(remote_desc);

    nixl_opt_args_t local_extra, remote_extra;
    local_extra.backends.push_back(local_backend);
    remote_extra.backends.push_back(remote_backend);

    if (s.action == action_t::REGISTER_MEM) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
    }

    status = local.agent().registerMem(local_reg, &local_extra);
    if (status != NIXL_SUCCESS) {
        return {"registerMem", status};
    }
    status = remote.agent().registerMem(remote_reg, &remote_extra);
    if (status != NIXL_SUCCESS) {
        return {"registerMem (peer)", status};
    }

    if (s.action == action_t::DEREGISTER_MEM) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        return {"deregisterMem", local.agent().deregisterMem(local_reg, &local_extra)};
    }

    std::string remote_md;
    status = remote.agent().getLocalMD(remote_md);
    if (status != NIXL_SUCCESS) {
        return {"getLocalMD (peer)", status};
    }

    std::string remote_name;
    if (s.action == action_t::LOAD_REMOTE_MD) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
    }
    status = local.agent().loadRemoteMD(remote_md, remote_name);
    if (status != NIXL_SUCCESS || s.action == action_t::LOAD_REMOTE_MD) {
        return {"loadRemoteMD", status};
    }

    if (s.action == action_t::MAKE_CONNECTION) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        return {"makeConnection", local.agent().makeConnection(remote_name, &local_extra)};
    }

    if (s.action == action_t::INVALIDATE_REMOTE_MD) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        return {"invalidateRemoteMD", local.agent().invalidateRemoteMD(remote_name)};
    }

    if (s.action == action_t::GET_NOTIFS) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        nixl_notifs_t notifs;
        return {"getNotifs", local.agent().getNotifs(notifs, &local_extra)};
    }

    if (s.action == action_t::GEN_NOTIF) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        return {"genNotif", local.agent().genNotif(remote_name, "mock notification", &local_extra)};
    }

    if (s.action == action_t::QUERY_MEM) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        std::vector<nixl_query_resp_t> responses;
        return {"queryMem", local.agent().queryMem(local_reg, responses, &local_extra)};
    }

    if (s.action == action_t::PREP_REMOTE_MEM_VIEW) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        nixl_remote_dlist_t remote_view(DRAM_SEG);
        remote_view.addDesc(
            nixlRemoteDesc(remote_desc.addr, remote_desc.len, remote_desc.devId, remote_name));
        nixlMemViewH view = nullptr;
        status = local.agent().prepMemView(remote_view, view, &local_extra);
        if (status == NIXL_SUCCESS) {
            local.agent().releaseMemView(view);
        }
        return {"prepMemView(remote)", status};
    }

    if (s.action == action_t::PREP_LOCAL_MEM_VIEW) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        nixl_local_dlist_t local_view(DRAM_SEG);
        local_view.addDesc(nixlBasicDesc(local_desc.addr, local_desc.len, local_desc.devId));
        nixlMemViewH view = nullptr;
        status = local.agent().prepMemView(local_view, view, &local_extra);
        if (status == NIXL_SUCCESS) {
            local.agent().releaseMemView(view);
        }
        return {"prepMemView(local)", status};
    }

    nixl_xfer_dlist_t src_dlist(DRAM_SEG), dst_dlist(DRAM_SEG);
    src_dlist.addDesc(local_desc);
    dst_dlist.addDesc(remote_desc);

    nixlXferReqH *req = nullptr;
    if (s.action == action_t::CREATE_XFER) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
    }
    status = local.agent().createXferReq(
        NIXL_WRITE, src_dlist, dst_dlist, remote_name, req, &local_extra);
    if (status != NIXL_SUCCESS) {
        return {"createXferReq", status};
    }

    if (s.action == action_t::STATUS_BEFORE_POST) {
        status = local.agent().getXferStatus(req);
        local.agent().releaseXferReq(req);
        return {"getXferStatus", status};
    }

    if (s.action == action_t::ESTIMATE_XFER_COST) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        std::chrono::microseconds duration;
        std::chrono::microseconds error_margin;
        nixl_cost_t method;
        status = local.agent().estimateXferCost(req, duration, error_margin, method, &local_extra);
        local.agent().releaseXferReq(req);
        return {"estimateXferCost", status};
    }

    if (s.action == action_t::POST_XFER) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
    } else if (s.action == action_t::CHECK_XFER) {
        ON_CALL(local.gmock(), postXfer(_, _, _, _, _, _)).WillByDefault(Return(NIXL_IN_PROG));
        applyInjection(local.gmock(), s.site, s.injected, calls);
    } else if (s.action == action_t::RELEASE_XFER) {
        ON_CALL(local.gmock(), postXfer(_, _, _, _, _, _)).WillByDefault(Return(NIXL_IN_PROG));
        ON_CALL(local.gmock(), checkXfer(_)).WillByDefault(Return(NIXL_IN_PROG));
        applyInjection(local.gmock(), s.site, s.injected, calls);
    } else if (s.action == action_t::POLL_TO_COMPLETION) {
        /* Both of these declare POST_XFER/NIXL_IN_PROG, so inject it the usual
         * way rather than stubbing postXfer directly, and stay counted. */
        applyInjection(local.gmock(), s.site, s.injected, calls);
    } else if (s.action == action_t::REPOST_ACTIVE) {
        applyInjection(local.gmock(), s.site, s.injected, calls);
        ON_CALL(local.gmock(), checkXfer(_)).WillByDefault(Return(NIXL_IN_PROG));
    }

    status = local.agent().postXferReq(req);
    if (status < 0) {
        const std::string side_effect = checkSideEffect(s, local.agent(), remote_name, dst_dlist);
        local.agent().releaseXferReq(req);
        return {"postXferReq", status, side_effect};
    }

    if (s.action == action_t::POST_XFER) {
        local.agent().releaseXferReq(req);
        return {"postXferReq", status};
    }

    if (s.action == action_t::CHECK_XFER) {
        status = local.agent().getXferStatus(req);
        const std::string side_effect = checkSideEffect(s, local.agent(), remote_name, dst_dlist);
        local.agent().releaseXferReq(req);
        return {"getXferStatus", status, side_effect};
    }

    if (s.action == action_t::RELEASE_XFER) {
        status = local.agent().releaseXferReq(req);
        if (status != NIXL_SUCCESS) {
            /*
             * releaseXferReq bails out with NIXL_ERR_REPOST_ACTIVE before it
             * deletes the handle, so the caller still owns it. The second call
             * takes the non-NIXL_IN_PROG path and frees it; not a double free.
             */
            local.agent().releaseXferReq(req);
        }
        return {"releaseXferReq", status};
    }

    if (s.action == action_t::REPOST_ACTIVE) {
        status = local.agent().postXferReq(req);
        local.agent().releaseXferReq(req);
        return {"postXferReq(repost)", status};
    }

    while (status == NIXL_IN_PROG) {
        status = local.agent().getXferStatus(req);
    }
    if (status != NIXL_SUCCESS) {
        local.agent().releaseXferReq(req);
        return {"getXferStatus", status};
    }

    if (s.action == action_t::TELEMETRY_DISABLED) {
        nixl_xfer_telem_t telemetry;
        status = local.agent().getXferTelemetry(req, telemetry);
        local.agent().releaseXferReq(req);
        return {"getXferTelemetry", status};
    }

    local.agent().releaseXferReq(req);
    return {"completed", NIXL_SUCCESS};
}

/*
 * An injected error that is never actually returned to the agent produces the
 * same clean result as one the agent deliberately discarded, so a scenario whose
 * site was not reached is reported as a failure rather than silently passing.
 * The counter outlives the agents, which call back into the engine on teardown.
 */
observation
runScenario(const scenario &s) {
    unsigned calls = 0;
    observation obs = runScenarioBody(s, calls);

    if (s.site != injection_site_t::NONE && calls == 0 && obs.failure.empty()) {
        obs.failure = "injected call was never reached";
    }
    return obs;
}

bool
passed(const scenario &s, const observation &obs) {
    return obs.status == s.expected && obs.failure.empty();
}

/*
 * Describes what the agent did with the injected status. Driven by the
 * scenario's declared behavior, not by comparing the returned status to the
 * injected one, so that a site collapsing NIXL_ERR_BACKEND to NIXL_ERR_BACKEND
 * is still reported as a collapse.
 */
std::string
verdictText(const scenario &s, const observation &obs) {
    const std::string observed_status = nixlEnumStrings::statusStr(obs.status);

    if (obs.status != s.expected) {
        return "FAIL: expected " + std::string(nixlEnumStrings::statusStr(s.expected));
    }
    if (!obs.failure.empty()) {
        return "FAIL: " + obs.failure;
    }

    switch (s.behavior) {
    case behavior_t::COMPLETED:
        return "-";
    case behavior_t::PASS_THROUGH:
        return "pass-through";
    case behavior_t::PASS_THROUGH_RECOVERABLE:
        return "pass-through + remote retained";
    case behavior_t::COLLAPSED:
        return "collapsed to " + observed_status;
    case behavior_t::TRANSFORMED:
        return "transformed to " + observed_status;
    case behavior_t::IGNORED:
        return "ignored";
    case behavior_t::PROGRESSED:
        return "progressed to " + observed_status;
    case behavior_t::AGENT_GENERATED:
        return "agent-generated";
    }
    return "(unknown)";
}

/*
 * Pads to a fixed column, but never below column_gap trailing spaces: std::setw
 * drops the separator entirely once a value grows past the field width, which
 * silently runs neighboring cells together.
 */
std::string
column(const std::string &value) {
    const size_t padding =
        column_width > value.size() + column_gap ? column_width - value.size() : column_gap;
    return value + std::string(padding, ' ');
}

} // namespace

int
main(int argc, char **argv) {
    testing::InitGoogleMock(&argc, argv);

#ifdef MOCK_PLUGIN_DIR
    /* Point the plugin manager at the built mock plugin unless told otherwise. */
    setenv("NIXL_PLUGIN_DIR", MOCK_PLUGIN_DIR, 0);
#endif

    bool csv = false;
    bool list = false;
    std::string site_filter;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--csv") {
            csv = true;
        } else if (arg == "--list") {
            list = true;
        } else if (site_filter.empty() && arg.rfind("--", 0) != 0) {
            site_filter = arg;
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--csv] [--list] [scenario-name|injection-site]\n";
            return 2;
        }
    }

    const std::vector<scenario> scenarios{
        {"baseline",
         action_t::COMPLETE,
         injection_site_t::NONE,
         NIXL_SUCCESS,
         NIXL_SUCCESS,
         behavior_t::COMPLETED},

        {"create.getConnInfo.invalidParam",
         action_t::CREATE_BACKEND,
         injection_site_t::GET_CONN_INFO,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_INVALID_PARAM,
         behavior_t::PASS_THROUGH},
        {"create.getConnInfo.backend",
         action_t::CREATE_BACKEND,
         injection_site_t::GET_CONN_INFO,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::PASS_THROUGH},
        {"create.connect.invalidParam",
         action_t::CREATE_BACKEND,
         injection_site_t::CONNECT,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_INVALID_PARAM,
         behavior_t::PASS_THROUGH},
        {"create.connect.backend",
         action_t::CREATE_BACKEND,
         injection_site_t::CONNECT,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::PASS_THROUGH},

        {"register.registerMem.invalidParam",
         action_t::REGISTER_MEM,
         injection_site_t::REGISTER_MEM,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_BACKEND,
         behavior_t::COLLAPSED},
        {"register.registerMem.notFound",
         action_t::REGISTER_MEM,
         injection_site_t::REGISTER_MEM,
         NIXL_ERR_NOT_FOUND,
         NIXL_ERR_BACKEND,
         behavior_t::COLLAPSED},
        {"register.registerMem.backend",
         action_t::REGISTER_MEM,
         injection_site_t::REGISTER_MEM,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::COLLAPSED},
        {"register.loadLocalMD.invalidParam",
         action_t::REGISTER_MEM,
         injection_site_t::LOAD_LOCAL_MD,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_BACKEND,
         behavior_t::COLLAPSED},
        {"register.loadLocalMD.backend",
         action_t::REGISTER_MEM,
         injection_site_t::LOAD_LOCAL_MD,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::COLLAPSED},
        {"register.getPublicData.invalidParam",
         action_t::REGISTER_MEM,
         injection_site_t::GET_PUBLIC_DATA,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_BACKEND,
         behavior_t::COLLAPSED},
        {"register.getPublicData.backend",
         action_t::REGISTER_MEM,
         injection_site_t::GET_PUBLIC_DATA,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::COLLAPSED},

        {"metadata.loadRemoteConnInfo.invalidParam",
         action_t::LOAD_REMOTE_MD,
         injection_site_t::LOAD_REMOTE_CONN_INFO,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_INVALID_PARAM,
         behavior_t::PASS_THROUGH},
        {"metadata.loadRemoteConnInfo.notAllowed",
         action_t::LOAD_REMOTE_MD,
         injection_site_t::LOAD_REMOTE_CONN_INFO,
         NIXL_ERR_NOT_ALLOWED,
         NIXL_ERR_NOT_ALLOWED,
         behavior_t::PASS_THROUGH},
        {"metadata.loadRemoteConnInfo.notSupported",
         action_t::LOAD_REMOTE_MD,
         injection_site_t::LOAD_REMOTE_CONN_INFO,
         NIXL_ERR_NOT_SUPPORTED,
         NIXL_ERR_BACKEND,
         behavior_t::COLLAPSED},
        {"metadata.loadRemoteMD.invalidParam",
         action_t::LOAD_REMOTE_MD,
         injection_site_t::LOAD_REMOTE_MD,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_INVALID_PARAM,
         behavior_t::PASS_THROUGH},
        {"metadata.loadRemoteMD.notFound",
         action_t::LOAD_REMOTE_MD,
         injection_site_t::LOAD_REMOTE_MD,
         NIXL_ERR_NOT_FOUND,
         NIXL_ERR_NOT_FOUND,
         behavior_t::PASS_THROUGH},

        {"connection.connect.backend",
         action_t::MAKE_CONNECTION,
         injection_site_t::CONNECT,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::PASS_THROUGH},
        {"connection.connect.notFound",
         action_t::MAKE_CONNECTION,
         injection_site_t::CONNECT,
         NIXL_ERR_NOT_FOUND,
         NIXL_ERR_NOT_FOUND,
         behavior_t::PASS_THROUGH},

        {"transfer.prepXfer.invalidParam",
         action_t::CREATE_XFER,
         injection_site_t::PREP_XFER,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_INVALID_PARAM,
         behavior_t::PASS_THROUGH},
        {"transfer.prepXfer.notFound",
         action_t::CREATE_XFER,
         injection_site_t::PREP_XFER,
         NIXL_ERR_NOT_FOUND,
         NIXL_ERR_NOT_FOUND,
         behavior_t::PASS_THROUGH},
        {"transfer.postXfer.backend",
         action_t::POST_XFER,
         injection_site_t::POST_XFER,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::PASS_THROUGH},
        {"transfer.postXfer.remoteDisconnect",
         action_t::POST_XFER,
         injection_site_t::POST_XFER,
         NIXL_ERR_REMOTE_DISCONNECT,
         NIXL_ERR_REMOTE_DISCONNECT,
         behavior_t::PASS_THROUGH_RECOVERABLE},
        {"transfer.checkXfer.backend",
         action_t::CHECK_XFER,
         injection_site_t::CHECK_XFER,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::PASS_THROUGH},
        {"transfer.checkXfer.remoteDisconnect",
         action_t::CHECK_XFER,
         injection_site_t::CHECK_XFER,
         NIXL_ERR_REMOTE_DISCONNECT,
         NIXL_ERR_REMOTE_DISCONNECT,
         behavior_t::PASS_THROUGH_RECOVERABLE},
        {"transfer.checkXfer.canceled",
         action_t::CHECK_XFER,
         injection_site_t::CHECK_XFER,
         NIXL_ERR_CANCELED,
         NIXL_ERR_CANCELED,
         behavior_t::PASS_THROUGH},
        {"transfer.releaseReqH.backend",
         action_t::RELEASE_XFER,
         injection_site_t::RELEASE_REQ,
         NIXL_ERR_BACKEND,
         NIXL_ERR_REPOST_ACTIVE,
         behavior_t::TRANSFORMED},

        {"memView.remote.notSupported",
         action_t::PREP_REMOTE_MEM_VIEW,
         injection_site_t::PREP_REMOTE_MEM_VIEW,
         NIXL_ERR_NOT_SUPPORTED,
         NIXL_ERR_NOT_SUPPORTED,
         behavior_t::PASS_THROUGH},
        {"memView.remote.backend",
         action_t::PREP_REMOTE_MEM_VIEW,
         injection_site_t::PREP_REMOTE_MEM_VIEW,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::PASS_THROUGH},
        {"memView.local.notSupported",
         action_t::PREP_LOCAL_MEM_VIEW,
         injection_site_t::PREP_LOCAL_MEM_VIEW,
         NIXL_ERR_NOT_SUPPORTED,
         NIXL_ERR_NOT_SUPPORTED,
         behavior_t::PASS_THROUGH},
        {"memView.local.backend",
         action_t::PREP_LOCAL_MEM_VIEW,
         injection_site_t::PREP_LOCAL_MEM_VIEW,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::PASS_THROUGH},

        {"notification.getNotifs.backend",
         action_t::GET_NOTIFS,
         injection_site_t::GET_NOTIFS,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::PASS_THROUGH},
        {"notification.getNotifs.invalidParam",
         action_t::GET_NOTIFS,
         injection_site_t::GET_NOTIFS,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_INVALID_PARAM,
         behavior_t::PASS_THROUGH},
        {"notification.genNotif.backend",
         action_t::GEN_NOTIF,
         injection_site_t::GEN_NOTIF,
         NIXL_ERR_BACKEND,
         NIXL_ERR_BACKEND,
         behavior_t::PASS_THROUGH},
        {"notification.genNotif.notFound",
         action_t::GEN_NOTIF,
         injection_site_t::GEN_NOTIF,
         NIXL_ERR_NOT_FOUND,
         NIXL_ERR_NOT_FOUND,
         behavior_t::PASS_THROUGH},

        {"optional.queryMem.notSupported",
         action_t::QUERY_MEM,
         injection_site_t::QUERY_MEM,
         NIXL_ERR_NOT_SUPPORTED,
         NIXL_ERR_NOT_SUPPORTED,
         behavior_t::PASS_THROUGH},
        {"optional.queryMem.invalidParam",
         action_t::QUERY_MEM,
         injection_site_t::QUERY_MEM,
         NIXL_ERR_INVALID_PARAM,
         NIXL_ERR_INVALID_PARAM,
         behavior_t::PASS_THROUGH},
        {"optional.estimateXferCost.notSupported",
         action_t::ESTIMATE_XFER_COST,
         injection_site_t::ESTIMATE_XFER_COST,
         NIXL_ERR_NOT_SUPPORTED,
         NIXL_ERR_NOT_SUPPORTED,
         behavior_t::PASS_THROUGH},
        {"optional.estimateXferCost.mismatch",
         action_t::ESTIMATE_XFER_COST,
         injection_site_t::ESTIMATE_XFER_COST,
         NIXL_ERR_MISMATCH,
         NIXL_ERR_MISMATCH,
         behavior_t::PASS_THROUGH},

        {"cleanup.deregisterMem.backend",
         action_t::DEREGISTER_MEM,
         injection_site_t::DEREGISTER_MEM,
         NIXL_ERR_BACKEND,
         NIXL_SUCCESS,
         behavior_t::IGNORED},
        {"cleanup.unloadMD.backend",
         action_t::DEREGISTER_MEM,
         injection_site_t::UNLOAD_MD,
         NIXL_ERR_BACKEND,
         NIXL_SUCCESS,
         behavior_t::IGNORED},
        {"cleanup.disconnect.backend",
         action_t::INVALIDATE_REMOTE_MD,
         injection_site_t::DISCONNECT,
         NIXL_ERR_BACKEND,
         NIXL_SUCCESS,
         behavior_t::IGNORED},

        {"control.pollToCompletion",
         action_t::POLL_TO_COMPLETION,
         injection_site_t::POST_XFER,
         NIXL_IN_PROG,
         NIXL_SUCCESS,
         behavior_t::PROGRESSED},
        {"control.repostActive",
         action_t::REPOST_ACTIVE,
         injection_site_t::POST_XFER,
         NIXL_IN_PROG,
         NIXL_ERR_REPOST_ACTIVE,
         behavior_t::AGENT_GENERATED},
        {"control.statusBeforePost",
         action_t::STATUS_BEFORE_POST,
         injection_site_t::NONE,
         NIXL_SUCCESS,
         NIXL_ERR_NOT_POSTED,
         behavior_t::AGENT_GENERATED},
        {"control.telemetryDisabled",
         action_t::TELEMETRY_DISABLED,
         injection_site_t::NONE,
         NIXL_SUCCESS,
         NIXL_ERR_NO_TELEMETRY,
         behavior_t::AGENT_GENERATED},
    };

    if (list) {
        for (const auto &s : scenarios) {
            std::cout << s.name << "\n";
        }
        return 0;
    }

    if (csv) {
        std::cout << "sb site,injected,agent stopped at,agent returned,verdict\n";
    } else {
        std::cout << "Injecting southbound errors into " << mock_backend_name << "\n\n"
                  << column("sb site") << column("injected") << column("agent stopped at")
                  << column("agent returned") << "verdict\n"
                  << std::string(4 * column_width + longest_verdict, '-') << "\n";
    }

    size_t selected = 0;
    size_t failures = 0;
    for (const auto &s : scenarios) {
        if (!site_filter.empty() && site_filter != s.name && site_filter != siteName(s.site)) {
            continue;
        }
        selected++;

        const observation obs = runScenario(s);
        const std::string injected_status =
            s.site == injection_site_t::NONE ? "-" : nixlEnumStrings::statusStr(s.injected);
        const std::string observed_status = nixlEnumStrings::statusStr(obs.status);
        const std::string verdict = verdictText(s, obs);
        failures += !passed(s, obs);

        if (csv) {
            std::cout << siteName(s.site) << "," << injected_status << "," << obs.stage << ","
                      << observed_status << "," << verdict << "\n";
        } else {
            std::cout << column(siteName(s.site)) << column(injected_status) << column(obs.stage)
                      << column(observed_status) << verdict << "\n";
        }
    }

    if (selected == 0) {
        std::cerr << "No scenario or injection site matches '" << site_filter << "'\n";
        return 2;
    }

    if (!csv) {
        std::cout << "\n" << selected << " scenarios, " << failures << " failures\n";
    }
    return failures == 0 ? 0 : 1;
}
