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
 *
 * The scenarios live in error_scenarios.cpp, the mock engine and the injection
 * itself in mock_error_engine.cpp, and the output formatting in
 * scenario_report.cpp. What remains here is the northbound sequence.
 */

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "mocks/error_injection/error_injection.h"
#include "mocks/error_injection/scenario_report.h"

/* A standalone driver, so pulling the model into scope keeps the sequence readable. */
using namespace mocks::error_injection;

namespace {

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

    const std::vector<scenario> &scenarios = errorScenarios();

    if (list) {
        for (const auto &s : scenarios) {
            std::cout << s.name << "\n";
        }
        return 0;
    }

    printHeader(csv);

    size_t selected = 0;
    size_t failures = 0;
    for (const auto &s : scenarios) {
        if (!site_filter.empty() && site_filter != s.name && site_filter != siteName(s.site)) {
            continue;
        }
        selected++;

        const observation obs = runScenario(s);
        failures += !passed(s, obs);
        printRow(s, obs, csv);
    }

    if (selected == 0) {
        std::cerr << "No scenario or injection site matches '" << site_filter << "'\n";
        return 2;
    }

    printSummary(selected, failures, csv);
    return failures == 0 ? 0 : 1;
}
