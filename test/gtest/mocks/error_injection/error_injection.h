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
 * The southbound error-injection model: which call sites a failure can be aimed
 * at, which northbound sequence a scenario drives, what the agent is expected to
 * do with the injected status, and the mock engine that carries the injection.
 * The scenario table lives in error_scenarios.cpp and the formatting in
 * scenario_report.h, so a driver only has to walk errorScenarios().
 */
#ifndef TEST_GTEST_MOCKS_ERROR_INJECTION_ERROR_INJECTION_H
#define TEST_GTEST_MOCKS_ERROR_INJECTION_ERROR_INJECTION_H

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "nixl.h"
#include "mocks/gmock_engine.h"

namespace mocks::error_injection {

/* Matches the shared_library name in test/gtest/mocks/meson.build. */
constexpr const char *mock_backend_name = "MOCK_BACKEND";
constexpr size_t buf_len = 256;

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
 * reaches these overrides.
 */
class mockErrorBackendEngine : public mocks::GMockBackendEngine {
public:
    mockErrorBackendEngine();

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

/*
 * One agent plus the GMock engine backing its MOCK_BACKEND instance. The mock is
 * declared before the agent so that reverse-order member destruction tears the
 * agent down first: agent cleanup calls back into the engine.
 */
class mockAgent {
public:
    explicit mockAgent(const std::string &name);

    mockAgent(const mockAgent &) = delete;
    mockAgent &
    operator=(const mockAgent &) = delete;

    nixlAgent &
    agent();

    mockErrorBackendEngine &
    gmock();

    nixl_status_t
    createBackend(nixl_b_params_t &params, nixlBackendH *&backend);

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

const char *
siteName(injection_site_t site);

/*
 * A later ON_CALL wins over the defaults set in the GMockBackendEngine c'tor.
 * The installed action bumps `calls`, separating an error the agent swallowed
 * from a site it never reached: zero once the sequence ends means a matching
 * status was accidental. The mock increments it after this returns, so `calls`
 * must outlive the engine, whose teardown can still bump it.
 */
void
applyInjection(mockErrorBackendEngine &gmock,
               injection_site_t site,
               nixl_status_t status,
               unsigned &calls);

/* The scenario table, shared by every driver so the two cannot drift apart. */
const std::vector<scenario> &
errorScenarios();

} // namespace mocks::error_injection

#endif // TEST_GTEST_MOCKS_ERROR_INJECTION_ERROR_INJECTION_H
