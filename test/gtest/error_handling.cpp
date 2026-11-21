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

#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <map>
#include <netdb.h>
#include <nixl_types.h>
#include <set>
#include <sys/socket.h>
#include "absl/cleanup/cleanup.h"
#include "common.h"
#include "common/hw_info.h"
#include "common/scoped_fd.h"
#include "nixl.h"
#ifdef HAVE_UCX_BACKEND
#include "ucx_utils.h"
#endif

namespace gtest {

constexpr std::chrono::seconds wait_timeout{100};

namespace nixl {
    constexpr const char* ucx_err_handling_mode_key  = "ucx_error_handling_mode";
    constexpr const char* ucx_err_handling_mode_peer = "peer";

    static nixlBackendH *
    createUcxBackend(nixlAgent &agent,
                     const std::string &backend_name,
                     size_t num_workers,
                     size_t num_threads) {
        std::vector<nixl_backend_t> plugins;
        nixl_status_t status = agent.getAvailPlugins(plugins);
        EXPECT_EQ(status, NIXL_SUCCESS);
        auto it = std::find(plugins.begin(), plugins.end(), backend_name);
        EXPECT_NE(it, plugins.end()) << "UCX plugin not found";

        nixl_b_params_t params;
        nixl_mem_list_t mems;
        status = agent.getPluginParams(*it, mems, params);
        EXPECT_EQ(NIXL_SUCCESS, status);

        nixlBackendH* backend_handle = nullptr;
        EXPECT_EQ(ucx_err_handling_mode_peer, params[ucx_err_handling_mode_key]);
        params["num_workers"] = std::to_string(num_workers);
        params["num_threads"] = std::to_string(num_threads);
        // If threadpool is configured always force split
        params["split_batch_size"] = "0";
        status = agent.createBackend(*it, params, backend_handle);
        EXPECT_EQ(NIXL_SUCCESS, status);
        EXPECT_NE(nullptr, backend_handle);
        return backend_handle;
    }

    template <typename DListT, typename DescT> void
    fillRegList(DListT &dlist, DescT &desc, const std::vector<std::byte>& data)
    {
        desc.addr  = reinterpret_cast<uintptr_t>(data.data());
        desc.len   = data.size();
        desc.devId = 0;
        dlist.addDesc(desc);
    }
} // namespace nixl

class TestErrorHandling : public nixl_test_t {
    class Agent {
        struct MemDesc {
            MemDesc() : m_dlist(DRAM_SEG), m_desc() {}

            void init(nixlBackendH* backend) {
                m_params = { .backends = {backend} };
                // Each testXfer call reuses the fixture's Agent and may call
                // init() multiple times. Reset m_dlist so a second init() does
                // not register the same memory range twice.
                m_dlist.clear();
                nixl::fillRegList(m_dlist, m_desc, m_data);
            }

            void fillData() {
                std::fill(m_data.begin(), m_data.end(), std::byte(std::rand()));
            }

            static constexpr size_t m_data_size = 256;
            std::vector<std::byte>  m_data = std::vector<std::byte>(m_data_size);
            nixl_opt_args_t         m_params;
            nixl_reg_dlist_t        m_dlist;
            nixlBlobDesc            m_desc;
        };

    public:
        void
        init(const std::string &name,
             const std::string &backend_name,
             bool use_prog_thread,
             size_t num_workers,
             size_t num_threads);

        void
        destroy();
        void fillRegList(nixl_xfer_dlist_t& dlist, nixlBasicDesc& desc) const;
        std::string getLocalMD() const;
        void loadRemoteMD(const std::string& remote_name);
        nixl_status_t createXferReq(const nixl_xfer_op_t& op,
                                    nixl_xfer_dlist_t& sReq_descs,
                                    nixl_xfer_dlist_t& rReq_descs,
                                    nixlXferReqH*& req_handle) const;
        nixl_status_t
        postXferReq(nixlXferReqH *req_handle) const;
        nixl_status_t
        releaseXferReq(nixlXferReqH *req_handle) const;

        nixlAgent *
        getAgent() const {
            return m_priv.get();
        }

        nixl_status_t
        waitForCompletion(nixlXferReqH *req_handle, Agent &peer, nixl_notifs_t &peer_notifs);
        nixl_status_t
        waitForNotif(const std::string &expectedNotif, nixl_notifs_t &notifs);
        void fillData();
        bool dataCmp(const Agent& other) const;

    private:
        std::string m_name;
        bool m_progThread = false;
        nixlBackendH*              m_backend = nullptr;
        std::unique_ptr<nixlAgent> m_priv    = nullptr;
        std::string                m_MetaRemote;
        MemDesc                    m_mem;
    };

protected:
    enum class TestType {
        BASIC_XFER,
        LOAD_REMOTE_THEN_FAIL,
        XFER_THEN_FAIL,
        XFER_FAIL_RESTORE,
        FAIL_AFTER_POST,
    };

    TestErrorHandling();
    template<TestType test_type, enum nixl_xfer_op_t op> void testXfer();
    void
    testNotifAfterFail();
    void
    testStalePreppedDlist();
    void
    testStaleXferHandle();
    void
    testMetadataReloadKeepsHandlesValid();
    void
    testReconnectAfterConnectionLoss();
    void
    excludeEfaTransports();

private:
    template<TestType test_type>
    bool
    failBeforePost(size_t iter);
    template<TestType test_type>
    bool
    failAfterPost(size_t iter);
    template<TestType test_type>
    bool
    isFailure(size_t iter);
    template<TestType test_type> size_t numIter();
    void
    exchangeMetaData();
    template<TestType test_type>
    std::variant<nixlXferReqH *, nixl_status_t>
    postXfer(enum nixl_xfer_op_t op, size_t iter);

    ScopedEnv    m_env;
    Agent        m_Initiator;
    Agent        m_Target;
    std::string  m_backend_name;
    const bool progThread_;
    size_t numWorkers_;
    size_t numThreads_;
};

void
TestErrorHandling::Agent::init(const std::string &name,
                               const std::string &backend_name,
                               bool use_prog_thread,
                               size_t num_workers,
                               size_t num_threads) {
    nixlAgentConfig cfg;
    m_progThread = use_prog_thread;
    cfg.useProgThread = use_prog_thread;
    // TODO: remove once access to dedicated workers is properly serialized.
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    m_priv = std::make_unique<nixlAgent>(name, cfg);
    // At the moment, only UCX backend is tested for error handling support.
    m_backend = nixl::createUcxBackend(*m_priv, backend_name, num_workers, num_threads);
    m_mem.init(m_backend);
    m_mem.fillData();

    // Ignore EFA hardware mismatch warning
    const gtest::LogIgnoreGuard lig_efa_warn(
        "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");

    EXPECT_EQ(NIXL_SUCCESS, m_priv->registerMem(m_mem.m_dlist, &m_mem.m_params));
}

void
TestErrorHandling::Agent::destroy() {
    m_priv->deregisterMem(m_mem.m_dlist, &m_mem.m_params);
    m_priv->invalidateRemoteMD(m_MetaRemote);
    m_priv.reset();
    m_backend = nullptr;
}

void TestErrorHandling::Agent::fillRegList(nixl_xfer_dlist_t& dlist,
                                        nixlBasicDesc& desc) const {
    nixl::fillRegList(dlist, desc, m_mem.m_data);
}

std::string TestErrorHandling::Agent::getLocalMD() const {
    std::string meta;
    EXPECT_EQ(NIXL_SUCCESS, m_priv->getLocalMD(meta));
    return meta;
}

void TestErrorHandling::Agent::loadRemoteMD(const std::string& remote_name) {
    EXPECT_EQ(NIXL_SUCCESS, m_priv->loadRemoteMD(remote_name, m_MetaRemote))
        << "Agent " << m_name << " failed to load remote metadata";
}

nixl_status_t
TestErrorHandling::Agent::createXferReq(const nixl_xfer_op_t& op,
                                     nixl_xfer_dlist_t& sReq_descs,
                                     nixl_xfer_dlist_t& rReq_descs,
                                     nixlXferReqH*& req_handle) const {
    nixl_opt_args_t extra_params = { .backends = {m_backend} };
    extra_params.notif = "notification";
    return m_priv->createXferReq(op, sReq_descs, rReq_descs, m_MetaRemote,
                                 req_handle, &extra_params);
}

nixl_status_t
TestErrorHandling::Agent::postXferReq(nixlXferReqH *req_handle) const {
    return m_priv->postXferReq(req_handle);
}

nixl_status_t
TestErrorHandling::Agent::releaseXferReq(nixlXferReqH *req_handle) const {
    return m_priv->releaseXferReq(req_handle);
}

nixl_status_t
TestErrorHandling::Agent::waitForCompletion(nixlXferReqH *req_handle,
                                            Agent &peer,
                                            nixl_notifs_t &peer_notifs) {
    const auto deadline = std::chrono::steady_clock::now() + wait_timeout;
    nixl_status_t status;

    do {
        status = m_priv->getXferStatus(req_handle);
        EXPECT_NE(NIXL_ERR_NOT_POSTED, status);
        if (!peer.m_progThread && peer.m_priv != nullptr) {
            const nixl_status_t peer_status = peer.m_priv->getNotifs(peer_notifs);
            EXPECT_EQ(NIXL_SUCCESS, peer_status);
            if (peer_status != NIXL_SUCCESS) {
                break;
            }
        }
    } while ((status == NIXL_IN_PROG) && (std::chrono::steady_clock::now() < deadline));

    EXPECT_EQ(NIXL_SUCCESS, m_priv->releaseXferReq(req_handle));

    return status;
}

nixl_status_t
TestErrorHandling::Agent::waitForNotif(const std::string &expectedNotif, nixl_notifs_t &notifs) {
    const auto deadline = std::chrono::steady_clock::now() + wait_timeout;

    while (notifs[m_MetaRemote].empty()) {
        const nixl_status_t status = m_priv->getNotifs(notifs);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return NIXL_IN_PROG;
        }
    }

    EXPECT_EQ(1u, notifs[m_MetaRemote].size());
    EXPECT_EQ(expectedNotif, notifs[m_MetaRemote].front());
    return NIXL_SUCCESS;
}

void TestErrorHandling::Agent::fillData() {
    m_mem.fillData();
}

bool TestErrorHandling::Agent::dataCmp(const TestErrorHandling::Agent& other) const {
    return m_mem.m_data == other.m_mem.m_data;
}

TestErrorHandling::TestErrorHandling()
    : m_backend_name(GetParam().backendName),
      progThread_(GetParam().progressThreadEnabled),
      numWorkers_(GetParam().numWorkers),
      numThreads_(GetParam().numThreads) {
    m_env.addVar("UCX_RC_TIMEOUT", "100us");
    m_env.addVar("UCX_RC_RETRY_COUNT", "4");
    m_env.addVar("UCX_UD_TIMEOUT", "3s");
    m_env.addVar("NIXL_PLUGIN_DIR", std::string(BUILD_DIR) + "/src/plugins/ucx");
}

void
TestErrorHandling::excludeEfaTransports() {
    if (::nixl::hwInfo::instance().numEfaDevices > 0) {
        // TODO: Remove once upgraded to UCX v1.24 (plus any follow-up fixes):
        // - EFA SRD peer-failure reporting (https://github.com/openucx/ucx/pull/12224):
        //   uct_srd does not report peer failure when a CTL_REQ or AM completes
        //   with error (send_op->ep is NULL).
        // - Wireup reconnect fix (https://github.com/openucx/ucx/pull/12227):
        //   remote_ep_id mismatch on a wireup reply after reconnect.
        // - ud_verbs on EFA detects peer failure only after UCX_UD_TIMEOUT and
        //   lacks CQ notifications as aux wireup.
        m_env.addVar("UCX_TLS", "^srd,ud_v,ud_v:aux");
    }
}

template<TestErrorHandling::TestType test_type, enum nixl_xfer_op_t op>
void TestErrorHandling::testXfer() {
    const std::string initiator_name = "initiator";
    const std::string target_name = "target";
    m_Initiator.init(initiator_name, m_backend_name, progThread_, numWorkers_, numThreads_);
    m_Target.init(target_name, m_backend_name, progThread_, numWorkers_, numThreads_);

    exchangeMetaData();

    for (size_t i = 0; i < numIter<test_type>(); ++i) {
        nixl_status_t status;
        nixl_notifs_t target_notifs;
        auto result = postXfer<test_type>(op, i);
        if (std::holds_alternative<nixl_status_t>(result)) {
            // Transfer completed immediately
            status = std::get<nixl_status_t>(result);
        } else {
            // Transfer was posted, wait for completion
            nixlXferReqH *req_handle = std::get<nixlXferReqH *>(result);
            status = m_Initiator.waitForCompletion(req_handle, m_Target, target_notifs);
        }

        if (isFailure<test_type>(i)) {
            if (failBeforePost<test_type>(i)) {
                EXPECT_EQ(status, NIXL_ERR_REMOTE_DISCONNECT);
            } else {
                EXPECT_TRUE((status == NIXL_ERR_REMOTE_DISCONNECT) || (status == NIXL_SUCCESS));
            }

            if (test_type == TestType::XFER_FAIL_RESTORE) {
                // postXferReq/getXferStatus no longer invalidate remote metadata
                // on disconnect: the consumer must do it explicitly before
                // re-registering the failed agent.
                EXPECT_EQ(m_Initiator.getAgent()->invalidateRemoteMD(target_name), NIXL_SUCCESS);
                m_Target.init(target_name, m_backend_name, progThread_, numWorkers_, numThreads_);
                exchangeMetaData();
            }
        } else {
            EXPECT_EQ(NIXL_SUCCESS, status);
            EXPECT_EQ(NIXL_SUCCESS, m_Target.waitForNotif("notification", target_notifs));
            EXPECT_TRUE(m_Target.dataCmp(m_Initiator));

            // Update the data for the next iteration
            m_Initiator.fillData();
            m_Target.fillData();
        }
    }

    switch (test_type) {
    case TestType::BASIC_XFER:
    case TestType::XFER_FAIL_RESTORE:
        m_Target.destroy();
        m_Initiator.destroy();
        return;
    case TestType::LOAD_REMOTE_THEN_FAIL:
    case TestType::XFER_THEN_FAIL:
    case TestType::FAIL_AFTER_POST:
        m_Initiator.destroy();
        return;
    }
}

template<TestErrorHandling::TestType test_type>
bool
TestErrorHandling::failBeforePost(size_t iter) {
    switch (test_type) {
    case TestType::BASIC_XFER:
        return false;
    case TestType::LOAD_REMOTE_THEN_FAIL:
        return iter == 0;
    case TestType::XFER_THEN_FAIL:
    case TestType::XFER_FAIL_RESTORE:
        return iter == 1;
    case TestType::FAIL_AFTER_POST:
        return false;
    }
}

template<TestErrorHandling::TestType test_type>
bool
TestErrorHandling::failAfterPost(size_t iter) {
    return (test_type == TestType::FAIL_AFTER_POST) && (iter == 1);
}

template<TestErrorHandling::TestType test_type>
bool
TestErrorHandling::isFailure(size_t iter) {
    return failBeforePost<test_type>(iter) || failAfterPost<test_type>(iter);
}

template<TestErrorHandling::TestType test_type>
size_t
TestErrorHandling::numIter() {
    switch (test_type) {
    case TestType::BASIC_XFER:
    case TestType::LOAD_REMOTE_THEN_FAIL:
        return 1;
    case TestType::XFER_THEN_FAIL:
    case TestType::FAIL_AFTER_POST:
        return 2;
    case TestType::XFER_FAIL_RESTORE:
        return 3;
    }
}

void TestErrorHandling::exchangeMetaData() {
    m_Initiator.loadRemoteMD(m_Target.getLocalMD());
    m_Target.loadRemoteMD(m_Initiator.getLocalMD());
}

template<TestErrorHandling::TestType test_type>
std::variant<nixlXferReqH *, nixl_status_t>
TestErrorHandling::postXfer(enum nixl_xfer_op_t op, size_t iter) {
    EXPECT_TRUE(op == NIXL_WRITE || op == NIXL_READ);

    nixlBasicDesc sReq_src;
    nixl_xfer_dlist_t sReq_descs(DRAM_SEG);
    m_Initiator.fillRegList(sReq_descs, sReq_src);

    nixlBasicDesc rReq_dst;
    nixl_xfer_dlist_t rReq_descs(DRAM_SEG);
    m_Target.fillRegList(rReq_descs, rReq_dst);

    nixlXferReqH* req_handle;
    nixl_status_t status = m_Initiator.createXferReq(op, sReq_descs, rReq_descs, req_handle);
    EXPECT_EQ(NIXL_SUCCESS, status)
        << "createXferReq failed with unexpected error: " << nixlEnumStrings::statusStr(status);

    if (failBeforePost<test_type>(iter)) {
        m_Target.destroy();
    }

    status = m_Initiator.postXferReq(req_handle);

    if (failAfterPost<test_type>(iter)) {
        m_Target.destroy();
    }

    if (isFailure<test_type>(iter) && (status == NIXL_ERR_REMOTE_DISCONNECT)) {
        // postXferReq does not take ownership of the request on failure (it only
        // invalidates the remote data), so release the handle here to avoid
        // leaking it and its backend request handle. The caller only gets the
        // status, so it cannot release it itself.
        m_Initiator.releaseXferReq(req_handle);
        return status;
    }

    EXPECT_LE(0, status) << "status: " << nixlEnumStrings::statusStr(status);
    return req_handle;
}

namespace {
    const std::string expected_log =
        "postXferReq: remote agent 'target' was disconnected after transfer request creation";

} // namespace

TEST_P(TestErrorHandling, BasicXfer) {
    testXfer<TestType::BASIC_XFER, NIXL_WRITE>();
    testXfer<TestType::BASIC_XFER, NIXL_READ>();
}

TEST_P(TestErrorHandling, LoadRemoteThenFail) {
    const LogIgnoreGuard lig(expected_log);
    excludeEfaTransports();
    testXfer<TestType::LOAD_REMOTE_THEN_FAIL, NIXL_WRITE>();
    testXfer<TestType::LOAD_REMOTE_THEN_FAIL, NIXL_READ>();
}

TEST_P(TestErrorHandling, XferThenFail) {
    const LogIgnoreGuard lig(expected_log);
    excludeEfaTransports();
    testXfer<TestType::XFER_THEN_FAIL, NIXL_WRITE>();
    testXfer<TestType::XFER_THEN_FAIL, NIXL_READ>();
}

TEST_P(TestErrorHandling, XferFailRestore) {
    const LogIgnoreGuard lig(expected_log);
    excludeEfaTransports();
    testXfer<TestType::XFER_FAIL_RESTORE, NIXL_WRITE>();
    testXfer<TestType::XFER_FAIL_RESTORE, NIXL_READ>();
}

TEST_P(TestErrorHandling, XferPostThenFail) {
    testXfer<TestType::FAIL_AFTER_POST, NIXL_WRITE>();
    testXfer<TestType::FAIL_AFTER_POST, NIXL_READ>();
}

TEST_P(TestErrorHandling, StalePreppedDlistRejectedAfterReregistration) {
    testStalePreppedDlist();
}

TEST_P(TestErrorHandling, StaleXferHandleRejectedAfterReregistration) {
    excludeEfaTransports();
    testStaleXferHandle();
}

TEST_P(TestErrorHandling, MetadataReloadKeepsHandlesValid) {
    testMetadataReloadKeepsHandlesValid();
}

TEST_P(TestErrorHandling, ReconnectAfterConnectionLoss) {
    testReconnectAfterConnectionLoss();
}

// Transfer handles and prepped dlists bind to a remote registration generation: they
// must be rejected after that generation is invalidated and the agent re-registers.
void
TestErrorHandling::testStalePreppedDlist() {
    const std::string initiator_name = "initiator";
    const std::string target_name = "target";
    m_Initiator.init(initiator_name, m_backend_name, progThread_, numWorkers_, numThreads_);
    m_Target.init(target_name, m_backend_name, progThread_, numWorkers_, numThreads_);
    exchangeMetaData();

    nixl_xfer_dlist_t l_descs(DRAM_SEG), r_descs(DRAM_SEG);
    nixlBasicDesc l_desc, r_desc;
    m_Initiator.fillRegList(l_descs, l_desc);
    m_Target.fillRegList(r_descs, r_desc);

    nixlAgent &initiator = *m_Initiator.getAgent();
    nixlDlistH *local_side = nullptr, *remote_side = nullptr;
    ASSERT_EQ(initiator.prepXferDlist(l_descs, local_side), NIXL_SUCCESS);
    ASSERT_EQ(initiator.prepXferDlist(target_name, r_descs, remote_side), NIXL_SUCCESS);

    const std::vector<int> indices = {0};
    const auto make_xfer = [&]() {
        nixlXferReqH *req = nullptr;
        const nixl_status_t ret =
            initiator.makeXferReq(NIXL_WRITE, *local_side, indices, *remote_side, indices, req);
        if (req) {
            EXPECT_EQ(initiator.releaseXferReq(req), NIXL_SUCCESS);
        }
        return ret;
    };
    ASSERT_EQ(make_xfer(), NIXL_SUCCESS);

    ASSERT_EQ(initiator.invalidateRemoteMD(target_name), NIXL_SUCCESS);
    m_Initiator.loadRemoteMD(m_Target.getLocalMD());

    {
        const LogIgnoreGuard lig("invalidated or re-registered after prepped xfer request");
        EXPECT_EQ(make_xfer(), NIXL_ERR_NOT_FOUND);
    }

    // Release the stale dlist before replacing it, then re-prep must recover
    EXPECT_EQ(initiator.releasedDlistH(remote_side), NIXL_SUCCESS);
    remote_side = nullptr;
    ASSERT_EQ(initiator.prepXferDlist(target_name, r_descs, remote_side), NIXL_SUCCESS);
    EXPECT_EQ(make_xfer(), NIXL_SUCCESS);
    EXPECT_EQ(initiator.releasedDlistH(local_side), NIXL_SUCCESS);
    EXPECT_EQ(initiator.releasedDlistH(remote_side), NIXL_SUCCESS);
    m_Target.destroy();
    m_Initiator.destroy();
}

void
TestErrorHandling::testStaleXferHandle() {
    const std::string initiator_name = "initiator";
    const std::string target_name = "target";
    m_Initiator.init(initiator_name, m_backend_name, progThread_, numWorkers_, numThreads_);
    m_Target.init(target_name, m_backend_name, progThread_, numWorkers_, numThreads_);
    exchangeMetaData();

    nixl_xfer_dlist_t l_descs(DRAM_SEG), r_descs(DRAM_SEG);
    nixlBasicDesc l_desc, r_desc;
    m_Initiator.fillRegList(l_descs, l_desc);
    m_Target.fillRegList(r_descs, r_desc);

    nixlAgent &initiator = *m_Initiator.getAgent();
    nixlXferReqH *req = nullptr;
    ASSERT_EQ(initiator.createXferReq(NIXL_WRITE, l_descs, r_descs, target_name, req),
              NIXL_SUCCESS);

    ASSERT_EQ(initiator.invalidateRemoteMD(target_name), NIXL_SUCCESS);
    m_Initiator.loadRemoteMD(m_Target.getLocalMD());

    {
        const LogIgnoreGuard lig("invalidated or re-registered after transfer request creation");
        std::chrono::microseconds duration, err_margin;
        nixl_cost_t method;
        EXPECT_EQ(initiator.estimateXferCost(req, duration, err_margin, method),
                  NIXL_ERR_NOT_FOUND);
        EXPECT_EQ(initiator.postXferReq(req), NIXL_ERR_NOT_FOUND);
    }
    // Releasing the rejected handle must keep working so callers can reclaim it
    EXPECT_EQ(initiator.releaseXferReq(req), NIXL_SUCCESS);

    // A fresh request against the new registration posts and completes
    ASSERT_EQ(initiator.createXferReq(NIXL_WRITE, l_descs, r_descs, target_name, req),
              NIXL_SUCCESS);
    const nixl_status_t status = initiator.postXferReq(req);
    ASSERT_TRUE(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
    nixl_notifs_t target_notifs;
    EXPECT_EQ(m_Initiator.waitForCompletion(req, m_Target, target_notifs), NIXL_SUCCESS);
    m_Target.destroy();
    m_Initiator.destroy();
}

// Reloading byte-identical metadata is an intentional refresh: the registration
// stays alive and handles created against it remain valid.
void
TestErrorHandling::testMetadataReloadKeepsHandlesValid() {
    const std::string initiator_name = "initiator";
    const std::string target_name = "target";
    m_Initiator.init(initiator_name, m_backend_name, progThread_, numWorkers_, numThreads_);
    m_Target.init(target_name, m_backend_name, progThread_, numWorkers_, numThreads_);
    exchangeMetaData();

    nixl_xfer_dlist_t l_descs(DRAM_SEG), r_descs(DRAM_SEG);
    nixlBasicDesc l_desc, r_desc;
    m_Initiator.fillRegList(l_descs, l_desc);
    m_Target.fillRegList(r_descs, r_desc);

    nixlAgent &initiator = *m_Initiator.getAgent();
    nixlDlistH *local_side = nullptr, *remote_side = nullptr;
    ASSERT_EQ(initiator.prepXferDlist(l_descs, local_side), NIXL_SUCCESS);
    ASSERT_EQ(initiator.prepXferDlist(target_name, r_descs, remote_side), NIXL_SUCCESS);
    nixlXferReqH *req = nullptr;
    const std::vector<int> indices = {0};
    ASSERT_EQ(initiator.makeXferReq(NIXL_WRITE, *local_side, indices, *remote_side, indices, req),
              NIXL_SUCCESS);

    // Unchanged re-broadcast of the same metadata
    m_Initiator.loadRemoteMD(m_Target.getLocalMD());

    const nixl_status_t status = initiator.postXferReq(req);
    ASSERT_TRUE(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
    nixl_notifs_t target_notifs;
    EXPECT_EQ(m_Initiator.waitForCompletion(req, m_Target, target_notifs), NIXL_SUCCESS);
    EXPECT_EQ(initiator.releasedDlistH(local_side), NIXL_SUCCESS);
    EXPECT_EQ(initiator.releasedDlistH(remote_side), NIXL_SUCCESS);
    m_Target.destroy();
    m_Initiator.destroy();
}

namespace {

    struct tcpSocket {
        int fd;
        std::string local;
        std::string peer;
    };

    // "host:port" of an INET socket address, empty for any other address.
    std::string
    addressString(const sockaddr_storage &addr, socklen_t len) {
        if ((addr.ss_family != AF_INET) && (addr.ss_family != AF_INET6)) {
            return {};
        }

        char host[NI_MAXHOST];
        char port[NI_MAXSERV];
        if (getnameinfo(reinterpret_cast<const sockaddr *>(&addr),
                        len,
                        host,
                        sizeof(host),
                        port,
                        sizeof(port),
                        NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
            return {};
        }
        return std::string(host) + ":" + port;
    }

    // The /proc/self/fd link ("socket:[inode]") of an fd, which stays unique while the
    // socket is open; empty if it cannot be read.
    std::string
    fdLink(int fd) {
        std::error_code ec;
        const auto link = std::filesystem::read_symlink("/proc/self/fd/" + std::to_string(fd), ec);
        return ec ? std::string() : link.string();
    }

    // The TCP sockets of this process that are connected to a peer, keyed by their fd link.
    // Each fd is read through a duplicate, so that its type, addresses and link all come
    // from one socket even if the fd is closed and reused meanwhile.
    std::map<std::string, tcpSocket>
    connectedTcpSockets() {
        std::map<std::string, tcpSocket> sockets;
        for (const auto &entry : std::filesystem::directory_iterator("/proc/self/fd")) {
            const int fd = std::stoi(entry.path().filename().string());
            const ::nixl::scopedFd pinned(fcntl(fd, F_DUPFD_CLOEXEC, 0));
            if (!pinned.valid()) {
                continue;
            }
            const int pfd = pinned.get();

            int type = 0;
            socklen_t type_len = sizeof(type);
            if ((getsockopt(pfd, SOL_SOCKET, SO_TYPE, &type, &type_len) != 0) ||
                (type != SOCK_STREAM)) {
                continue;
            }

            sockaddr_storage local{};
            sockaddr_storage peer{};
            socklen_t local_len = sizeof(local);
            socklen_t peer_len = sizeof(peer);
            if ((getsockname(pfd, reinterpret_cast<sockaddr *>(&local), &local_len) != 0) ||
                (getpeername(pfd, reinterpret_cast<sockaddr *>(&peer), &peer_len) != 0)) {
                continue;
            }

            tcpSocket sock{fd, addressString(local, local_len), addressString(peer, peer_len)};
            const std::string link = fdLink(pfd);
            if (!link.empty() && !sock.local.empty() && !sock.peer.empty()) {
                sockets.emplace(link, std::move(sock));
            }
        }
        return sockets;
    }

    // Shuts down both ends of the TCP connections that were opened since `before` at both
    // ends, between two sockets of this process, as a network failure would. In this test,
    // those are the connections between its two agents. Listening sockets have no peer and
    // stay open, so the agents keep running with unchanged metadata and still accept new
    // connections.
    size_t
    shutdownInProcessTcpConnections(const std::map<std::string, tcpSocket> &before) {
        std::map<std::string, tcpSocket> opened = connectedTcpSockets();
        std::erase_if(opened, [&](const auto &entry) { return before.count(entry.first) != 0; });
        std::set<std::pair<std::string, std::string>> ends;
        for (const auto &[link, sock] : opened) {
            ends.emplace(sock.local, sock.peer);
        }

        size_t count = 0;
        for (const auto &[link, sock] : opened) {
            if (ends.count({sock.peer, sock.local}) == 0) {
                continue;
            }

            // The fd may have been closed, and reused for another socket, since it was
            // listed. A duplicate pins whatever it refers to now; shut it down only if that
            // is still the listed socket.
            const ::nixl::scopedFd pinned(fcntl(sock.fd, F_DUPFD_CLOEXEC, 0));
            if (pinned.valid() && (fdLink(pinned.get()) == link) &&
                (shutdown(pinned.get(), SHUT_RDWR) == 0)) {
                ++count;
            }
        }
        return count;
    }

} // namespace

// When the connection to an agent fails while the agent keeps running, its metadata stays
// loaded. Once the backend knows about the failure, loading the agent's metadata again,
// in full or in part, returns NIXL_ERR_REMOTE_DISCONNECT and changes nothing, and so does
// makeConnection(). Invalidating the metadata disconnects the backend, and loading the full
// metadata again then connects anew.
void
TestErrorHandling::testReconnectAfterConnectionLoss() {
    // TCP only, so that shutting down the agents' TCP connections fails the endpoints.
    // cuda_copy satisfies CUDA-enabled UCX validation. The test reads, as
    // UcxRequestRelease does over TCP.
    m_env.addVar("UCX_TLS", "tcp,cuda_copy");
    const LogIgnoreGuard lig(expected_log);
    const LogIgnoreGuard lig_failed("knows that its connection to agent 'target' has failed");
    const LogIgnoreGuard lig_stale("prepped descriptor lists must be re-created");
    // How makeConnection() reports it when an endpoint fails between its check and connect().
    const LogIgnoreGuard lig_connect(
        "failed on backend 'UCX' with status NIXL_ERR_REMOTE_DISCONNECT");
    const auto sockets_before = connectedTcpSockets();
    const std::string target_name = "target";
    m_Initiator.init("initiator", m_backend_name, progThread_, numWorkers_, numThreads_);
    m_Target.init(target_name, m_backend_name, progThread_, numWorkers_, numThreads_);
    exchangeMetaData();
    nixlAgent &initiator = *m_Initiator.getAgent();
    nixlAgent &target = *m_Target.getAgent();

    nixl_xfer_dlist_t local_descs(DRAM_SEG), remote_descs(DRAM_SEG);
    nixlBasicDesc local_desc, remote_desc;
    m_Initiator.fillRegList(local_descs, local_desc);
    m_Target.fillRegList(remote_descs, remote_desc);

    // The target's metadata in full, and in part without connection info.
    const std::string target_md = m_Target.getLocalMD();
    nixl_reg_dlist_t target_reg(DRAM_SEG);
    target_reg.addDesc(nixlBlobDesc(remote_desc.addr, remote_desc.len, remote_desc.devId));
    std::string target_partial;
    ASSERT_EQ(NIXL_SUCCESS, target.getLocalPartialMD(target_reg, target_partial, nullptr));

    // The same with a second target buffer as a new descriptor.
    std::vector<std::byte> extra_buf(256);
    nixl_reg_dlist_t extra_reg(DRAM_SEG);
    extra_reg.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(extra_buf.data()), extra_buf.size(), 0));
    ASSERT_EQ(NIXL_SUCCESS, target.registerMem(extra_reg));
    // Deregister while extra_buf is alive, also when an assertion ends the test early.
    absl::Cleanup deregister_extra = [&] {
        EXPECT_EQ(NIXL_SUCCESS, target.deregisterMem(extra_reg));
    };
    std::string target_md_extra, target_partial_extra;
    ASSERT_EQ(NIXL_SUCCESS, target.getLocalMD(target_md_extra));
    ASSERT_EQ(NIXL_SUCCESS, target.getLocalPartialMD(extra_reg, target_partial_extra, nullptr));
    const nixl_xfer_dlist_t extra_descs = extra_reg.trim();
    const nixl_xfer_dlist_t no_descs(DRAM_SEG);

    // Dlists prepped before the failure, and a request made from them.
    nixlDlistH *local_side = nullptr;
    nixlDlistH *remote_side = nullptr;
    absl::Cleanup release_dlists = [&] {
        for (nixlDlistH *dlist : {local_side, remote_side}) {
            if (dlist != nullptr) {
                EXPECT_EQ(NIXL_SUCCESS, initiator.releasedDlistH(dlist));
            }
        }
    };
    ASSERT_EQ(NIXL_SUCCESS, initiator.prepXferDlist(local_descs, local_side));
    ASSERT_EQ(NIXL_SUCCESS, initiator.prepXferDlist(target_name, remote_descs, remote_side));
    const std::vector<int> first{0};
    const auto make_prepped_req = [&]() {
        nixlXferReqH *req = nullptr;
        const nixl_status_t status =
            initiator.makeXferReq(NIXL_READ, local_side, first, remote_side, first, req);
        if (status == NIXL_SUCCESS) {
            EXPECT_EQ(NIXL_SUCCESS, initiator.releaseXferReq(req));
        }
        return status;
    };

    const auto read = [this]() {
        nixl_xfer_dlist_t local_descs(DRAM_SEG), remote_descs(DRAM_SEG);
        nixlBasicDesc local_desc, remote_desc;
        m_Initiator.fillRegList(local_descs, local_desc);
        m_Target.fillRegList(remote_descs, remote_desc);

        nixlXferReqH *req = nullptr;
        nixl_status_t status = m_Initiator.createXferReq(NIXL_READ, local_descs, remote_descs, req);
        if (status != NIXL_SUCCESS) {
            return status;
        }

        status = m_Initiator.postXferReq(req);
        if (status < 0) {
            // A failed post leaves the request with the caller.
            EXPECT_EQ(NIXL_SUCCESS, m_Initiator.releaseXferReq(req));
            return status;
        }

        nixl_notifs_t target_notifs;
        status = m_Initiator.waitForCompletion(req, m_Target, target_notifs);
        if (status == NIXL_SUCCESS) {
            EXPECT_EQ(NIXL_SUCCESS, m_Target.waitForNotif("notification", target_notifs));
            EXPECT_TRUE(m_Target.dataCmp(m_Initiator));
            m_Target.fillData();
        }
        return status;
    };

    ASSERT_EQ(NIXL_SUCCESS, read());
    ASSERT_EQ(NIXL_SUCCESS, make_prepped_req());

    ASSERT_GT(shutdownInProcessTcpConnections(sockets_before), 0u);
    // No transfer is posted: the endpoints fail when their workers next make progress, on
    // the progress thread or, without one, in getNotifs().
    const auto deadline = std::chrono::steady_clock::now() + wait_timeout;
    nixl_status_t status;
    do {
        nixl_notifs_t notifs;
        EXPECT_EQ(NIXL_SUCCESS, initiator.getNotifs(notifs));
        status = initiator.makeConnection(target_name);
    } while ((status == NIXL_SUCCESS) && (std::chrono::steady_clock::now() < deadline));
    ASSERT_EQ(NIXL_ERR_REMOTE_DISCONNECT, status);

    // Loading the metadata again, in full or in part, with or without the new descriptor,
    // reports the failure and changes nothing: the metadata stays loaded without the new
    // descriptor, and the prepped dlists still bind to it.
    std::string name;
    EXPECT_EQ(NIXL_ERR_REMOTE_DISCONNECT, initiator.loadRemoteMD(target_md, name));
    EXPECT_EQ(NIXL_ERR_REMOTE_DISCONNECT, initiator.loadRemoteMD(target_md_extra, name));
    EXPECT_EQ(NIXL_ERR_REMOTE_DISCONNECT, initiator.loadRemoteMD(target_partial, name));
    EXPECT_EQ(NIXL_ERR_REMOTE_DISCONNECT, initiator.loadRemoteMD(target_partial_extra, name));
    EXPECT_EQ(NIXL_SUCCESS, initiator.checkRemoteMD(target_name, no_descs));
    EXPECT_EQ(NIXL_ERR_NOT_FOUND, initiator.checkRemoteMD(target_name, extra_descs));
    EXPECT_EQ(NIXL_SUCCESS, make_prepped_req());
    EXPECT_EQ(NIXL_ERR_REMOTE_DISCONNECT, read());

    // Invalidating disconnects the backend, and loading the full metadata again connects
    // anew. Dlists prepped before stay bound to the invalidated metadata.
    EXPECT_EQ(NIXL_SUCCESS, initiator.invalidateRemoteMD(target_name));
    EXPECT_EQ(NIXL_SUCCESS, initiator.loadRemoteMD(target_md, name));
    EXPECT_EQ(NIXL_SUCCESS, initiator.makeConnection(target_name));
    EXPECT_EQ(NIXL_ERR_NOT_FOUND, make_prepped_req());
    EXPECT_EQ(NIXL_SUCCESS, read());

    std::move(release_dlists).Invoke();
    std::move(deregister_extra).Invoke();
    m_Target.destroy();
    m_Initiator.destroy();
}

#ifdef HAVE_UCX_BACKEND
TEST_P(TestErrorHandling, ErrorCallbackMarksEndpointFailedWithoutClosingIt) {
    std::vector<std::string> devices;
    const size_t num_workers = GetParam().numWorkers;
    const bool use_progress_thread = GetParam().progressThreadEnabled;
    nixlUcxContext consumer_context(
        devices, use_progress_thread, num_workers, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 1);
    nixlUcxContext producer_context(
        devices, use_progress_thread, num_workers, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 1);
    nixlUcxWorker consumer(consumer_context, UCP_ERR_HANDLING_MODE_PEER);
    nixlUcxWorker producer(producer_context, UCP_ERR_HANDLING_MODE_PEER);
    std::string producer_address = producer.epAddr();
    auto endpoint = consumer.connect(producer_address.data());
    ASSERT_NE(endpoint, nullptr);

    const ucp_ep_h native_endpoint = endpoint->getEp();
    endpoint->err_cb(native_endpoint, UCS_ERR_CONNECTION_RESET);

    EXPECT_EQ(endpoint->checkTxState(), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(endpoint->getEp(), native_endpoint);
}
#endif

NIXL_INSTANTIATE_TEST(ucx, TestErrorHandling, "UCX", true, 1, 0, "");
NIXL_INSTANTIATE_TEST(ucx_no_pt, TestErrorHandling, "UCX", false, 1, 0, "");
NIXL_INSTANTIATE_TEST(ucx_threadpool, TestErrorHandling, "UCX", true, 2, 1, "");
NIXL_INSTANTIATE_TEST(ucx_threadpool_no_pt, TestErrorHandling, "UCX", false, 2, 1, "");

} // namespace gtest
