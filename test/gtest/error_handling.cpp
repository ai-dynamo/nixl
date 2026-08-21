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
#include <gtest/gtest.h>
#include <nixl_types.h>
#include "common.h"
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

    m_priv->releaseXferReq(req_handle);

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
    testXfer<TestType::LOAD_REMOTE_THEN_FAIL, NIXL_WRITE>();
    testXfer<TestType::LOAD_REMOTE_THEN_FAIL, NIXL_READ>();
}

TEST_P(TestErrorHandling, XferThenFail) {
    const LogIgnoreGuard lig(expected_log);
    testXfer<TestType::XFER_THEN_FAIL, NIXL_WRITE>();
    testXfer<TestType::XFER_THEN_FAIL, NIXL_READ>();
}

TEST_P(TestErrorHandling, XferFailRestore) {
    const LogIgnoreGuard lig(expected_log);
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
    testStaleXferHandle();
}

TEST_P(TestErrorHandling, MetadataReloadKeepsHandlesValid) {
    testMetadataReloadKeepsHandlesValid();
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
            initiator.makeXferReq(NIXL_WRITE, local_side, indices, remote_side, indices, req);
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
    ASSERT_EQ(initiator.makeXferReq(NIXL_WRITE, local_side, indices, remote_side, indices, req),
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
