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

#include <algorithm>
#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include <nixl_types.h>
#include "common.h"
#include "nixl.h"

namespace gtest {
namespace nixl {
    constexpr const char *uccl_backend_name = "UCCL";

    static nixlBackendH *
    createUcclBackend(nixlAgent &agent) {
        std::vector<nixl_backend_t> plugins;
        nixl_status_t status = agent.getAvailPlugins(plugins);
        EXPECT_EQ(status, NIXL_SUCCESS);
        auto it = std::find(plugins.begin(), plugins.end(), uccl_backend_name);
        EXPECT_NE(it, plugins.end()) << "UCCL plugin not found";

        nixl_b_params_t params;
        nixl_mem_list_t mems;
        status = agent.getPluginParams(*it, mems, params);
        EXPECT_EQ(NIXL_SUCCESS, status);
        params["in_python"] = "0";
        nixlBackendH *backend_handle = nullptr;
        status = agent.createBackend(*it, params, backend_handle);
        EXPECT_EQ(NIXL_SUCCESS, status);
        EXPECT_NE(nullptr, backend_handle);
        return backend_handle;
    }

    template<typename DListT, typename DescT>
    void
    fillRegList(DListT &dlist, DescT &desc, const std::vector<std::byte> &data) {
        desc.addr = reinterpret_cast<uintptr_t>(data.data());
        desc.len = data.size();
        desc.devId = 0;
        dlist.addDesc(desc);
    }
} // namespace nixl

class TestUcclBackend : public testing::Test {
    class Agent {
        struct MemDesc {
            MemDesc() : m_dlist(DRAM_SEG), m_desc() {}

            void
            init(nixlBackendH *backend) {
                m_params = {.backends = {backend}};
                nixl::fillRegList(m_dlist, m_desc, m_data);
            }

            void
            fillData() {
                std::fill(m_data.begin(), m_data.end(), std::byte(std::rand()));
            }

            static constexpr size_t m_data_size = 256;
            std::vector<std::byte> m_data = std::vector<std::byte>(m_data_size);
            nixl_opt_args_t m_params;
            nixl_reg_dlist_t m_dlist;
            nixlBlobDesc m_desc;
        };

    public:
        void
        init(const std::string &name);

        void
        destroy();
        void
        fillRegList(nixl_xfer_dlist_t &dlist, nixlBasicDesc &desc) const;
        std::string
        getLocalMD() const;
        void
        loadRemoteMD(const std::string &remote_name);
        nixl_status_t
        createXferReq(const nixl_xfer_op_t &op,
                      nixl_xfer_dlist_t &sReq_descs,
                      nixl_xfer_dlist_t &rReq_descs,
                      nixlXferReqH *&req_handle) const;
        nixl_status_t
        postXferReq(nixlXferReqH *req_handle) const;
        nixl_status_t
        waitForCompletion(nixlXferReqH *req_handle);
        nixl_status_t
        waitForNotif(const std::string &expectedNotif);
        void
        fillData();
        bool
        dataCmp(const Agent &other) const;

    private:
        std::string m_name;
        nixlBackendH *m_backend = nullptr;
        std::unique_ptr<nixlAgent> m_priv = nullptr;
        std::string m_MetaRemote;
        MemDesc m_mem;
    };

protected:
    enum class TestType {
        BASIC_XFER,
        LOAD_REMOTE_THEN_FAIL,
        XFER_THEN_FAIL,
        XFER_FAIL_RESTORE,
        FAIL_AFTER_POST,
    };

    TestUcclBackend();
    template<TestType test_type, enum nixl_xfer_op_t op>
    void
    testXfer();

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
    template<TestType test_type>
    size_t
    numIter();
    void
    exchangeMetaData();
    template<TestType test_type>
    std::variant<nixlXferReqH *, nixl_status_t>
    postXfer(enum nixl_xfer_op_t op, size_t iter);

    ScopedEnv m_env;
    Agent m_Initiator;
    Agent m_Target;
    std::string m_backend_name;
};

void
TestUcclBackend::Agent::init(const std::string &name) {
    m_priv = std::make_unique<nixlAgent>(name, nixlAgentConfig(true));
    // Create UCCL backend for testing
    m_backend = nixl::createUcclBackend(*m_priv);
    m_mem.init(m_backend);
    m_mem.fillData();

    EXPECT_EQ(NIXL_SUCCESS, m_priv->registerMem(m_mem.m_dlist, &m_mem.m_params));
}

void
TestUcclBackend::Agent::destroy() {
    m_priv->deregisterMem(m_mem.m_dlist, &m_mem.m_params);
    m_priv->invalidateRemoteMD(m_MetaRemote);
    m_priv.reset();
    m_backend = nullptr;
}

void
TestUcclBackend::Agent::fillRegList(nixl_xfer_dlist_t &dlist, nixlBasicDesc &desc) const {
    nixl::fillRegList(dlist, desc, m_mem.m_data);
}

std::string
TestUcclBackend::Agent::getLocalMD() const {
    std::string meta;
    EXPECT_EQ(NIXL_SUCCESS, m_priv->getLocalMD(meta));
    return meta;
}

void
TestUcclBackend::Agent::loadRemoteMD(const std::string &remote_name) {
    EXPECT_EQ(NIXL_SUCCESS, m_priv->loadRemoteMD(remote_name, m_MetaRemote))
        << "Agent " << m_name << " failed to load remote metadata";
}

nixl_status_t
TestUcclBackend::Agent::createXferReq(const nixl_xfer_op_t &op,
                                      nixl_xfer_dlist_t &sReq_descs,
                                      nixl_xfer_dlist_t &rReq_descs,
                                      nixlXferReqH *&req_handle) const {
    nixl_opt_args_t extra_params = {.backends = {m_backend}};
    extra_params.notifMsg = "notification";
    extra_params.hasNotif = true;
    return m_priv->createXferReq(
        op, sReq_descs, rReq_descs, m_MetaRemote, req_handle, &extra_params);
}

nixl_status_t
TestUcclBackend::Agent::postXferReq(nixlXferReqH *req_handle) const {
    return m_priv->postXferReq(req_handle);
}

nixl_status_t
TestUcclBackend::Agent::waitForCompletion(nixlXferReqH *req_handle) {
    nixl_status_t status;

    do {
        status = m_priv->getXferStatus(req_handle);
        EXPECT_NE(NIXL_ERR_NOT_POSTED, status);
    } while (status == NIXL_IN_PROG);

    m_priv->releaseXferReq(req_handle);

    return status;
}

nixl_status_t
TestUcclBackend::Agent::waitForNotif(const std::string &expectedNotif) {
    nixl_notifs_t notif_map;

    do {
        EXPECT_EQ(NIXL_SUCCESS, m_priv->getNotifs(notif_map));
    } while (notif_map.empty());

    std::vector<std::string> notifs = notif_map[m_MetaRemote];
    EXPECT_EQ(1u, notifs.size());
    EXPECT_EQ(expectedNotif, notifs.front());
    return NIXL_SUCCESS;
}

void
TestUcclBackend::Agent::fillData() {
    m_mem.fillData();
}

bool
TestUcclBackend::Agent::dataCmp(const TestUcclBackend::Agent &other) const {
    return m_mem.m_data == other.m_mem.m_data;
}

TestUcclBackend::TestUcclBackend() {
    m_backend_name = "UCCL";
    m_env.addVar("NIXL_PLUGIN_DIR", std::string(BUILD_DIR) + "/src/plugins/uccl");
}

template<TestUcclBackend::TestType test_type, enum nixl_xfer_op_t op>
void
TestUcclBackend::testXfer() {
    if (op == NIXL_READ) {
        m_env.addVar("UCCL_RCMODE", "1");
    }
    const std::string initiator_name = "initiator";
    const std::string target_name = "target";

    m_Initiator.init(initiator_name);
    m_Target.init(target_name);

    exchangeMetaData();

    for (size_t i = 0; i < numIter<test_type>(); ++i) {
        nixl_status_t status;
        auto result = postXfer<test_type>(op, i);
        if (std::holds_alternative<nixl_status_t>(result)) {
            // Transfer completed immediately
            status = std::get<nixl_status_t>(result);
        } else {
            // Transfer was posted, wait for completion
            nixlXferReqH *req_handle = std::get<nixlXferReqH *>(result);
            status = m_Initiator.waitForCompletion(req_handle);
        }

        if (isFailure<test_type>(i)) {
            if (failBeforePost<test_type>(i)) {
                EXPECT_EQ(status, NIXL_ERR_REMOTE_DISCONNECT);
            } else {
                EXPECT_TRUE((status == NIXL_ERR_REMOTE_DISCONNECT) || (status == NIXL_SUCCESS));
            }

            if (test_type == TestType::XFER_FAIL_RESTORE) {
                m_Target.init(target_name);
                exchangeMetaData();
            }
        } else {
            EXPECT_EQ(NIXL_SUCCESS, status);
            EXPECT_EQ(NIXL_SUCCESS, m_Target.waitForNotif("notification"));
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

template<TestUcclBackend::TestType test_type>
bool
TestUcclBackend::failBeforePost(size_t iter) {
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

template<TestUcclBackend::TestType test_type>
bool
TestUcclBackend::failAfterPost(size_t iter) {
    return (test_type == TestType::FAIL_AFTER_POST) && (iter == 1);
}

template<TestUcclBackend::TestType test_type>
bool
TestUcclBackend::isFailure(size_t iter) {
    return failBeforePost<test_type>(iter) || failAfterPost<test_type>(iter);
}

template<TestUcclBackend::TestType test_type>
size_t
TestUcclBackend::numIter() {
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

void
TestUcclBackend::exchangeMetaData() {
    m_Initiator.loadRemoteMD(m_Target.getLocalMD());
    m_Target.loadRemoteMD(m_Initiator.getLocalMD());
}

template<TestUcclBackend::TestType test_type>
std::variant<nixlXferReqH *, nixl_status_t>
TestUcclBackend::postXfer(enum nixl_xfer_op_t op, size_t iter) {
    EXPECT_TRUE(op == NIXL_WRITE || op == NIXL_READ);

    nixlBasicDesc sReq_src;
    nixl_xfer_dlist_t sReq_descs(DRAM_SEG);
    m_Initiator.fillRegList(sReq_descs, sReq_src);

    nixlBasicDesc rReq_dst;
    nixl_xfer_dlist_t rReq_descs(DRAM_SEG);
    m_Target.fillRegList(rReq_descs, rReq_dst);

    nixlXferReqH *req_handle;
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
        // failed handle destroyed on post
        return status;
    }

    EXPECT_LE(0, status) << "status: " << nixlEnumStrings::statusStr(status);
    return req_handle;
}

TEST_F(TestUcclBackend, BasicXfer) {
    testXfer<TestType::BASIC_XFER, NIXL_READ>();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    testXfer<TestType::BASIC_XFER, NIXL_WRITE>();
}

// TODO: Enable failure tests after hardening corner cases
// TEST_F(TestUcclBackend, LoadRemoteThenFail) {
//     testXfer<TestType::LOAD_REMOTE_THEN_FAIL, NIXL_WRITE>();
//     testXfer<TestType::LOAD_REMOTE_THEN_FAIL, NIXL_READ>();
// }

// TEST_F(TestUcclBackend, XferThenFail) {
//     testXfer<TestType::XFER_THEN_FAIL, NIXL_WRITE>();
//     testXfer<TestType::XFER_THEN_FAIL, NIXL_READ>();
// }

// TEST_F(TestUcclBackend, XferFailRestore) {
//     testXfer<TestType::XFER_FAIL_RESTORE, NIXL_WRITE>();
//     testXfer<TestType::XFER_FAIL_RESTORE, NIXL_READ>();
// }

// TEST_F(TestUcclBackend, XferPostThenFail) {
//     testXfer<TestType::FAIL_AFTER_POST, NIXL_WRITE>();
//     testXfer<TestType::FAIL_AFTER_POST, NIXL_READ>();
// }

} // namespace gtest
