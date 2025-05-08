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
#include <gtest/gtest.h>
#include <nixl_types.h>
#include "nixl.h"

namespace gtest {
namespace nixl {
    static nixlBackendH* createUcxBackend(nixlAgent& agent)
    {
        std::vector<nixl_backend_t> plugins;
        nixl_status_t status = agent.getAvailPlugins(plugins);
        EXPECT_EQ(status, NIXL_SUCCESS);
        auto it = std::find(plugins.begin(), plugins.end(), "UCX");
        EXPECT_NE(it, plugins.end()) << "UCX plugin not found";

        nixl_b_params_t params;
        nixl_mem_list_t mems;
        status = agent.getPluginParams("UCX", mems, params);
        EXPECT_EQ(NIXL_SUCCESS, status);

        nixlBackendH* backend_handle = nullptr;
        status = agent.createBackend("UCX", params, backend_handle);
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

class TestUcxBackend : public testing::Test {
    class Agent {
        struct MemDesc {
            MemDesc() : m_dlist(DRAM_SEG), m_desc() {}

            void init(nixlBackendH* backend) {
                m_params = { .backends = {backend} };
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
        void init(const std::string& name);
        void destroy();
        void fillRegList(nixl_xfer_dlist_t& dlist, nixlBasicDesc& desc) const;
        std::string getLocalMD() const;
        void loadRemoteMD(const std::string& remote_name);
        nixl_status_t createXferReq(const nixl_xfer_op_t& op,
                                    nixl_xfer_dlist_t& sReq_descs,
                                    nixl_xfer_dlist_t& rReq_descs,
                                    nixlXferReqH*& req_handle) const;
        nixl_status_t postXferReq(nixlXferReqH* req_handle) const;
        nixl_status_t waitForCompletion(nixlXferReqH* req_handle);
        nixl_status_t waitForNotif(const std::string& expectedNotif);
        void fillData();
        bool dataCmp(const Agent& other) const;

    private:
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
    };

    TestUcxBackend();
    ~TestUcxBackend();
    template<TestType test_type, enum nixl_xfer_op_t op> void testXfer();

private:
    template<TestType test_type> bool isFailure(size_t iter);
    template<TestType test_type> size_t numIter();
    void exchangeMetaData();
    nixlXferReqH* postXfer(enum nixl_xfer_op_t op, bool target_failure);

    std::string m_plugin_dir_backup;
    Agent       m_Initiator;
    Agent       m_Target;
};

void TestUcxBackend::Agent::init(const std::string& name) {
    m_priv    = std::make_unique<nixlAgent>(name, nixlAgentConfig(true));
    m_backend = nixl::createUcxBackend(*m_priv);
    m_mem.init(m_backend);
    m_mem.fillData();

    EXPECT_EQ(NIXL_SUCCESS,
              m_priv->registerMem(m_mem.m_dlist, &m_mem.m_params));
}

void TestUcxBackend::Agent::destroy() {
    m_MetaRemote.clear();
    m_priv->deregisterMem(m_mem.m_dlist, &m_mem.m_params);
    m_priv.reset();
}

void TestUcxBackend::Agent::fillRegList(nixl_xfer_dlist_t& dlist,
                                        nixlBasicDesc& desc) const {
    nixl::fillRegList(dlist, desc, m_mem.m_data);
}

std::string TestUcxBackend::Agent::getLocalMD() const {
    std::string meta;
    EXPECT_EQ(NIXL_SUCCESS, m_priv->getLocalMD(meta));
    return meta;
}

void TestUcxBackend::Agent::loadRemoteMD(const std::string& remote_name) {
    EXPECT_EQ(NIXL_SUCCESS, m_priv->loadRemoteMD(remote_name, m_MetaRemote));
}

nixl_status_t
TestUcxBackend::Agent::createXferReq(const nixl_xfer_op_t& op,
                                     nixl_xfer_dlist_t& sReq_descs,
                                     nixl_xfer_dlist_t& rReq_descs,
                                     nixlXferReqH*& req_handle) const {
    nixl_opt_args_t extra_params = { .backends = {m_backend} };
    extra_params.notifMsg        = "notification";
    extra_params.hasNotif        = true;
    return m_priv->createXferReq(op, sReq_descs, rReq_descs, m_MetaRemote,
                                 req_handle, &extra_params);
}

nixl_status_t
TestUcxBackend::Agent::postXferReq(nixlXferReqH* req_handle) const {
    return m_priv->postXferReq(req_handle);
}

nixl_status_t
TestUcxBackend::Agent::waitForCompletion(nixlXferReqH* req_handle) {
    nixl_status_t status;

    do {
        status = m_priv->getXferStatus(req_handle);
    } while (status == NIXL_IN_PROG);

    m_priv->releaseXferReq(req_handle);
    return status;
}

nixl_status_t
TestUcxBackend::Agent::waitForNotif(const std::string& expectedNotif) {
    nixl_notifs_t notif_map;

    do {
        EXPECT_EQ(NIXL_SUCCESS, m_priv->getNotifs(notif_map));
    } while (notif_map.empty());

    std::vector<std::string> notifs = notif_map[m_MetaRemote];
    EXPECT_EQ(1,             notifs.size());
    EXPECT_EQ(expectedNotif, notifs.front());
    return NIXL_SUCCESS;
}

void TestUcxBackend::Agent::fillData() {
    m_mem.fillData();
}

bool TestUcxBackend::Agent::dataCmp(const TestUcxBackend::Agent& other) const {
    return m_mem.m_data == other.m_mem.m_data;
}

TestUcxBackend::TestUcxBackend() {
    // Set up test environment
    m_plugin_dir_backup = getenv("NIXL_PLUGIN_DIR");

    // Load plugins from build directory
    std::string plugin_dir = std::string(BUILD_DIR) + "/src/plugins/ucx";
    setenv("NIXL_PLUGIN_DIR", plugin_dir.c_str(), 1);

    std::cout << "set NIXL_PLUGIN_DIR: " << getenv("NIXL_PLUGIN_DIR")
              << std::endl;
}

TestUcxBackend::~TestUcxBackend() {
    setenv("NIXL_PLUGIN_DIR", m_plugin_dir_backup.c_str(), 1);
    std::cout << "restore NIXL_PLUGIN_DIR: " << getenv("NIXL_PLUGIN_DIR")
              << std::endl;
}

template<TestUcxBackend::TestType test_type, enum nixl_xfer_op_t op>
void TestUcxBackend::testXfer() {
    m_Initiator.init("initiator");
    m_Target.init("target");

    exchangeMetaData();

    for (size_t i = 0; i < numIter<test_type>(); ++i) {
        nixlXferReqH* req_handle = postXfer(op, isFailure<test_type>(i));
        nixl_status_t status     = m_Initiator.waitForCompletion(req_handle);

        if (isFailure<test_type>(i)) {
            EXPECT_EQ(NIXL_ERR_REMOTE_DISCONNECT, status);
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
        m_Target.destroy();
    case TestType::LOAD_REMOTE_THEN_FAIL:
    case TestType::XFER_THEN_FAIL:
        m_Initiator.destroy();
        break;
    default:
        EXPECT_TRUE(false) << "Invalid test type";
    }
}

template<TestUcxBackend::TestType test_type>
bool TestUcxBackend::isFailure(size_t iter) {
    switch (test_type) {
    case TestType::BASIC_XFER:            return false;
    case TestType::LOAD_REMOTE_THEN_FAIL: return iter == 0;
    case TestType::XFER_THEN_FAIL:        return iter == 1;
    }
}

template<TestUcxBackend::TestType test_type> size_t TestUcxBackend::numIter() {
    return (test_type == TestType::XFER_THEN_FAIL) ? 2 : 1;
}

void TestUcxBackend::exchangeMetaData() {
    m_Initiator.loadRemoteMD(m_Target.getLocalMD());
    m_Target.loadRemoteMD(m_Initiator.getLocalMD());
}

nixlXferReqH*
TestUcxBackend::postXfer(enum nixl_xfer_op_t op, bool target_failure) {
    EXPECT_TRUE(op == NIXL_WRITE || op == NIXL_READ);

    nixlBasicDesc sReq_src;
    nixl_xfer_dlist_t sReq_descs(DRAM_SEG);
    m_Initiator.fillRegList(sReq_descs, sReq_src);

    nixlBasicDesc rReq_dst;
    nixl_xfer_dlist_t rReq_descs(DRAM_SEG);
    m_Target.fillRegList(rReq_descs, rReq_dst);

    nixlXferReqH* req_handle;
    nixl_status_t status;

    status = m_Initiator.createXferReq(op, sReq_descs, rReq_descs, req_handle);
    EXPECT_EQ(NIXL_SUCCESS, status)
        << "createXferReq failed with unexpected error: "
        << nixlEnumStrings::statusStr(status);

    if (target_failure) {
        m_Target.destroy();
    }

    status = m_Initiator.postXferReq(req_handle);
    if (target_failure) {
        // If the target is destroyed, the transfer may fail immediately
        // or later
        EXPECT_TRUE((status == NIXL_ERR_REMOTE_DISCONNECT) ||
                    (status == NIXL_IN_PROG));
    } else {
        EXPECT_LE(0, status) << "status: "
                             << nixlEnumStrings::statusStr(status);
    }

    return req_handle;
}

TEST_F(TestUcxBackend, BasicXfer) {
    testXfer<TestType::BASIC_XFER, NIXL_WRITE>();
    testXfer<TestType::BASIC_XFER, NIXL_READ>();
}

TEST_F(TestUcxBackend, LoadRemoteThenFail) {
    testXfer<TestType::LOAD_REMOTE_THEN_FAIL, NIXL_WRITE>();
    testXfer<TestType::LOAD_REMOTE_THEN_FAIL, NIXL_READ>();
}

TEST_F(TestUcxBackend, XferThenFail) {
    testXfer<TestType::XFER_THEN_FAIL, NIXL_WRITE>();
    testXfer<TestType::XFER_THEN_FAIL, NIXL_READ>();
}

} // namespace gtest
