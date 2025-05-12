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
namespace ucx {

namespace nixl {
    static std::unique_ptr<nixlAgent> createAgent(const std::string& name)
    {
        return std::make_unique<nixlAgent>(name, nixlAgentConfig(true));
    }

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

    static nixl_opt_args_t createExtraParams(nixlBackendH* backend)
    {
        nixl_opt_args_t extra_params;
        extra_params.backends = {backend};
        return extra_params;
    }

    template <typename DListT, typename DescT> void
    fillRegList(DListT &dlist, DescT &desc, std::vector<std::byte>& data)
    {
        desc.addr  = reinterpret_cast<uintptr_t>(data.data());
        desc.len   = data.size();
        desc.devId = 0;
        dlist.addDesc(desc);
    }

    static nixl_status_t
    wait_for_completion(nixlAgent& agent, nixlXferReqH* req_handle)
    {
        nixl_status_t status;

        do {
            status = agent.getXferStatus(req_handle);
        } while (status == NIXL_IN_PROG);

        agent.releaseXferReq(req_handle);

        if (status == NIXL_ERR_REMOTE_DISCONNECT) {
            std::cout << "Handled error: "
                      << nixlEnumStrings::statusStr(status) << std::endl;
        } else if (status != NIXL_SUCCESS) {
            std::cout << "Unexpected error: "
                      << nixlEnumStrings::statusStr(status) << std::endl;
        }

        return status;
    }

    static nixl_status_t
    wait_for_notif(nixlAgent& agent, const std::string& remoteAgentName,
                   const std::string& expectedNotif) {
        nixl_notifs_t notif_map;

        do {
            EXPECT_EQ(NIXL_SUCCESS, agent.getNotifs(notif_map));
        } while (notif_map.empty());

        std::vector<std::string> notifs = notif_map[remoteAgentName];
        EXPECT_EQ(1u, notifs.size());
        EXPECT_EQ(expectedNotif, notifs.front());

        return NIXL_SUCCESS;
    }
} // namespace nixl

class UcxTestFixture : public testing::Test {
protected:
    UcxTestFixture() {
        // Set up test environment
        m_plugin_dir_backup = getenv("NIXL_PLUGIN_DIR");

        // Load plugins from build directory
        std::string plugin_dir = std::string(BUILD_DIR) + "/src/plugins/ucx";
        setenv("NIXL_PLUGIN_DIR", plugin_dir.c_str(), 1);

        std::cout << "set NIXL_PLUGIN_DIR: " << getenv("NIXL_PLUGIN_DIR")
                  << std::endl;
    }

    ~UcxTestFixture() {
        setenv("NIXL_PLUGIN_DIR", m_plugin_dir_backup.c_str(), 1);
        std::cout << "restore NIXL_PLUGIN_DIR: " << getenv("NIXL_PLUGIN_DIR")
                  << std::endl;
    }

    static void test_xfer(bool receiver_failure = false)
    {
        const size_t len                  = 256;
        std::unique_ptr<nixlAgent> sAgent = nixl::createAgent("sender");
        std::unique_ptr<nixlAgent> rAgent = nixl::createAgent("receiver");

        nixlBackendH* sBackend = nixl::createUcxBackend(*sAgent);
        nixlBackendH* rBackend = nixl::createUcxBackend(*rAgent);

        nixl_opt_args_t sExtraParams = nixl::createExtraParams(sBackend);
        nixl_opt_args_t rExtraParams = nixl::createExtraParams(rBackend);

        nixl_reg_dlist_t sDlist(DRAM_SEG);
        nixlBlobDesc sBuff;
        std::vector<std::byte> sData(len, std::byte{0xbb});
        nixl::fillRegList(sDlist, sBuff, sData);
        EXPECT_EQ(NIXL_SUCCESS, sAgent->registerMem(sDlist, &sExtraParams));

        nixl_reg_dlist_t rDlist(DRAM_SEG);
        nixlBlobDesc rBuff;
        std::vector<std::byte> rData(len, std::byte{0});
        nixl::fillRegList(rDlist, rBuff, rData);

        EXPECT_EQ(NIXL_SUCCESS, rAgent->registerMem(rDlist, &rExtraParams));

        std::string sMeta;
        EXPECT_EQ(NIXL_SUCCESS, sAgent->getLocalMD(sMeta));
        std::string rMeta;
        EXPECT_EQ(NIXL_SUCCESS, rAgent->getLocalMD(rMeta));

        std::string sMeta_remote;
        EXPECT_EQ(NIXL_SUCCESS, sAgent->loadRemoteMD(rMeta, sMeta_remote));
        std::string rMeta_remote;
        EXPECT_EQ(NIXL_SUCCESS, rAgent->loadRemoteMD(sMeta, rMeta_remote));

        nixlBasicDesc sReq_src;
        nixl_xfer_dlist_t sReq_descs(DRAM_SEG);
        nixl::fillRegList(sReq_descs, sReq_src, sData);

        if (receiver_failure) {
            rAgent->deregisterMem(rDlist, &rExtraParams);
            rAgent.reset();
        }

        nixlBasicDesc rReq_dst;
        nixl_xfer_dlist_t rReq_descs(DRAM_SEG);
        nixl::fillRegList(rReq_descs, rReq_dst, rData);

        sExtraParams.notifMsg = "notification";
        sExtraParams.hasNotif = true;
        nixlXferReqH* req_handle;
        EXPECT_EQ(NIXL_SUCCESS,
                  sAgent->createXferReq(NIXL_WRITE, sReq_descs, rReq_descs,
                                        "receiver", req_handle, &sExtraParams));
        if (receiver_failure) {
            // the error may be returned immediately or later
            nixl_status_t status = sAgent->postXferReq(req_handle);
            EXPECT_TRUE((status == NIXL_ERR_REMOTE_DISCONNECT) ||
                        (status == NIXL_IN_PROG));
        } else {
            EXPECT_LE(0, sAgent->postXferReq(req_handle));
        }

        if (receiver_failure) {
            EXPECT_EQ(NIXL_ERR_REMOTE_DISCONNECT,
                      nixl::wait_for_completion(*sAgent, req_handle));
        } else {
            EXPECT_EQ(NIXL_SUCCESS, nixl::wait_for_completion(*sAgent, req_handle));
            EXPECT_EQ(NIXL_SUCCESS, nixl::wait_for_notif(*rAgent, "sender", "notification"));
            EXPECT_EQ(sData, rData);
        }

        EXPECT_EQ(NIXL_SUCCESS, sAgent->deregisterMem(sDlist, &sExtraParams));
        if (!receiver_failure) {
            EXPECT_EQ(NIXL_SUCCESS, rAgent->deregisterMem(rDlist, &rExtraParams));
        }
    }
private:
    std::string m_plugin_dir_backup;
};

TEST_F(UcxTestFixture, basic_xfer) {
    test_xfer();
}

TEST_F(UcxTestFixture, receiver_failure) {
    test_xfer(true);
}

} // namespace ucx
} // namespace gtest
