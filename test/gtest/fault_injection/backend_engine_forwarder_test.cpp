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

#include <chrono>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "backend_engine_forwarder.h"
#include "mocks/gmock_engine.h"

namespace {

using testing::DoAll;
using testing::ElementsAre;
using testing::Ref;
using testing::Return;
using testing::SetArgReferee;
using testing::StrictMock;
using testing::_;

class dummyBackendMD : public nixlBackendMD {
public:
    dummyBackendMD() : nixlBackendMD(false) {}
};

class dummyBackendReqH : public nixlBackendReqH {};

class backendEngineForwarderTest : public testing::Test {
protected:
    backendEngineForwarderTest()
        : init_params_{.localAgent = "local", .type = "FORWARDER", .customParams = &custom_params_},
          forwarder_(&init_params_, inner_) {}

    nixl_b_params_t custom_params_;
    nixlBackendInitParams init_params_;
    StrictMock<mocks::GMockBackendEngine> inner_;
    nixlBackendEngineForwarder forwarder_;
};

static_assert(!std::is_abstract_v<nixlBackendEngineForwarder>);

TEST_F(backendEngineForwarderTest, ForwardsCapabilitiesAndMemoryRegistration) {
    nixlBlobDesc blob(0, 1, 0);
    dummyBackendMD metadata;
    nixlBackendMD *registered = nullptr;

    EXPECT_CALL(inner_, supportsRemote()).WillOnce(Return(true));
    EXPECT_CALL(inner_, supportsLocal()).WillOnce(Return(false));
    EXPECT_CALL(inner_, supportsNotif()).WillOnce(Return(true));
    EXPECT_CALL(inner_, getSupportedMems()).WillOnce(Return(nixl_mem_list_t{DRAM_SEG}));
    EXPECT_CALL(inner_, registerMem(Ref(blob), DRAM_SEG, _))
        .WillOnce(DoAll(SetArgReferee<2>(&metadata), Return(NIXL_SUCCESS)));
    EXPECT_CALL(inner_, deregisterMem(&metadata)).WillOnce(Return(NIXL_ERR_BACKEND));

    EXPECT_TRUE(forwarder_.supportsRemote());
    EXPECT_FALSE(forwarder_.supportsLocal());
    EXPECT_TRUE(forwarder_.supportsNotif());
    EXPECT_THAT(forwarder_.getSupportedMems(), ElementsAre(DRAM_SEG));
    EXPECT_EQ(forwarder_.registerMem(blob, DRAM_SEG, registered), NIXL_SUCCESS);
    EXPECT_EQ(registered, &metadata);
    EXPECT_EQ(forwarder_.deregisterMem(&metadata), NIXL_ERR_BACKEND);
}

TEST_F(backendEngineForwarderTest, ForwardsConnectionsAndMetadata) {
    const std::string remote_agent = "remote";
    nixlBlobDesc blob(0, 1, 0);
    dummyBackendMD input;
    dummyBackendMD output_metadata;
    nixlBackendMD *output = nullptr;
    std::string serialized;

    EXPECT_CALL(inner_, connect(remote_agent)).WillOnce(Return(NIXL_SUCCESS));
    EXPECT_CALL(inner_, disconnect(remote_agent)).WillOnce(Return(NIXL_ERR_BACKEND));
    EXPECT_CALL(inner_, unloadMD(&input)).WillOnce(Return(NIXL_SUCCESS));
    EXPECT_CALL(inner_, getPublicData(&input, Ref(serialized)))
        .WillOnce(DoAll(SetArgReferee<1>("metadata"), Return(NIXL_SUCCESS)));
    EXPECT_CALL(inner_, getConnInfo(Ref(serialized)))
        .WillOnce(DoAll(SetArgReferee<0>("connection"), Return(NIXL_SUCCESS)));
    EXPECT_CALL(inner_, loadRemoteConnInfo(remote_agent, "connection"))
        .WillOnce(Return(NIXL_ERR_BACKEND));
    EXPECT_CALL(inner_, loadRemoteMD(Ref(blob), DRAM_SEG, remote_agent, _))
        .WillOnce(DoAll(SetArgReferee<3>(&output_metadata), Return(NIXL_SUCCESS)));
    EXPECT_CALL(inner_, loadLocalMD(&input, _))
        .WillOnce(DoAll(SetArgReferee<1>(&output_metadata), Return(NIXL_SUCCESS)));

    EXPECT_EQ(forwarder_.connect(remote_agent), NIXL_SUCCESS);
    EXPECT_EQ(forwarder_.disconnect(remote_agent), NIXL_ERR_BACKEND);
    EXPECT_EQ(forwarder_.unloadMD(&input), NIXL_SUCCESS);
    EXPECT_EQ(forwarder_.getPublicData(&input, serialized), NIXL_SUCCESS);
    EXPECT_EQ(serialized, "metadata");
    EXPECT_EQ(forwarder_.getConnInfo(serialized), NIXL_SUCCESS);
    EXPECT_EQ(serialized, "connection");
    EXPECT_EQ(forwarder_.loadRemoteConnInfo(remote_agent, serialized), NIXL_ERR_BACKEND);
    EXPECT_EQ(forwarder_.loadRemoteMD(blob, DRAM_SEG, remote_agent, output), NIXL_SUCCESS);
    EXPECT_EQ(output, &output_metadata);
    output = nullptr;
    EXPECT_EQ(forwarder_.loadLocalMD(&input, output), NIXL_SUCCESS);
    EXPECT_EQ(output, &output_metadata);
}

TEST_F(backendEngineForwarderTest, ForwardsTransfersAndOpaqueHandles) {
    const std::string remote_agent = "remote";
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(DRAM_SEG);
    dummyBackendReqH request;
    nixlBackendReqH *handle = nullptr;

    EXPECT_CALL(inner_, prepXfer(NIXL_WRITE, Ref(local), Ref(remote), remote_agent, _, nullptr))
        .WillOnce(DoAll(SetArgReferee<4>(&request), Return(NIXL_SUCCESS)));
    EXPECT_CALL(inner_, postXfer(NIXL_WRITE, Ref(local), Ref(remote), remote_agent, _, nullptr))
        .WillOnce(Return(NIXL_IN_PROG));
    EXPECT_CALL(inner_, checkXfer(&request)).WillOnce(Return(NIXL_SUCCESS));
    EXPECT_CALL(inner_, releaseReqH(&request)).WillOnce(Return(NIXL_ERR_BACKEND));

    EXPECT_EQ(forwarder_.prepXfer(NIXL_WRITE, local, remote, remote_agent, handle), NIXL_SUCCESS);
    EXPECT_EQ(handle, &request);
    EXPECT_EQ(forwarder_.postXfer(NIXL_WRITE, local, remote, remote_agent, handle), NIXL_IN_PROG);
    EXPECT_EQ(forwarder_.checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(forwarder_.releaseReqH(handle), NIXL_ERR_BACKEND);
}

TEST_F(backendEngineForwarderTest, ForwardsOptionalOperations) {
    const std::string remote_agent = "remote";
    nixl_remote_meta_dlist_t remote_descs(DRAM_SEG);
    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_reg_dlist_t query_descs(DRAM_SEG);
    std::vector<nixl_query_resp_t> query_response;
    notif_list_t notifications;
    int remote_view_token;
    int local_view_token;
    nixlMemViewH remote_view = nullptr;
    nixlMemViewH local_view = nullptr;
    nixlBackendReqH *handle = nullptr;
    std::chrono::microseconds duration;
    std::chrono::microseconds error_margin;
    nixl_cost_t method = nixl_cost_t::ANALYTICAL_BACKEND;
    const testing::Matcher<const nixl_remote_meta_dlist_t &> remote_descs_matcher =
        Ref(remote_descs);
    const testing::Matcher<const nixl_meta_dlist_t &> local_descs_matcher = Ref(local_descs);

    EXPECT_CALL(inner_, prepMemView(remote_descs_matcher, _, nullptr))
        .WillOnce(DoAll(SetArgReferee<1>(&remote_view_token), Return(NIXL_SUCCESS)));
    EXPECT_CALL(inner_, prepMemView(local_descs_matcher, _, nullptr))
        .WillOnce(DoAll(SetArgReferee<1>(&local_view_token), Return(NIXL_ERR_BACKEND)));
    EXPECT_CALL(inner_, releaseMemView(&remote_view_token));
    EXPECT_CALL(inner_, getNotifs(Ref(notifications))).WillOnce(Return(NIXL_SUCCESS));
    EXPECT_CALL(inner_, genNotif(remote_agent, "message")).WillOnce(Return(NIXL_ERR_BACKEND));
    EXPECT_CALL(inner_, queryMem(Ref(query_descs), Ref(query_response)))
        .WillOnce(Return(NIXL_SUCCESS));
    EXPECT_CALL(inner_,
                estimateXferCost(NIXL_READ,
                                 Ref(local_descs),
                                 Ref(local_descs),
                                 remote_agent,
                                 Ref(handle),
                                 Ref(duration),
                                 Ref(error_margin),
                                 Ref(method),
                                 nullptr))
        .WillOnce(Return(NIXL_SUCCESS));

    EXPECT_EQ(forwarder_.prepMemView(remote_descs, remote_view), NIXL_SUCCESS);
    EXPECT_EQ(remote_view, &remote_view_token);
    EXPECT_EQ(forwarder_.prepMemView(local_descs, local_view), NIXL_ERR_BACKEND);
    EXPECT_EQ(local_view, &local_view_token);
    forwarder_.releaseMemView(remote_view);
    EXPECT_EQ(forwarder_.getNotifs(notifications), NIXL_SUCCESS);
    EXPECT_EQ(forwarder_.genNotif(remote_agent, "message"), NIXL_ERR_BACKEND);
    EXPECT_EQ(forwarder_.queryMem(query_descs, query_response), NIXL_SUCCESS);
    EXPECT_EQ(forwarder_.estimateXferCost(NIXL_READ,
                                          local_descs,
                                          local_descs,
                                          remote_agent,
                                          handle,
                                          duration,
                                          error_margin,
                                          method),
              NIXL_SUCCESS);
}

} // namespace
