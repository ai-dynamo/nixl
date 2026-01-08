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

#include <gtest/gtest.h>

#include "plugins_common.h"
#include "transfer_handler.h"
#include "obj/obj_backend.h"

namespace gtest::plugins::obj {
/**
 * @note To run OBJ plugin tests, the following environment variables must be set:
 *       - AWS_ACCESS_KEY_ID
 *       - AWS_SECRET_ACCESS_KEY
 *       - AWS_DEFAULT_REGION
 *       - AWS_DEFAULT_BUCKET
 *
 * These variables are required for authenticating and interacting with the S3 bucket
 * used during the tests.
 */

nixl_b_params_t obj_params;
const std::string local_agent_name = "Agent1";
const nixlBackendInitParams obj_test_params = {.localAgent = local_agent_name,
                                               .type = "OBJ",
                                               .customParams = &obj_params,
                                               .enableProgTh = false,
                                               .pthrDelay = 0,
                                               .syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW};

class setupObjTestFixture : public setupBackendTestFixture {
protected:
    setupObjTestFixture() {
        localBackendEngine_ = std::make_shared<nixlObjEngine>(&GetParam());
    }
};

TEST_P(setupObjTestFixture, XferTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, 1);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupObjTestFixture, XferMultiBufsTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, 3);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupObjTestFixture, queryMemTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, 3);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-obj-key-0"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-obj-key-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-obj-key-nonexistent"));
    std::vector<nixl_query_resp_t> resp;
    localBackendEngine_->queryMem(descs, resp);

    EXPECT_EQ(resp.size(), 3);
    EXPECT_EQ(resp[0].has_value(), true);
    EXPECT_EQ(resp[1].has_value(), true);
    EXPECT_EQ(resp[2].has_value(), false);
}

TEST_P(setupObjTestFixture, writeWithOffsetsTest) {
    transferMemConfig mem_cfg{.numEntries_ = 2};
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, mem_cfg);
    transfer.setupMems();

    nixl_xfer_op_t op = NIXL_WRITE;
    ASSERT_EQ(transfer.prepareTransfer(op), NIXL_SUCCESS);
    ASSERT_EQ(transfer.postTransfer(op), NIXL_SUCCESS);
    ASSERT_EQ(transfer.waitForTransfer(), NIXL_ERR_BACKEND);
}

TEST_P(setupObjTestFixture, readWithOffsetsTest) {
    uint8_t src_byte = getRandomInt(0, 255);

    // First, write entire buffer to OBJ in a single chunk (no offsets)
    transferMemConfig mem_cfg_write{.numEntries_ = 1, .entrySize_ = 128, .srcBufByte_ = src_byte};
    transferHandler<DRAM_SEG, OBJ_SEG> transferWrite(localBackendEngine_,
                                                     localBackendEngine_,
                                                     local_agent_name,
                                                     local_agent_name,
                                                     mem_cfg_write);
    transferWrite.setupMems();
    transferWrite.setSrcMem();
    transferWrite.testTransfer(NIXL_WRITE);

    // Then, read buffer from OBJ in smaller offset-based chunks
    transferMemConfig mem_cfg_read{.numEntries_ = 2, .entrySize_ = 64, .srcBufByte_ = src_byte};
    transferHandler<DRAM_SEG, OBJ_SEG> transferRead(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, mem_cfg_read);
    transferRead.setupMems();
    transferRead.resetSrcMem();
    transferRead.testTransfer(NIXL_READ);
    transferRead.checkSrcMem();
}

TEST_P(setupObjTestFixture, xferUnregisteredMemTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name);

    nixlMetaDesc meta_desc;
    meta_desc.devId = 1;
    transfer.addSrcDesc(meta_desc);
    transfer.addDstDesc(meta_desc);

    nixl_xfer_op_t op = NIXL_WRITE;
    ASSERT_EQ(transfer.prepareTransfer(op), NIXL_SUCCESS);
    ASSERT_EQ(transfer.postTransfer(op), NIXL_ERR_BACKEND);
}

TEST_P(setupObjTestFixture, objAsSrcMemTest) {
    transferHandler<OBJ_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name);
    ASSERT_EQ(transfer.prepareTransfer(NIXL_WRITE), NIXL_ERR_INVALID_PARAM);
}

TEST_P(setupObjTestFixture, dramAsDstMemTest) {
    transferHandler<DRAM_SEG, DRAM_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name);
    ASSERT_EQ(transfer.prepareTransfer(NIXL_WRITE), NIXL_ERR_INVALID_PARAM);
}

INSTANTIATE_TEST_SUITE_P(ObjTests, setupObjTestFixture, testing::Values(obj_test_params));

} // namespace gtest::plugins::obj
