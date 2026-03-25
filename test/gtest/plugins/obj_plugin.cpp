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

#include <gtest/gtest.h>
#include <cstdlib>

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
 *
 * Test suites:
 * - ObjTests: Standard S3 client tests (crtMinLimit = 0)
 * - ObjCrtTests: S3 CRT client tests (crtMinLimit = 5 MiB, buffer = 10 MiB)
 *                crtMinLimit is set to the S3 minimum part size (5 MiB) so that
 *                partSize is not clamped and MPU is exercised with multiple parts.
 * - ObjAccelTests: S3 Accelerated client tests (accelerated = true)
 *                  Note: Defined in obj_cuobj_plugin.cpp, only compiled if
 *                  HAVE_CUOBJ_CLIENT is defined
 * - ObjDellTests: Dell ObjectScale S3 over RDMA tests
 *                 (accelerated = true, type = dell, req_checksum = required, scheme = http)
 *                 Note: Defined in obj_cuobj_plugin.cpp, only compiled if
 *                 HAVE_CUOBJ_CLIENT is defined.
 *                 Skipped at runtime if NIXL_OBJ_ENDPOINT_OVERRIDE is not set.
 *                 Includes DellVramXferTest (GPU memory) if HAVE_CUDA is also defined.
 *
 * Environment:
 *       - NIXL_OBJ_ENDPOINT_OVERRIDE  (e.g. http://100.68.213.151:9020)
 *         Optional for all test suites. When set, overrides the S3 endpoint for
 *         Standard, CRT, Accel, and Dell tests alike.
 */

nixl_b_params_t obj_params = {{"crtMinLimit", "0"}};
nixl_b_params_t obj_crt_params = {{"crtMinLimit", "5242880"}}; // 5 MiB: S3 minimum part size
const std::string local_agent_name = "Agent1";
const std::string crt_agent_name = "Agent2-CRT";
const nixlBackendInitParams obj_test_params = {.localAgent = local_agent_name,
                                               .type = "OBJ",
                                               .customParams = &obj_params,
                                               .enableProgTh = false,
                                               .pthrDelay = 0,
                                               .syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW};

const nixlBackendInitParams obj_crt_test_params = {.localAgent = crt_agent_name,
                                                   .type = "OBJ",
                                                   .customParams = &obj_crt_params,
                                                   .enableProgTh = false,
                                                   .pthrDelay = 0,
                                                   .syncMode =
                                                       nixl_thread_sync_t::NIXL_THREAD_SYNC_RW};

class setupObjTestFixture : public setupBackendTestFixture {
protected:
    setupObjTestFixture() {
        const char *endpoint = std::getenv("NIXL_OBJ_ENDPOINT_OVERRIDE");
        if (endpoint && endpoint[0] != '\0') {
            obj_params["endpoint_override"] = endpoint;
            obj_params["req_checksum"] = "required";
        }
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

INSTANTIATE_TEST_SUITE_P(ObjTests, setupObjTestFixture, testing::Values(obj_test_params));

// Separate test suite for S3 CRT client with crtMinLimit enabled
class setupObjCrtTestFixture : public setupBackendTestFixture {
protected:
    setupObjCrtTestFixture() {
        const char *endpoint = std::getenv("NIXL_OBJ_ENDPOINT_OVERRIDE");
        if (endpoint && endpoint[0] != '\0') {
            obj_crt_params["endpoint_override"] = endpoint;
        }
        localBackendEngine_ = std::make_shared<nixlObjEngine>(&GetParam());
    }
};

TEST_P(setupObjCrtTestFixture, CrtXferTest) {
    // 10 MiB buffer: above the 5 MiB CRT threshold, exercises MPU (two 5 MiB parts)
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(localBackendEngine_,
                                                localBackendEngine_,
                                                crt_agent_name,
                                                crt_agent_name,
                                                false,
                                                1,
                                                10485760);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupObjCrtTestFixture, CrtXferMultiBufsTest) {
    // 10 MiB buffer: above the 5 MiB CRT threshold, exercises MPU (two 5 MiB parts)
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(localBackendEngine_,
                                                localBackendEngine_,
                                                crt_agent_name,
                                                crt_agent_name,
                                                false,
                                                3,
                                                10485760);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupObjCrtTestFixture, CrtQueryMemTest) {
    // 10 MiB buffer: above the 5 MiB CRT threshold, exercises MPU (two 5 MiB parts)
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(localBackendEngine_,
                                                localBackendEngine_,
                                                crt_agent_name,
                                                crt_agent_name,
                                                false,
                                                3,
                                                10485760);
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

INSTANTIATE_TEST_SUITE_P(ObjCrtTests, setupObjCrtTestFixture, testing::Values(obj_crt_test_params));

} // namespace gtest::plugins::obj
