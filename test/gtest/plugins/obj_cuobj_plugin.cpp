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

#if defined HAVE_CUOBJ_CLIENT

#include <gtest/gtest.h>
#include <cstdlib>

#include "plugins_common.h"
#include "transfer_handler.h"
#include "obj/obj_backend.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gtest::plugins::obj {

nixl_b_params_t obj_accel_params = {{"accelerated", "true"}};
nixl_b_params_t obj_dell_params = {{"accelerated", "true"},
                                   {"type", "dell"},
                                   {"req_checksum", "required"},
                                   {"scheme", "http"}};
const std::string accel_agent_name = "Agent3-Accel";
const std::string dell_agent_name = "Agent4-Dell";

const nixlBackendInitParams obj_accel_test_params = {.localAgent = accel_agent_name,
                                                     .type = "OBJ",
                                                     .customParams = &obj_accel_params,
                                                     .enableProgTh = false,
                                                     .pthrDelay = 0,
                                                     .syncMode =
                                                         nixl_thread_sync_t::NIXL_THREAD_SYNC_RW};

const nixlBackendInitParams obj_dell_test_params = {.localAgent = dell_agent_name,
                                                    .type = "OBJ",
                                                    .customParams = &obj_dell_params,
                                                    .enableProgTh = false,
                                                    .pthrDelay = 0,
                                                    .syncMode =
                                                        nixl_thread_sync_t::NIXL_THREAD_SYNC_RW};

// Separate test suite for S3 Accelerated client with accelerated=true
// Note: These tests require the cuobjclient library to be available at compile time
class setupObjAccelTestFixture : public setupBackendTestFixture {
protected:
    nixl_b_params_t localParams_;

    setupObjAccelTestFixture() {
        localParams_ = *GetParam().customParams;
        const char *endpoint = std::getenv("NIXL_OBJ_ENDPOINT_OVERRIDE");
        if (endpoint && endpoint[0] != '\0') {
            localParams_["endpoint_override"] = endpoint;
            localParams_["req_checksum"] = "required";
        }
        nixlBackendInitParams initParams = GetParam();
        initParams.customParams = &localParams_;
        localBackendEngine_ = std::make_shared<nixlObjEngine>(&initParams);
    }
};

TEST_P(setupObjAccelTestFixture, AccelXferTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, accel_agent_name, accel_agent_name, false, 1);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupObjAccelTestFixture, AccelXferMultiBufsTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, accel_agent_name, accel_agent_name, false, 3);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupObjAccelTestFixture, AccelQueryMemTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, accel_agent_name, accel_agent_name, false, 3);
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

INSTANTIATE_TEST_SUITE_P(ObjAccelTests,
                         setupObjAccelTestFixture,
                         testing::Values(obj_accel_test_params));

/**
 * @brief Test fixture for Dell ObjectScale S3 over RDMA engine.
 *
 * Reads the NIXL_OBJ_ENDPOINT_OVERRIDE environment variable to configure
 * the Dell ObjectScale endpoint. Tests are skipped if the variable is not set.
 * Requires cuobjclient library (HAVE_CUOBJ_CLIENT).
 */
class setupObjDellTestFixture : public setupBackendTestFixture {
protected:
    nixl_b_params_t localParams_;

    setupObjDellTestFixture() {
        localParams_ = *GetParam().customParams;
        const char *endpoint = std::getenv("NIXL_OBJ_ENDPOINT_OVERRIDE");
        if (endpoint && endpoint[0] != '\0') {
            localParams_["endpoint_override"] = endpoint;
            nixlBackendInitParams initParams = GetParam();
            initParams.customParams = &localParams_;
            localBackendEngine_ = std::make_shared<nixlObjEngine>(&initParams);
        }
    }

    void
    SetUp() override {
        const char *endpoint = std::getenv("NIXL_OBJ_ENDPOINT_OVERRIDE");
        if (!endpoint || endpoint[0] == '\0') {
            GTEST_SKIP() << "NIXL_OBJ_ENDPOINT_OVERRIDE not set, skipping Dell tests";
        }
        setupBackendTestFixture::SetUp();
    }
};

TEST_P(setupObjDellTestFixture, DellXferTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, dell_agent_name, dell_agent_name, false, 1);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupObjDellTestFixture, DellXferMultiBufsTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, dell_agent_name, dell_agent_name, false, 3);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupObjDellTestFixture, DellQueryMemTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, dell_agent_name, dell_agent_name, false, 3);
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

#ifdef HAVE_CUDA
// GPU memory (VRAM_SEG) transfer test for Dell ObjectScale RDMA engine.
// Exercises the VRAM-specific code paths: cuMemObjGetDescriptor/PutDescriptor for
// RDMA descriptor registration, and putObjectRdmaAsync/getObjectRdmaAsync for
// GPU-direct RDMA transfers.
TEST_P(setupObjDellTestFixture, DellVramXferTest) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available, skipping VRAM test";
    }
    transferHandler<VRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, dell_agent_name, dell_agent_name, false, 1);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}
#endif // HAVE_CUDA

INSTANTIATE_TEST_SUITE_P(ObjDellTests,
                         setupObjDellTestFixture,
                         testing::Values(obj_dell_test_params));

} // namespace gtest::plugins::obj

#endif // HAVE_CUOBJ_CLIENT
