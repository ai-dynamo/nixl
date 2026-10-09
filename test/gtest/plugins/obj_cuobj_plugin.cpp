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
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <absl/time/clock.h>
#include <absl/time/time.h>

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
    std::string skipReason_;

    setupObjAccelTestFixture() {
        localParams_ = *GetParam().customParams;
        const char *endpoint = std::getenv("NIXL_OBJ_ENDPOINT_OVERRIDE");
        if (endpoint && endpoint[0] != '\0') {
            localParams_["endpoint_override"] = endpoint;
            localParams_["req_checksum"] = "required";
        }
        nixlBackendInitParams initParams = GetParam();
        initParams.customParams = &localParams_;
        // accelerated=true has no HTTP fallback, so the engine throws when the RDMA
        // fast path is unavailable (e.g. a cuObject build with no RDMA NIC, as the
        // CI container produces). Skip these tests there rather than fail.
        try {
            localBackendEngine_ = std::make_shared<nixlObjEngine>(&initParams);
        }
        catch (const std::exception &e) {
            skipReason_ = e.what();
        }
    }

    void
    SetUp() override {
        if (!localBackendEngine_) {
            GTEST_SKIP() << "S3 accelerated engine unavailable: " << skipReason_;
        }
        setupBackendTestFixture::SetUp();
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

#ifdef HAVE_CUDA
// GPU-direct (VRAM_SEG) transfer test for the generic standard-protocol
// S3-over-RDMA engine. Exercises the accel-layer VRAM paths: buffer pinning in
// registerMem(), VRAM advertisement in getSupportedMems(), and the RDMA
// putObjectAsync/getObjectAsync data path.
TEST_P(setupObjAccelTestFixture, AccelVramXferTest) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available, skipping VRAM test";
    }
    transferHandler<VRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, accel_agent_name, accel_agent_name, false, 1);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}
#endif // HAVE_CUDA

template<nixl_mem_t memType>
void
copyToMem(uintptr_t dst, const std::vector<uint8_t> &src) {
#ifdef HAVE_CUDA
    if (memType == VRAM_SEG) {
        ASSERT_EQ(
            cudaMemcpy(
                reinterpret_cast<void *>(dst), src.data(), src.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
        return;
    }
#endif
    std::memcpy(reinterpret_cast<void *>(dst), src.data(), src.size());
}

template<nixl_mem_t memType>
std::vector<uint8_t>
copyFromMem(uintptr_t src, size_t len) {
    std::vector<uint8_t> out(len);
#ifdef HAVE_CUDA
    if (memType == VRAM_SEG) {
        EXPECT_EQ(
            cudaMemcpy(out.data(), reinterpret_cast<void *>(src), len, cudaMemcpyDeviceToHost),
            cudaSuccess);
        return out;
    }
#endif
    std::memcpy(out.data(), reinterpret_cast<void *>(src), len);
    return out;
}

nixl_status_t
runXfer(nixlBackendEngine &engine,
        nixl_xfer_op_t op,
        const nixl_meta_dlist_t &local,
        const nixl_meta_dlist_t &remote) {
    nixlBackendReqH *handle = nullptr;
    nixl_status_t ret = engine.prepXfer(op, local, remote, accel_agent_name, handle);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }
    ret = engine.postXfer(op, local, remote, accel_agent_name, handle);
    const auto deadline = absl::Now() + absl::Seconds(30);
    while (ret == NIXL_IN_PROG && absl::Now() < deadline) {
        absl::SleepFor(absl::Milliseconds(10));
        ret = engine.checkXfer(handle);
    }
    engine.releaseReqH(handle);
    return ret;
}

// Registers one buffer and transfers two non-adjacent sub-ranges of it. cuObject
// mints tokens only against a registration's base address, so this covers the
// base + offset translation in SharedCuObjClient::getToken().
template<nixl_mem_t memType>
void
testSubRangeXfer(nixlBackendEngine &engine) {
    constexpr size_t entry_size = 1 << 20;
    constexpr int num_entries = 4;
    const std::vector<int> xfer_entries = {1, 3};

    memoryHandler<memType> mem(num_entries * entry_size, 0);
    nixlBlobDesc mem_desc;
    mem.populateBlobDesc(&mem_desc);
    nixlBackendMD *mem_md = nullptr;
    ASSERT_EQ(engine.registerMem(mem_desc, memType, mem_md), NIXL_SUCCESS);
    mem.setMD(mem_md);

    nixl_meta_dlist_t local(memType);
    nixl_meta_dlist_t remote(OBJ_SEG);
    std::vector<nixlBackendMD *> obj_mds;
    for (size_t i = 0; i < xfer_entries.size(); ++i) {
        nixlBlobDesc obj_desc(0, entry_size, i, "test-obj-key-subrange-" + std::to_string(i));
        nixlBackendMD *obj_md = nullptr;
        ASSERT_EQ(engine.registerMem(obj_desc, OBJ_SEG, obj_md), NIXL_SUCCESS);
        obj_mds.push_back(obj_md);

        nixlMetaDesc local_desc;
        mem.populateMetaDesc(&local_desc, xfer_entries[i], entry_size);
        local.addDesc(local_desc);

        remote.addDesc(nixlMetaDesc(0, entry_size, i, obj_md));
    }

    std::vector<uint8_t> pattern(num_entries * entry_size);
    std::mt19937 rng(2368);
    std::generate(pattern.begin(), pattern.end(), [&rng] { return static_cast<uint8_t>(rng()); });
    copyToMem<memType>(mem_desc.addr, pattern);

    EXPECT_EQ(runXfer(engine, NIXL_WRITE, local, remote), NIXL_SUCCESS);

    copyToMem<memType>(mem_desc.addr, std::vector<uint8_t>(pattern.size(), 0));
    EXPECT_EQ(runXfer(engine, NIXL_READ, local, remote), NIXL_SUCCESS);

    const auto result = copyFromMem<memType>(mem_desc.addr, pattern.size());
    for (int e = 0; e < num_entries; ++e) {
        const auto begin = e * entry_size;
        const bool transferred =
            std::find(xfer_entries.begin(), xfer_entries.end(), e) != xfer_entries.end();
        const std::vector<uint8_t> expected = transferred ?
            std::vector<uint8_t>(pattern.begin() + begin, pattern.begin() + begin + entry_size) :
            std::vector<uint8_t>(entry_size, 0);
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), result.begin() + begin))
            << "entry " << e << (transferred ? " was not read back" : " was modified");
    }

    for (auto *obj_md : obj_mds) {
        EXPECT_EQ(engine.deregisterMem(obj_md), NIXL_SUCCESS);
    }
    EXPECT_EQ(engine.deregisterMem(mem_md), NIXL_SUCCESS);
}

TEST_P(setupObjAccelTestFixture, AccelSubRangeXferTest) {
    testSubRangeXfer<DRAM_SEG>(*localBackendEngine_);
}

#ifdef HAVE_CUDA
TEST_P(setupObjAccelTestFixture, AccelVramSubRangeXferTest) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available, skipping VRAM test";
    }
    testSubRangeXfer<VRAM_SEG>(*localBackendEngine_);
}
#endif // HAVE_CUDA

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
