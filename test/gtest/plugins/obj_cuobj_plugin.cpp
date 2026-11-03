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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <random>
#include <thread>
#include <vector>

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <cuobjclient.h>

#include "../common.h"
#include "plugins_common.h"
#include "transfer_handler.h"
#include "obj/obj_backend.h"
#include "object/rdma/cuobj_client.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace gtest::plugins::obj {

nixl_b_params_t obj_accel_params = {{"accelerated", "true"}};
nixl_b_params_t obj_dell_params = {{"accelerated", "true"},
                                   {"type", "dell"},
                                   {"req_checksum", "required"},
                                   {"scheme", "http"}};
constexpr const char *endpoint_env_var = "NIXL_OBJ_ENDPOINT_OVERRIDE";
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
    std::string constructionError_;

    setupObjAccelTestFixture() {
        localParams_ = *GetParam().customParams;
        const char *endpoint = std::getenv(endpoint_env_var);
        if (endpoint && endpoint[0] != '\0') {
            localParams_["endpoint_override"] = endpoint;
            localParams_["req_checksum"] = "required";
            nixlBackendInitParams initParams = GetParam();
            initParams.customParams = &localParams_;
            // accelerated=true has no HTTP fallback, so the engine throws when the
            // fast path is missing. SetUp() decides whether that is a skip.
            try {
                localBackendEngine_ = std::make_shared<nixlObjEngine>(&initParams);
            }
            catch (const std::exception &e) {
                constructionError_ = e.what();
            }
        }
    }

    void
    SetUp() override {
        const char *endpoint = std::getenv(endpoint_env_var);
        if (!endpoint || endpoint[0] == '\0') {
            GTEST_SKIP() << endpoint_env_var << " not set, skipping accelerated tests";
        }
        if (!localBackendEngine_) {
            // Only a missing fast path is a skip; anything else must fail.
            ASSERT_THAT(constructionError_, testing::HasSubstr("fast path is unavailable"));
            GTEST_SKIP() << "S3 accelerated engine unavailable: " << constructionError_;
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
        EXPECT_EQ(
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
// base + offset translation in SharedCuObjClient::getToken(). With
// nested_registration, a smaller region inside the buffer is also registered so
// that it is the nearest base below both transfers without containing either.
template<nixl_mem_t memType>
void
testSubRangeXfer(nixlBackendEngine &engine, bool nested_registration = false) {
    constexpr size_t entry_size = 1 << 20;
    constexpr int num_entries = 4;
    const std::vector<int> xfer_entries = {1, 3};

    memoryHandler<memType> mem(num_entries * entry_size, 0);
    nixlBlobDesc mem_desc;
    mem.populateBlobDesc(&mem_desc);
    nixlBackendMD *mem_md = nullptr;
    ASSERT_EQ(engine.registerMem(mem_desc, memType, mem_md), NIXL_SUCCESS);
    mem.setMD(mem_md);

    nixlBackendMD *inner_md = nullptr;
    if (nested_registration) {
        const nixlBlobDesc inner_desc(mem_desc.addr + entry_size, entry_size / 2, 0, "");
        ASSERT_EQ(engine.registerMem(inner_desc, memType, inner_md), NIXL_SUCCESS);
    }

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
    if (inner_md) {
        EXPECT_EQ(engine.deregisterMem(inner_md), NIXL_SUCCESS);
    }
    EXPECT_EQ(engine.deregisterMem(mem_md), NIXL_SUCCESS);
}

TEST_P(setupObjAccelTestFixture, AccelSubRangeXferTest) {
    testSubRangeXfer<DRAM_SEG>(*localBackendEngine_);
}

TEST_P(setupObjAccelTestFixture, AccelNestedRegistrationXferTest) {
    testSubRangeXfer<DRAM_SEG>(*localBackendEngine_, true);
}

// LMCache object keys use '@' as a separator. The control plane must sign such
// keys with the same RFC 3986 encoding the server uses to verify them.
TEST_P(setupObjAccelTestFixture, AccelReservedCharKeyXferTest) {
    constexpr size_t len = 1 << 20;
    memoryHandler<DRAM_SEG> mem(len, 0);
    nixlBlobDesc mem_desc;
    mem.populateBlobDesc(&mem_desc);
    nixlBackendMD *mem_md = nullptr;
    ASSERT_EQ(localBackendEngine_->registerMem(mem_desc, DRAM_SEG, mem_md), NIXL_SUCCESS);
    mem.setMD(mem_md);

    const nixlBlobDesc obj_desc(0, len, 0, "test-obj-key-model@0x01000100@0@a:b=c$d&e,f.data");
    nixlBackendMD *obj_md = nullptr;
    ASSERT_EQ(localBackendEngine_->registerMem(obj_desc, OBJ_SEG, obj_md), NIXL_SUCCESS);

    nixl_meta_dlist_t local(DRAM_SEG);
    nixlMetaDesc local_desc;
    mem.populateMetaDesc(&local_desc, 0, len);
    local.addDesc(local_desc);
    nixl_meta_dlist_t remote(OBJ_SEG);
    remote.addDesc(nixlMetaDesc(0, len, 0, obj_md));

    std::vector<uint8_t> pattern(len);
    std::mt19937 rng(2383);
    std::generate(pattern.begin(), pattern.end(), [&rng] { return static_cast<uint8_t>(rng()); });
    copyToMem<DRAM_SEG>(mem_desc.addr, pattern);
    EXPECT_EQ(runXfer(*localBackendEngine_, NIXL_WRITE, local, remote), NIXL_SUCCESS);

    copyToMem<DRAM_SEG>(mem_desc.addr, std::vector<uint8_t>(len, 0));
    EXPECT_EQ(runXfer(*localBackendEngine_, NIXL_READ, local, remote), NIXL_SUCCESS);
    EXPECT_EQ(copyFromMem<DRAM_SEG>(mem_desc.addr, len), pattern);

    EXPECT_EQ(localBackendEngine_->deregisterMem(obj_md), NIXL_SUCCESS);
    EXPECT_EQ(localBackendEngine_->deregisterMem(mem_md), NIXL_SUCCESS);
}

// Writes a random pattern from the whole of mem to a new object, zeroes mem,
// reads the object back and checks that the pattern returned.
template<nixl_mem_t memType>
void
testRoundTrip(nixlBackendEngine &engine,
              memoryHandler<memType> &mem,
              const nixlBlobDesc &mem_desc,
              const std::string &key) {
    nixlBlobDesc obj_desc(0, mem_desc.len, 0, key);
    nixlBackendMD *obj_md = nullptr;
    ASSERT_EQ(engine.registerMem(obj_desc, OBJ_SEG, obj_md), NIXL_SUCCESS);

    nixl_meta_dlist_t local(memType);
    nixl_meta_dlist_t remote(OBJ_SEG);
    nixlMetaDesc local_desc;
    mem.populateMetaDesc(&local_desc, 0, mem_desc.len);
    local.addDesc(local_desc);
    remote.addDesc(nixlMetaDesc(0, mem_desc.len, 0, obj_md));

    std::vector<uint8_t> pattern(mem_desc.len);
    std::mt19937 rng(2370);
    std::generate(pattern.begin(), pattern.end(), [&rng] { return static_cast<uint8_t>(rng()); });
    copyToMem<memType>(mem_desc.addr, pattern);
    EXPECT_EQ(runXfer(engine, NIXL_WRITE, local, remote), NIXL_SUCCESS);

    copyToMem<memType>(mem_desc.addr, std::vector<uint8_t>(pattern.size(), 0));
    EXPECT_EQ(runXfer(engine, NIXL_READ, local, remote), NIXL_SUCCESS);
    EXPECT_EQ(copyFromMem<memType>(mem_desc.addr, pattern.size()), pattern);

    EXPECT_EQ(engine.deregisterMem(obj_md), NIXL_SUCCESS);
}

// Several agents in one process registering the same buffer share one cuObject
// registration, so it must stay usable until the last of them deregisters.
TEST_P(setupObjAccelTestFixture, AccelSharedBufferRegistrationTest) {
    nixlBackendInitParams init_params = GetParam();
    init_params.customParams = &localParams_;
    nixlObjEngine other_engine(&init_params);

    memoryHandler<DRAM_SEG> mem(1 << 20, 0);
    nixlBlobDesc mem_desc;
    mem.populateBlobDesc(&mem_desc);
    nixlBackendMD *first_md = nullptr;
    nixlBackendMD *second_md = nullptr;
    ASSERT_EQ(localBackendEngine_->registerMem(mem_desc, DRAM_SEG, first_md), NIXL_SUCCESS);
    const nixl_status_t second = other_engine.registerMem(mem_desc, DRAM_SEG, second_md);
    EXPECT_EQ(localBackendEngine_->deregisterMem(first_md), NIXL_SUCCESS);
    ASSERT_EQ(second, NIXL_SUCCESS);
    mem.setMD(second_md);

    testRoundTrip<DRAM_SEG>(other_engine, mem, mem_desc, "test-obj-key-shared-registration");
    EXPECT_EQ(other_engine.deregisterMem(second_md), NIXL_SUCCESS);
}

// Agents register and release a buffer while another agent holds it and
// transfers through it, and then the last release races a new registration.
// The shared registration must stay usable for whichever agent still holds it.
TEST_P(setupObjAccelTestFixture, AccelSharedBufferConcurrentRegistrationTest) {
    constexpr int num_threads = 4;
    constexpr int iterations = 20;
    nixlBackendInitParams init_params = GetParam();
    init_params.customParams = &localParams_;

    memoryHandler<DRAM_SEG> mem(1 << 20, 0);
    nixlBlobDesc mem_desc;
    mem.populateBlobDesc(&mem_desc);
    nixlBackendMD *holder_md = nullptr;
    ASSERT_EQ(localBackendEngine_->registerMem(mem_desc, DRAM_SEG, holder_md), NIXL_SUCCESS);
    mem.setMD(holder_md);

    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&] {
            nixlObjEngine engine(&init_params);
            for (int i = 0; i < iterations; ++i) {
                nixlBackendMD *md = nullptr;
                if (engine.registerMem(mem_desc, DRAM_SEG, md) != NIXL_SUCCESS ||
                    engine.deregisterMem(md) != NIXL_SUCCESS) {
                    ++failures;
                }
            }
        });
    }
    for (int i = 0; i < iterations; ++i) {
        testRoundTrip<DRAM_SEG>(
            *localBackendEngine_, mem, mem_desc, "test-obj-key-shared-concurrent");
    }
    for (auto &thread : threads) {
        thread.join();
    }
    EXPECT_EQ(failures, 0);

    nixlObjEngine other_engine(&init_params);
    for (int i = 0; i < iterations; ++i) {
        nixlBackendMD *other_md = nullptr;
        nixl_status_t release_status = NIXL_ERR_UNKNOWN;
        nixl_status_t acquire_status = NIXL_ERR_UNKNOWN;
        std::thread release(
            [&] { release_status = localBackendEngine_->deregisterMem(holder_md); });
        std::thread acquire(
            [&] { acquire_status = other_engine.registerMem(mem_desc, DRAM_SEG, other_md); });
        release.join();
        acquire.join();
        EXPECT_EQ(release_status, NIXL_SUCCESS);
        ASSERT_EQ(acquire_status, NIXL_SUCCESS);
        mem.setMD(other_md);
        testRoundTrip<DRAM_SEG>(other_engine, mem, mem_desc, "test-obj-key-shared-race");

        const nixl_status_t reacquire =
            localBackendEngine_->registerMem(mem_desc, DRAM_SEG, holder_md);
        EXPECT_EQ(other_engine.deregisterMem(other_md), NIXL_SUCCESS);
        ASSERT_EQ(reacquire, NIXL_SUCCESS);
        mem.setMD(holder_md);
    }
    EXPECT_EQ(localBackendEngine_->deregisterMem(holder_md), NIXL_SUCCESS);
}

// A transfer inside another agent's nested registration gets its token from
// that registration. It must stay registered until the token is released, even
// if that agent deregisters it while the transfer is in flight.
TEST_P(setupObjAccelTestFixture, AccelTokenHoldsNestedRegistrationTest) {
    constexpr size_t len = 4 << 20;
    auto *rdma = nixl_obj_rdma::SharedCuObjClient::instance();
    ASSERT_NE(rdma, nullptr);
    nixlBackendInitParams init_params = GetParam();
    init_params.customParams = &localParams_;
    nixlObjEngine other_engine(&init_params);

    memoryHandler<DRAM_SEG> mem(len, 0);
    nixlBlobDesc mem_desc;
    mem.populateBlobDesc(&mem_desc);
    nixlBackendMD *outer_md = nullptr;
    ASSERT_EQ(localBackendEngine_->registerMem(mem_desc, DRAM_SEG, outer_md), NIXL_SUCCESS);
    const nixlBlobDesc inner_desc(mem_desc.addr + len / 4, len / 4, 0, "");
    nixlBackendMD *inner_md = nullptr;
    auto *inner = reinterpret_cast<void *>(inner_desc.addr);
    // Non-fatal checks only from here, so every registration is released even
    // when a check fails.
    if (other_engine.registerMem(inner_desc, DRAM_SEG, inner_md) != NIXL_SUCCESS) {
        ADD_FAILURE() << "registering the nested range failed";
    } else if (char *token = rdma->getToken(inner, 1 << 20, CUOBJ_PUT); token == nullptr) {
        ADD_FAILURE() << "minting a token in the nested range failed";
        EXPECT_EQ(other_engine.deregisterMem(inner_md), NIXL_SUCCESS);
    } else {
        EXPECT_EQ(other_engine.deregisterMem(inner_md), NIXL_SUCCESS);
        const bool released_early = rdma->registerBuffer(inner, len / 8);
        EXPECT_FALSE(released_early)
            << "the nested registration was released while its token was outstanding";
        rdma->putToken(token);
        if (released_early) {
            rdma->deregisterBuffer(inner);
        } else if (rdma->registerBuffer(inner, len / 8)) {
            rdma->deregisterBuffer(inner);
        } else {
            ADD_FAILURE() << "the nested registration was not released with its token";
        }
    }
    EXPECT_EQ(localBackendEngine_->deregisterMem(outer_md), NIXL_SUCCESS);
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

// cuObject moves at most CUOBJ_MAX_MEMORY_REG_SIZE bytes per request, so a
// larger object is written as a multipart upload and read back in ranges.
TEST_P(setupObjAccelTestFixture, AccelVramLargeObjectXferTest) {
    constexpr size_t chunk = 64 << 20;
    constexpr size_t len = size_t(9) << 30;
    constexpr size_t part = CUOBJ_MAX_MEMORY_REG_SIZE;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available, skipping VRAM test";
    }
    std::unique_ptr<memoryHandler<VRAM_SEG>> mem;
    try {
        const gtest::LogIgnoreGuard lig("cudaMalloc.* failed");
        mem = std::make_unique<memoryHandler<VRAM_SEG>>(len, 0);
    }
    catch (const std::runtime_error &) {
        GTEST_SKIP() << "Cannot allocate " << len << " bytes of VRAM";
    }
    nixlBlobDesc mem_desc;
    mem->populateBlobDesc(&mem_desc);
    nixlBackendMD *mem_md = nullptr;
    nixl_status_t reg_status;
    {
        const gtest::LogIgnoreGuard lig_desc("cuMemObjGetDescriptor failed");
        const gtest::LogIgnoreGuard lig_reg("RDMA buffer registration failed");
        reg_status = localBackendEngine_->registerMem(mem_desc, VRAM_SEG, mem_md);
    }
    if (reg_status != NIXL_SUCCESS) {
        GTEST_SKIP() << "cuObject cannot register " << len << " bytes (needs CUDA 13.4 or later)";
    }
    mem->setMD(mem_md);
    auto *base = reinterpret_cast<uint8_t *>(mem_desc.addr);

    auto pattern = [](size_t index) {
        std::vector<uint32_t> words(chunk / sizeof(uint32_t));
        std::mt19937 rng(static_cast<uint32_t>(index));
        std::generate(words.begin(), words.end(), [&rng] { return rng(); });
        return words;
    };
    auto matches = [&](size_t begin, size_t end) {
        std::vector<uint32_t> words(chunk / sizeof(uint32_t));
        for (size_t i = begin / chunk; i * chunk < end; ++i) {
            if (cudaMemcpy(words.data(), base + i * chunk, chunk, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
                words != pattern(i)) {
                return false;
            }
        }
        return true;
    };
    for (size_t i = 0; i < len / chunk; ++i) {
        ASSERT_EQ(cudaMemcpy(base + i * chunk, pattern(i).data(), chunk, cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    nixlBlobDesc obj_desc(0, len, 0, "test-obj-key-large-object");
    nixlBackendMD *obj_md = nullptr;
    ASSERT_EQ(localBackendEngine_->registerMem(obj_desc, OBJ_SEG, obj_md), NIXL_SUCCESS);
    nixl_meta_dlist_t local(VRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    nixlMetaDesc local_desc;
    mem->populateMetaDesc(&local_desc, 0, len);
    local.addDesc(local_desc);
    remote.addDesc(nixlMetaDesc(0, len, 0, obj_md));
    EXPECT_EQ(runXfer(*localBackendEngine_, NIXL_WRITE, local, remote), NIXL_SUCCESS);

    ASSERT_EQ(cudaMemset(base, 0, len), cudaSuccess);
    EXPECT_EQ(runXfer(*localBackendEngine_, NIXL_READ, local, remote), NIXL_SUCCESS);
    EXPECT_TRUE(matches(0, len));

    const size_t across = part - chunk;
    ASSERT_EQ(cudaMemset(base + across, 0, 2 * chunk), cudaSuccess);
    nixl_meta_dlist_t local_range(VRAM_SEG);
    nixl_meta_dlist_t remote_range(OBJ_SEG);
    local_desc.addr = mem_desc.addr + across;
    local_desc.len = 2 * chunk;
    local_range.addDesc(local_desc);
    remote_range.addDesc(nixlMetaDesc(across, 2 * chunk, 0, obj_md));
    EXPECT_EQ(runXfer(*localBackendEngine_, NIXL_READ, local_range, remote_range), NIXL_SUCCESS);
    EXPECT_TRUE(matches(across, across + 2 * chunk));

    EXPECT_EQ(localBackendEngine_->deregisterMem(obj_md), NIXL_SUCCESS);
    EXPECT_EQ(localBackendEngine_->deregisterMem(mem_md), NIXL_SUCCESS);
}

TEST_P(setupObjAccelTestFixture, AccelVramNestedRegistrationXferTest) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available, skipping VRAM test";
    }
    testSubRangeXfer<VRAM_SEG>(*localBackendEngine_, true);
}

// VRAM registered from a thread that has made no CUDA calls, so no CUDA context
// is current on it, as with an application's worker thread.
TEST_P(setupObjAccelTestFixture, AccelVramWorkerThreadRegistrationTest) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available, skipping VRAM test";
    }
    memoryHandler<VRAM_SEG> mem(1 << 20, 0);
    nixlBlobDesc mem_desc;
    mem.populateBlobDesc(&mem_desc);
    nixlBackendMD *mem_md = nullptr;
    nixl_status_t status = NIXL_ERR_UNKNOWN;
    CUcontext context_after = nullptr;
    CUresult context_rc = CUDA_ERROR_UNKNOWN;
    std::thread([&] {
        status = localBackendEngine_->registerMem(mem_desc, VRAM_SEG, mem_md);
        context_rc = cuCtxGetCurrent(&context_after);
    }).join();
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(context_rc, CUDA_SUCCESS);
    EXPECT_EQ(context_after, nullptr);
    mem.setMD(mem_md);

    testRoundTrip<VRAM_SEG>(*localBackendEngine_, mem, mem_desc, "test-obj-key-vram-worker-thread");
    EXPECT_EQ(localBackendEngine_->deregisterMem(mem_md), NIXL_SUCCESS);
}

// Registering VRAM must leave the caller's own CUDA context current.
TEST_P(setupObjAccelTestFixture, AccelVramRegistrationRestoresContextTest) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available, skipping VRAM test";
    }
    memoryHandler<VRAM_SEG> mem(1 << 20, 0);
    nixlBlobDesc mem_desc;
    mem.populateBlobDesc(&mem_desc);
    std::thread([&] {
        CUdevice device;
        ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
        CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
        ASSERT_EQ(cuCtxCreate(&context, nullptr, 0, device), CUDA_SUCCESS);
#else
        ASSERT_EQ(cuCtxCreate(&context, 0, device), CUDA_SUCCESS);
#endif
        nixlBackendMD *mem_md = nullptr;
        EXPECT_EQ(localBackendEngine_->registerMem(mem_desc, VRAM_SEG, mem_md), NIXL_SUCCESS);
        CUcontext current = nullptr;
        EXPECT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
        EXPECT_EQ(current, context);
        if (mem_md) {
            EXPECT_EQ(localBackendEngine_->deregisterMem(mem_md), NIXL_SUCCESS);
        }
        EXPECT_EQ(cuCtxDestroy(context), CUDA_SUCCESS);
    }).join();
}
#endif // HAVE_CUDA

INSTANTIATE_TEST_SUITE_P(ObjAccelTests,
                         setupObjAccelTestFixture,
                         testing::Values(obj_accel_test_params));

// Stands in for cuObject: a second registration at the same start address
// fails, and tokens are minted only for ranges inside a live registration.
class fakeCuObjApi : public nixl_obj_rdma::cuObjApi {
public:
    cuObjErr_t
    cuMemObjGetDescriptor(void *ptr, size_t size) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        ++registerCalls_;
        return registered_.emplace(ptr, size).second ? CU_OBJ_SUCCESS : CU_OBJ_FAIL;
    }

    cuObjErr_t
    cuMemObjPutDescriptor(void *ptr) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        ++deregisterCalls_;
        return registered_.erase(ptr) ? CU_OBJ_SUCCESS : CU_OBJ_FAIL;
    }

    cuObjErr_t
    cuMemObjGetRDMAToken(void *ptr,
                         size_t size,
                         size_t buffer_offset,
                         cuObjOpType_t,
                         char **desc_str_out) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto it = registered_.find(ptr);
        if (it == registered_.end() || buffer_offset + size > it->second) {
            return CU_OBJ_FAIL;
        }
        *desc_str_out = strdup("token");
        return CU_OBJ_SUCCESS;
    }

    cuObjErr_t
    cuMemObjPutRDMAToken(char *desc_str) override {
        free(desc_str);
        return CU_OBJ_SUCCESS;
    }

    [[nodiscard]] bool
    isRegistered(void *ptr) {
        const std::lock_guard<std::mutex> lock(mutex_);
        return registered_.count(ptr) != 0;
    }

    [[nodiscard]] int
    registerCalls() {
        const std::lock_guard<std::mutex> lock(mutex_);
        return registerCalls_;
    }

    [[nodiscard]] int
    deregisterCalls() {
        const std::lock_guard<std::mutex> lock(mutex_);
        return deregisterCalls_;
    }

private:
    std::mutex mutex_;
    std::map<void *, size_t> registered_;
    int registerCalls_ = 0;
    int deregisterCalls_ = 0;
};

class sharedCuObjClientTest : public testing::Test {
protected:
    static constexpr size_t len = 1 << 20;

    sharedCuObjClientTest()
        : fake_(new fakeCuObjApi()),
          client_(std::unique_ptr<nixl_obj_rdma::cuObjApi>(fake_)) {}

    // Mints and releases a token for the first KiB of the buffer.
    [[nodiscard]] bool
    canMintToken() {
        char *token = client_.getToken(buf_.data(), 1024, CUOBJ_PUT);
        client_.putToken(token);
        return token != nullptr;
    }

    std::vector<uint8_t> buf_ = std::vector<uint8_t>(len);
    fakeCuObjApi *fake_;
    nixl_obj_rdma::SharedCuObjClient client_;
};

TEST_F(sharedCuObjClientTest, LastOwnerReleasesRegistration) {
    ASSERT_TRUE(client_.registerBuffer(buf_.data(), len));
    ASSERT_TRUE(client_.registerBuffer(buf_.data(), len));
    EXPECT_EQ(fake_->registerCalls(), 1);

    client_.deregisterBuffer(buf_.data());
    EXPECT_TRUE(fake_->isRegistered(buf_.data()));
    EXPECT_TRUE(canMintToken());

    client_.deregisterBuffer(buf_.data());
    EXPECT_FALSE(fake_->isRegistered(buf_.data()));
    EXPECT_EQ(fake_->deregisterCalls(), 1);
}

TEST_F(sharedCuObjClientTest, OwnersReleasingWhileAnotherTransfers) {
    constexpr int num_threads = 4;
    constexpr int iterations = 1000;
    ASSERT_TRUE(client_.registerBuffer(buf_.data(), len));

    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < iterations; ++i) {
                if (!client_.registerBuffer(buf_.data(), len)) {
                    ++failures;
                }
                client_.deregisterBuffer(buf_.data());
            }
        });
    }
    for (int i = 0; i < iterations; ++i) {
        if (!canMintToken()) {
            ++failures;
        }
    }
    for (auto &thread : threads) {
        thread.join();
    }
    EXPECT_EQ(failures, 0);
    EXPECT_EQ(fake_->registerCalls(), 1);
    EXPECT_EQ(fake_->deregisterCalls(), 0);

    client_.deregisterBuffer(buf_.data());
    EXPECT_FALSE(fake_->isRegistered(buf_.data()));
}

TEST_F(sharedCuObjClientTest, FinalReleaseRacesNewRegistration) {
    constexpr int iterations = 1000;
    for (int i = 0; i < iterations; ++i) {
        ASSERT_TRUE(client_.registerBuffer(buf_.data(), len));
        bool registered = false;
        std::thread release([&] { client_.deregisterBuffer(buf_.data()); });
        std::thread acquire([&] { registered = client_.registerBuffer(buf_.data(), len); });
        release.join();
        acquire.join();
        ASSERT_TRUE(registered);
        ASSERT_TRUE(fake_->isRegistered(buf_.data()));
        ASSERT_TRUE(canMintToken());

        client_.deregisterBuffer(buf_.data());
        ASSERT_FALSE(fake_->isRegistered(buf_.data()));
    }
    EXPECT_EQ(fake_->registerCalls(), fake_->deregisterCalls());
}

TEST_F(sharedCuObjClientTest, TokenKeepsNestedRegistration) {
    ASSERT_TRUE(client_.registerBuffer(buf_.data(), len));
    uint8_t *inner = buf_.data() + len / 4;
    ASSERT_TRUE(client_.registerBuffer(inner, len / 4));

    char *token = client_.getToken(inner, 1024, CUOBJ_GET);
    ASSERT_NE(token, nullptr);
    client_.deregisterBuffer(inner);
    EXPECT_TRUE(fake_->isRegistered(inner));

    client_.putToken(token);
    EXPECT_FALSE(fake_->isRegistered(inner));
    client_.deregisterBuffer(buf_.data());
    EXPECT_FALSE(fake_->isRegistered(buf_.data()));
}

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
        const char *endpoint = std::getenv(endpoint_env_var);
        if (endpoint && endpoint[0] != '\0') {
            localParams_["endpoint_override"] = endpoint;
            nixlBackendInitParams initParams = GetParam();
            initParams.customParams = &localParams_;
            localBackendEngine_ = std::make_shared<nixlObjEngine>(&initParams);
        }
    }

    void
    SetUp() override {
        const char *endpoint = std::getenv(endpoint_env_var);
        if (!endpoint || endpoint[0] == '\0') {
            GTEST_SKIP() << endpoint_env_var << " not set, skipping Dell tests";
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
