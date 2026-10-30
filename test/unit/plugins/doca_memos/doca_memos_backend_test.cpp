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

#include <gtest/gtest.h>
#include <array>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#include <cstring>

// Include mocks before the backend
#include "doca_memos_mocks.h"

// Now we can include the backend header (which will use our mocked DOCA types)
#include "doca_memos_backend.h"
#include "nixl_types.h"

namespace {

std::string
hexKey(const std::string &key) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(key.size() * 2);
    for (unsigned char c : key) {
        out.push_back(kDigits[c >> 4]);
        out.push_back(kDigits[c & 0xf]);
    }
    return out;
}

// The progress thread updates the mock under its lock, so threaded tests must
// read it the same way.
size_t
submittedTaskCount() {
    auto lk = DocaMockControl::lock();
    return DocaMockControl::instance().submitted_tasks.size();
}

int
taskAllocCallCount() {
    auto lk = DocaMockControl::lock();
    return DocaMockControl::instance().task_alloc_call_count;
}

} // namespace

struct xferBuffers {
    nixl_meta_dlist_t local{DRAM_SEG};
    nixl_meta_dlist_t remote{OBJ_SEG};
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> mds;
};

class DocMemosBackendTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        // Reset mock state before each test
        DocaMockControl::instance().reset();

        // Setup default init params
        setupInitParams();
    }

    void
    TearDown() override {
        // Clean up engine if created
        if (engine_) {
            engine_.reset();
        }
    }

    void
    setupInitParams(bool enable_progress_thread = false,
                    nixl_thread_sync_t sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE,
                    const std::string &device_name = "/dev/nvme0n1",
                    const std::string &num_tasks = "8192") {
        // Setup backend parameters
        backend_params_ = std::make_unique<nixl_b_params_t>();
        (*backend_params_)["device_name"] = device_name;
        (*backend_params_)["num_tasks"] = num_tasks;

        // Setup init params
        init_params_ = std::make_unique<nixlBackendInitParams>();
        init_params_->customParams = backend_params_.get();
        init_params_->enableProgTh = enable_progress_thread;
        init_params_->syncMode = sync_mode;
        init_params_->pthrDelay = 1000; // 1ms
    }

    void
    createEngine() {
        engine_ = std::make_unique<nixlDocaMemosEngine>(init_params_.get());
    }

    // Helper method to register memory and return metadata
    // Returns the metadata object that should be attached to the descriptor
    nixlBackendMD *
    registerMemory(uint64_t dev_id, const std::string &key) {
        if (!engine_) {
            return nullptr;
        }

        nixlBlobDesc blob;
        blob.devId = dev_id;
        blob.metaInfo = hexKey(key);
        blob.addr = 0; // Not used for KV
        blob.len = 0; // Not used for KV

        nixlBackendMD *out = nullptr;
        engine_->registerMem(blob, OBJ_SEG, out);
        return out;
    }

    // Retries releaseReqH() while it returns NIXL_ERR_REPOST_ACTIVE, as the
    // agent's caller would, polling the request in between.
    nixl_status_t
    releaseWhenDrained(nixlBackendReqH *handle) {
        for (int i = 0; i < 1000; i++) {
            nixl_status_t status = engine_->releaseReqH(handle);
            if (status != NIXL_ERR_REPOST_ACTIVE) {
                return status;
            }
            engine_->checkXfer(handle);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return NIXL_ERR_REPOST_ACTIVE;
    }

    void
    buildXfer(xferBuffers &x, int count, uint64_t dev_id_base, const std::string &key_prefix) {
        for (int i = 0; i < count; i++) {
            nixlMetaDesc local_desc, remote_desc;
            void *buf = malloc(1024);
            x.buffers.push_back(buf);
            local_desc.addr = reinterpret_cast<uintptr_t>(buf);
            local_desc.len = 1024;
            remote_desc.devId = dev_id_base + i;
            remote_desc.addr = 0;
            remote_desc.len = 1024;
            remote_desc.metadataP =
                registerMemory(remote_desc.devId, key_prefix + std::to_string(i));
            x.mds.push_back(remote_desc.metadataP);
            x.local.addDesc(local_desc);
            x.remote.addDesc(remote_desc);
        }
    }

    nixl_status_t
    waitForCompletion(nixlBackendReqH *handle) {
        nixl_status_t status = NIXL_IN_PROG;
        for (int i = 0; i < 1000 && status == NIXL_IN_PROG; i++) {
            status = engine_->checkXfer(handle);
            if (status == NIXL_IN_PROG) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        return status;
    }

    std::unique_ptr<nixlDocaMemosEngine> engine_;
    std::unique_ptr<nixlBackendInitParams> init_params_;
    std::unique_ptr<nixl_b_params_t> backend_params_;
};

namespace {

std::string
keyBytes(const docaMemosKey &key) {
    return std::string(reinterpret_cast<const char *>(key.key), key.keyLen);
}

std::string
bytes(std::initializer_list<uint8_t> values) {
    std::string out;
    out.reserve(values.size());
    for (uint8_t value : values) {
        out.push_back(static_cast<char>(value));
    }
    return out;
}

std::string
devIdBytes(uint64_t dev_id) {
    return std::string(reinterpret_cast<const char *>(&dev_id), sizeof(dev_id));
}

void
expectKeyBytes(const docaMemosKey &key, const std::string &expected) {
    ASSERT_EQ(expected.size(), key.keyLen);
    EXPECT_EQ(expected, keyBytes(key));
}

constexpr const char *kShortStringKey = "short-key";
constexpr const char *kLongStringKey = "this-is-a-long-application-object-key";
constexpr const char *kHexLookingKey = "00112233445566778899aabbccddeeff";

const std::string kShortStringBlake2b128 = bytes({0xcb,
                                                  0x14,
                                                  0x14,
                                                  0x08,
                                                  0x31,
                                                  0xcf,
                                                  0x3e,
                                                  0x3b,
                                                  0x47,
                                                  0xc0,
                                                  0xbd,
                                                  0x73,
                                                  0x9f,
                                                  0x48,
                                                  0xd4,
                                                  0xf3});
const std::string kLongStringBlake2b128 = bytes({0xdf,
                                                 0x29,
                                                 0x94,
                                                 0xb4,
                                                 0xd6,
                                                 0x6c,
                                                 0xb5,
                                                 0x05,
                                                 0x2f,
                                                 0x8a,
                                                 0xa1,
                                                 0xbb,
                                                 0xb2,
                                                 0x3f,
                                                 0xb9,
                                                 0xed});
const std::string kHexLookingBlake2b128 = bytes({0xcf,
                                                 0x99,
                                                 0x21,
                                                 0xf4,
                                                 0xf6,
                                                 0x26,
                                                 0x19,
                                                 0x29,
                                                 0x59,
                                                 0x51,
                                                 0x82,
                                                 0x9e,
                                                 0x3d,
                                                 0x56,
                                                 0xad,
                                                 0x56});
const std::string kHexDecodedKey = bytes({0x00,
                                          0x11,
                                          0x22,
                                          0x33,
                                          0x44,
                                          0x55,
                                          0x66,
                                          0x77,
                                          0x88,
                                          0x99,
                                          0xaa,
                                          0xbb,
                                          0xcc,
                                          0xdd,
                                          0xee,
                                          0xff});

} // namespace

// ============================================================================
// Constructor/Destructor Tests
// ============================================================================

TEST_F(DocMemosBackendTest, ConstructorSuccess) {
    createEngine();
    EXPECT_TRUE(engine_ != nullptr);
    EXPECT_FALSE(engine_->supportsRemote());
    EXPECT_TRUE(engine_->supportsLocal());
}

TEST_F(DocMemosBackendTest, ConstructorMissingDeviceName) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE, ""); // Empty device name
    EXPECT_THROW(createEngine(), std::runtime_error);
}

TEST_F(DocMemosBackendTest, ConstructorInvalidNumTasks) {
    for (const char *num_tasks : {"0", "64junk", "4294967296", "-1"}) {
        setupInitParams(
            false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE, "/dev/nvme0n1", num_tasks);
        EXPECT_THROW(createEngine(), std::runtime_error) << "num_tasks=" << num_tasks;
    }
}

TEST_F(DocMemosBackendTest, PluginParamsListAcceptedOptions) {
    const nixl_b_params_t params = nixlDocaMemosEngine::getPluginParams();
    for (const char *name : {"device_name", "num_tasks", "nguid", "convert_key_to_128bit"}) {
        EXPECT_EQ(params.count(name), 1u) << name;
    }
    EXPECT_EQ(params.size(), 4u);
}

TEST_F(DocMemosBackendTest, ConstructorDocaKvdevCreateFailure) {
    DocaMockControl::instance().nvme_kvdev_create_result = DOCA_ERROR_INVALID_VALUE;
    EXPECT_THROW(createEngine(), std::runtime_error);
}

TEST_F(DocMemosBackendTest, ConstructorDocaPECreateFailure) {
    DocaMockControl::instance().pe_create_result = DOCA_ERROR_NO_MEMORY;
    EXPECT_THROW(createEngine(), std::runtime_error);
}

TEST_F(DocMemosBackendTest, ConstructorWithProgressThread) {
    setupInitParams(true); // Enable progress thread
    createEngine();
    EXPECT_TRUE(engine_ != nullptr);
}

TEST_F(DocMemosBackendTest, ConvertKeyTo128bitDefaultOmitted) {
    createEngine();
    EXPECT_FALSE(engine_->getInitErr());
}

TEST_F(DocMemosBackendTest, ConvertKeyTo128bitExplicitTrue) {
    (*backend_params_)["convert_key_to_128bit"] = "true";
    createEngine();
    EXPECT_FALSE(engine_->getInitErr());
}

TEST_F(DocMemosBackendTest, ConvertKeyTo128bitExplicitFalse) {
    (*backend_params_)["convert_key_to_128bit"] = "false";
    createEngine();
    EXPECT_FALSE(engine_->getInitErr());
}

TEST_F(DocMemosBackendTest, ConvertKeyTo128bitInvalidValue) {
    (*backend_params_)["convert_key_to_128bit"] = "maybe";
    EXPECT_THROW(createEngine(), std::runtime_error);
}

// ============================================================================
// Memory Registration Tests
// ============================================================================

TEST_F(DocMemosBackendTest, RegisterMemObjSeg) {
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 100;
    desc.metaInfo = hexKey("test_key_123");

    nixl_mem_t nixl_mem = OBJ_SEG;
    nixlBackendMD *out = nullptr;

    nixl_status_t status = engine_->registerMem(desc, nixl_mem, out);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(out, nullptr);

    // Cleanup
    engine_->deregisterMem(out);
}

TEST_F(DocMemosBackendTest, RegisterMemObjSegWithoutMetaInfo) {
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 42;
    desc.metaInfo = ""; // Empty, should use devId

    nixl_mem_t nixl_mem = OBJ_SEG;
    nixlBackendMD *out = nullptr;

    nixl_status_t status = engine_->registerMem(desc, nixl_mem, out);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(out, nullptr);

    // Cleanup
    engine_->deregisterMem(out);
}

TEST_F(DocMemosBackendTest, RegisterMemUnsupportedType) {
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 1;

    nixl_mem_t unsupported_mem = VRAM_SEG; // Not supported by DOCA KV
    nixlBackendMD *out = nullptr;

    nixl_status_t status = engine_->registerMem(desc, unsupported_mem, out);
    EXPECT_NE(status, NIXL_SUCCESS);
}

TEST_F(DocMemosBackendTest, RegisterMemLongStringMetaInfoFailsDefaultMode) {
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 1234;
    desc.metaInfo = kLongStringKey;
    nixlBackendMD *out = nullptr;

    EXPECT_EQ(engine_->registerMem(desc, OBJ_SEG, out), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(out, nullptr);
}

TEST_F(DocMemosBackendTest, RegisterMemLongStringMetaInfoSucceedsInHashMode) {
    (*backend_params_)["convert_key_to_128bit"] = "true";
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 1234;
    desc.metaInfo = kLongStringKey;
    nixlBackendMD *out = nullptr;

    EXPECT_EQ(engine_->registerMem(desc, OBJ_SEG, out), NIXL_SUCCESS);
    ASSERT_NE(out, nullptr);

    engine_->deregisterMem(out);
}

// ============================================================================
// Key Resolver Tests
// ============================================================================

TEST_F(DocMemosBackendTest, ResolveDefaultModeEmptyMetaInfoUsesDevIdBytes) {
    docaMemosKey key;
    uint64_t dev_id = 0;

    ASSERT_TRUE(nixlDocaMemosEngine::resolveMemosKey(dev_id, "", key));
    expectKeyBytes(key, devIdBytes(dev_id));
}

TEST_F(DocMemosBackendTest, ResolveDefaultModeHexStringDecodesToSixteenBytes) {
    docaMemosKey key;

    ASSERT_TRUE(nixlDocaMemosEngine::resolveMemosKey(999, kHexLookingKey, key));
    expectKeyBytes(key, kHexDecodedKey);
}

TEST_F(DocMemosBackendTest, ResolveDefaultModeShortNonHexStringIsRejected) {
    docaMemosKey key;

    EXPECT_FALSE(nixlDocaMemosEngine::resolveMemosKey(999, kShortStringKey, key));
}

TEST_F(DocMemosBackendTest, ResolveDefaultModeLongNonHexStringIsRejected) {
    docaMemosKey key;

    EXPECT_FALSE(nixlDocaMemosEngine::resolveMemosKey(999, kLongStringKey, key));
}

TEST_F(DocMemosBackendTest, ResolveHashModeEmptyMetaInfoStillUsesDevIdBytes) {
    docaMemosKey key;
    uint64_t dev_id = 0;

    ASSERT_TRUE(nixlDocaMemosEngine::resolveMemosKey(dev_id, "", key, true));
    expectKeyBytes(key, devIdBytes(dev_id));
}

TEST_F(DocMemosBackendTest, ResolveHashModeShortStringUsesGoldenBlake2b128Digest) {
    docaMemosKey key;

    ASSERT_TRUE(nixlDocaMemosEngine::resolveMemosKey(999, kShortStringKey, key, true));
    expectKeyBytes(key, kShortStringBlake2b128);
}

TEST_F(DocMemosBackendTest, ResolveHashModeLongStringUsesGoldenBlake2b128Digest) {
    docaMemosKey key;

    ASSERT_TRUE(nixlDocaMemosEngine::resolveMemosKey(999, kLongStringKey, key, true));
    expectKeyBytes(key, kLongStringBlake2b128);
}

TEST_F(DocMemosBackendTest, ResolveHashModeHexLookingStringIsHashedNotDecoded) {
    docaMemosKey key;

    ASSERT_TRUE(nixlDocaMemosEngine::resolveMemosKey(999, kHexLookingKey, key, true));
    expectKeyBytes(key, kHexLookingBlake2b128);
    EXPECT_NE(kHexDecodedKey, keyBytes(key));
}

// ============================================================================
// Transfer Preparation Tests
// ============================================================================

// A handle can be re-posted after its prior submission completes; the second
// post must succeed and produce a fresh round of submissions.
TEST_F(DocMemosBackendTest, PostXferHandleReuse) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 9101;
    remote_desc.addr = 0;
    remote_desc.len = 1024;
    remote_desc.metadataP = registerMemory(remote_desc.devId, "reuse_key");
    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);

    for (int round = 0; round < 2; round++) {
        ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle),
                  NIXL_IN_PROG)
            << "round " << round;
        nixl_status_t status = NIXL_IN_PROG;
        for (int i = 0; i < 100 && status == NIXL_IN_PROG; i++) {
            status = engine_->checkXfer(handle);
        }
        EXPECT_EQ(status, NIXL_SUCCESS) << "round " << round;
    }

    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// Exercise a NIXLBench-shaped 128-bit hex key through the real plugin STORE
// and RETRIEVE paths while the DOCA device is replaced by the in-memory mock.
TEST_F(DocMemosBackendTest, Hex128BitKeyWriteReadRoundTrip) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    std::array<char, 16> decoded_key{};
    for (size_t i = 0; i < decoded_key.size(); i++) {
        decoded_key[i] = static_cast<char>(i);
    }
    const std::string expected_device_key(decoded_key.data(), decoded_key.size());
    ASSERT_EQ(hexKey(expected_device_key), "000102030405060708090a0b0c0d0e0f");

    std::vector<uint8_t> write_buffer(256);
    for (size_t i = 0; i < write_buffer.size(); i++) {
        write_buffer[i] = static_cast<uint8_t>(i);
    }
    std::vector<uint8_t> read_buffer(write_buffer.size(), 0);

    nixlMetaDesc remote_desc;
    remote_desc.addr = 0;
    remote_desc.len = write_buffer.size();
    remote_desc.devId = 9201;
    remote_desc.metadataP = registerMemory(remote_desc.devId, expected_device_key);
    ASSERT_NE(remote_desc.metadataP, nullptr);

    auto run_transfer = [&](nixl_xfer_op_t operation, std::vector<uint8_t> &buffer) {
        nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
        nixlMetaDesc local_desc;
        local_desc.addr = reinterpret_cast<uintptr_t>(buffer.data());
        local_desc.len = buffer.size();
        local_desc.devId = 0;
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);

        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(engine_->prepXfer(operation, local_dlist, remote_dlist, "", handle),
                  NIXL_SUCCESS);
        if (handle == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        EXPECT_EQ(engine_->postXfer(operation, local_dlist, remote_dlist, "", handle),
                  NIXL_IN_PROG);

        nixl_status_t status = NIXL_IN_PROG;
        for (int i = 0; i < 100 && status == NIXL_IN_PROG; i++) {
            status = engine_->checkXfer(handle);
        }
        engine_->releaseReqH(handle);
        return status;
    };

    ASSERT_EQ(run_transfer(NIXL_WRITE, write_buffer), NIXL_SUCCESS);
    ASSERT_EQ(mock.kv_store.size(), 1u);
    const auto stored = mock.kv_store.find(expected_device_key);
    ASSERT_NE(stored, mock.kv_store.end());
    EXPECT_EQ(stored->second, write_buffer);

    ASSERT_EQ(run_transfer(NIXL_READ, read_buffer), NIXL_SUCCESS);
    EXPECT_EQ(read_buffer, write_buffer);

    engine_->deregisterMem(remote_desc.metadataP);
}

// ============================================================================
// Query Memory Tests
// ============================================================================

// queryMem submits an EXIST task per descriptor and reports presence based on
// the DOCA error callback.
TEST_F(DocMemosBackendTest, QueryMemActualKeyExists) {
    setupInitParams(false);
    createEngine();
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // The mock keys EXIST results off this map; the key string here must
    // match what resolveMemosKey produces from the hex metaInfo.
    mock.kv_store["exists"] = std::vector<uint8_t>(8, 0x42);

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    nixlBlobDesc desc;
    desc.devId = 700;
    desc.metaInfo = hexKey("exists");
    reg_dlist.addDesc(desc);

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 1u);
    EXPECT_TRUE(resp[0].has_value());
}

TEST_F(DocMemosBackendTest, QueryMemActualHashModeUsesBlake2DerivedKey) {
    setupInitParams(false);
    (*backend_params_)["convert_key_to_128bit"] = "true";
    createEngine();
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.kv_store[kLongStringBlake2b128] = std::vector<uint8_t>(8, 0x42);

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    nixlBlobDesc desc;
    desc.devId = 702;
    desc.metaInfo = kLongStringKey;
    reg_dlist.addDesc(desc);

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 1u);
    EXPECT_TRUE(resp[0].has_value());
}

TEST_F(DocMemosBackendTest, QueryMemActualDefaultModeUsesHexDecodedKey) {
    setupInitParams(false);
    createEngine();
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.kv_store[kHexDecodedKey] = std::vector<uint8_t>(8, 0x42);

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    nixlBlobDesc desc;
    desc.devId = 703;
    desc.metaInfo = kHexLookingKey;
    reg_dlist.addDesc(desc);

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 1u);
    EXPECT_TRUE(resp[0].has_value());
}

TEST_F(DocMemosBackendTest, QueryMemActualKeyDoesNotExist) {
    setupInitParams(false);
    createEngine();
    DocaMockControl::instance().auto_complete_tasks = true;

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    nixlBlobDesc desc;
    desc.devId = 701;
    desc.metaInfo = hexKey("missing");
    reg_dlist.addDesc(desc);

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 1u);
    EXPECT_FALSE(resp[0].has_value());
}

// Mix of present and absent keys in a single batch.
TEST_F(DocMemosBackendTest, QueryMemActualMixed) {
    setupInitParams(false);
    createEngine();
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.kv_store["alpha"] = std::vector<uint8_t>(4, 0x11);
    mock.kv_store["gamma"] = std::vector<uint8_t>(4, 0x33);

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    for (const char *name : {"alpha", "beta", "gamma", "delta"}) {
        nixlBlobDesc desc;
        desc.devId = 800;
        desc.metaInfo = hexKey(name);
        reg_dlist.addDesc(desc);
    }

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 4u);
    EXPECT_TRUE(resp[0].has_value()); // alpha
    EXPECT_FALSE(resp[1].has_value()); // beta
    EXPECT_TRUE(resp[2].has_value()); // gamma
    EXPECT_FALSE(resp[3].has_value()); // delta
    EXPECT_EQ(mock.task_submit_ex_call_count, 4);
    EXPECT_EQ(mock.ctx_flush_tasks_call_count, 1);
    EXPECT_TRUE(std::all_of(mock.submitted_task_flags.begin(),
                            mock.submitted_task_flags.end(),
                            [](uint32_t flags) { return flags == DOCA_TASK_SUBMIT_FLAG_NONE; }));
}

// ============================================================================
// Threading Mode Tests
// ============================================================================

TEST_F(DocMemosBackendTest, SingleThreadedMode) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Verify no progress thread is created and no locks are required
    // This is implicit in the implementation, but we can verify behavior
}

TEST_F(DocMemosBackendTest, MultiThreadedModeWithProgressThread) {
    setupInitParams(true, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Progress thread should be created
}

// Under STRICT the no-thread engine takes a real, non-recursive lock, so a
// full write, query and read must not re-enter it.
TEST_F(DocMemosBackendTest, MultiThreadedModeWithSyncMode) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);
    createEngine();
    DocaMockControl::instance().auto_complete_tasks = true;

    std::vector<uint8_t> buf(1024, 0x7e);
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(buf.data());
    local_desc.len = buf.size();
    remote_desc.devId = 900;
    remote_desc.len = buf.size();
    remote_desc.metadataP = registerMemory(remote_desc.devId, "strict_key");
    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    for (nixl_xfer_op_t op : {NIXL_WRITE, NIXL_READ}) {
        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine_->prepXfer(op, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
        nixl_status_t status = engine_->postXfer(op, local_dlist, remote_dlist, "", handle);
        while (status == NIXL_IN_PROG) {
            status = engine_->checkXfer(handle);
        }
        EXPECT_EQ(status, NIXL_SUCCESS);
        engine_->releaseReqH(handle);

        if (op == NIXL_WRITE) {
            nixl_reg_dlist_t reg_dlist(OBJ_SEG);
            nixlBlobDesc desc;
            desc.devId = remote_desc.devId;
            desc.metaInfo = hexKey("strict_key");
            reg_dlist.addDesc(desc);
            std::vector<nixl_query_resp_t> resp;
            EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_SUCCESS);
            ASSERT_EQ(resp.size(), 1u);
            EXPECT_TRUE(resp[0].has_value());
        }
    }

    engine_->deregisterMem(remote_desc.metadataP);
}

TEST_F(DocMemosBackendTest, ThreadedEngineQueuedRequestsComplete) {
    setupInitParams(true);
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.task_alloc_call_count = 0;
    mock.task_alloc_fail_after_n = 2;

    createEngine();

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> mds;
    for (int i = 0; i < 4; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);
        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 20000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "thr_key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        mds.push_back(remote_desc.metadataP);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    // Submission runs on the background progress thread, so wait until it has
    // actually hit the exhaustion window (2 successful allocs followed by at
    // least one failed retry) before lifting the fault. Clearing it eagerly
    // lets the thread drain the whole batch without ever queuing, leaving the
    // resume path untested.
    bool exhaustion_observed = false;
    for (int i = 0; i < 1000; i++) {
        {
            auto lk = DocaMockControl::lock();
            if (taskAllocCallCount() > 2) {
                exhaustion_observed = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        auto lk = DocaMockControl::lock();
        ASSERT_TRUE(exhaustion_observed)
            << "Progress thread never reached the task-pool exhaustion window";
        mock.task_alloc_fail_after_n = -1;
        mock.task_alloc_call_count = 0;
    }

    nixl_status_t status = NIXL_IN_PROG;
    for (int i = 0; i < 200; i++) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(status, NIXL_SUCCESS)
        << "Threaded engine should drain pending queue and complete all tasks";

    for (int i = 0; i < 4; i++) {
        free(buffers[i]);
        engine_->deregisterMem(mds[i]);
    }
    engine_->releaseReqH(handle);
}

// ============================================================================
// Edge Cases
// ============================================================================

// prepXfer must reject empty descriptor lists rather than allocating a
// zero-task request handle.
TEST_F(DocMemosBackendTest, EmptyDescriptorList) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);
}

// =============================================================================
// Failure Mode Tests
// =============================================================================

// Test 1: Task Pool Exhaustion - With queueing, requests succeed after resources free up
TEST_F(DocMemosBackendTest, TaskPoolExhaustion) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Configure mock to fail after 2 successful task allocations initially
    auto &mock = DocaMockControl::instance();
    mock.task_alloc_fail_after_n = 2;

    // Create 3 descriptors, but pool can only handle 2 initially
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    for (int i = 0; i < 3; i++) {
        nixlMetaDesc local_desc, remote_desc;
        local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
        local_desc.len = 1024;
        remote_desc.devId = 1000 + i; // Unique devId for each descriptor
        remote_desc.addr = 0;
        remote_desc.len = 1024;

        std::string key = "key_" + std::to_string(i);
        // Register memory and get metadata
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);

        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    // Prepare transfer - should succeed
    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    // Post transfer - first 2 tasks submit, 3rd gets queued due to resource exhaustion
    status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Verify partial submission: 2 tasks should be submitted, 1 queued
    EXPECT_EQ(submittedTaskCount(), 2) << "Should have submitted 2 tasks (partial submission)";
    EXPECT_EQ(mock.task_submit_ex_call_count, 2);
    EXPECT_EQ(mock.ctx_flush_tasks_call_count, 1)
        << "A partial batch should be flushed once before it is queued for retry";
    EXPECT_TRUE(std::all_of(mock.submitted_task_flags.begin(),
                            mock.submitted_task_flags.end(),
                            [](uint32_t flags) { return flags == DOCA_TASK_SUBMIT_FLAG_NONE; }));

    // Now allow more allocations and let progress drain the queue
    mock.task_alloc_fail_after_n = -1;
    mock.task_alloc_call_count = 0;
    mock.auto_complete_tasks = true;

    // Check status - should eventually succeed after queued task is retried
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(status, NIXL_SUCCESS) << "Request should succeed after resources become available";

    // Cleanup
    for (int i = 0; i < 3; i++) {
        free(reinterpret_cast<void *>(local_dlist[i].addr));
        engine_->deregisterMem(remote_dlist[i].metadataP);
    }
    engine_->releaseReqH(handle);
}

// Test 2: Key Not Found - RETRIEVE operation on non-existent key
TEST_F(DocMemosBackendTest, KeyNotFoundOnRetrieve) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Don't store any key in the mock KV store
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.kv_store.clear(); // Ensure store is empty

    // Try to retrieve a non-existent key
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 2000;
    remote_desc.addr = 0;
    remote_desc.len = 1024;

    std::string key = "nonexistent_key";
    remote_desc.metadataP = registerMemory(remote_desc.devId, key);

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_READ, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = engine_->postXfer(NIXL_READ, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_IN_PROG);

    // Wait for completion - should fail with NOT_FOUND
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(status, NIXL_ERR_BACKEND); // Key not found

    // Cleanup
    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// A stored value shorter than the read buffer leaves part of the buffer
// unwritten, so the read fails.
TEST_F(DocMemosBackendTest, ShortRetrieveFails) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    std::string key = "short_value";
    mock.kv_store[key] = std::vector<uint8_t>(512, 0x5a);

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 2001;
    remote_desc.addr = 0;
    remote_desc.len = 1024;
    remote_desc.metadataP = registerMemory(remote_desc.devId, key);

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_READ, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = engine_->postXfer(NIXL_READ, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_IN_PROG);

    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(status, NIXL_ERR_BACKEND);

    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// Test 3: Device I/O Error - Simulated during STORE operation
TEST_F(DocMemosBackendTest, DeviceIOErrorOnStore) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Force task error to simulate I/O failure
    auto &mock = DocaMockControl::instance();
    mock.force_task_error = true;
    mock.forced_task_error_code = DOCA_ERROR_IO_FAILED;
    mock.auto_complete_tasks = true;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 3000;
    remote_desc.addr = 0;
    remote_desc.len = 1024;

    std::string key = "test_key";
    remote_desc.metadataP = registerMemory(remote_desc.devId, key);

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_IN_PROG);

    // Wait for completion - should fail with I/O error
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(status, NIXL_ERR_BACKEND);

    // Cleanup
    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// Test 4: Task Queue Full - With queueing, requests succeed when queue has space
TEST_F(DocMemosBackendTest, TaskQueueFull) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Configure mock to fail after 1 successful submission initially
    auto &mock = DocaMockControl::instance();
    mock.task_submit_fail_after_n = 1;
    mock.auto_complete_tasks = true;

    // Create 2 descriptors
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    for (int i = 0; i < 2; i++) {
        nixlMetaDesc local_desc, remote_desc;
        local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
        local_desc.len = 1024;
        remote_desc.devId = 4000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;

        std::string key = "key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);

        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // Post transfer - first task submits, second gets queued due to queue full
    status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Now allow more submissions (simulating queue having space)
    mock.task_submit_fail_after_n = -1;
    mock.task_submit_call_count = 0;

    // Check status - should eventually succeed after queued task is retried
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(status, NIXL_SUCCESS) << "Request should succeed after queue has space";

    // Cleanup
    for (int i = 0; i < 2; i++) {
        free(reinterpret_cast<void *>(local_dlist[i].addr));
        engine_->deregisterMem(remote_dlist[i].metadataP);
    }
    engine_->releaseReqH(handle);
}

TEST_F(DocMemosBackendTest, DeviceNotFound) {
    DocaMockControl::instance().nvme_kvdev_create_result = DOCA_ERROR_NOT_FOUND;
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    EXPECT_THROW(createEngine(), std::runtime_error);
}

TEST_F(DocMemosBackendTest, GetMaxValueLenFailure) {
    DocaMockControl::instance().get_max_value_len_result = DOCA_ERROR_INVALID_VALUE;
    EXPECT_THROW(createEngine(), std::runtime_error);
}

// Value larger than the device's max value length is rejected host-side before
// any DOCA submission, not with an opaque DOCA_ERROR_DRIVER from the device.
TEST_F(DocMemosBackendTest, ValueExceedsDeviceMax) {
    DocaMockControl::instance().max_value_len = 4096; // 4 KiB device limit
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    // Allocate a buffer larger than the device's 4 KiB max.
    const size_t oversized = 8192;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(oversized));
    ASSERT_NE(local_desc.addr, static_cast<uintptr_t>(0));
    local_desc.len = oversized;
    local_desc.devId = 0;

    remote_desc.addr = 0;
    remote_desc.len = oversized;
    remote_desc.devId = 1;
    remote_desc.metadataP = registerMemory(remote_desc.devId, "key0001");
    ASSERT_NE(remote_desc.metadataP, nullptr);

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    engine_->releaseReqH(handle);
    engine_->deregisterMem(remote_desc.metadataP);
    free(reinterpret_cast<void *>(local_desc.addr));
}

// Plugin caps non-hex metaInfo at DOCA_MEMOS_MAX_OBJECT_KEY_LEN (16) bytes.
TEST_F(DocMemosBackendTest, KeyTooLarge) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 6000;
    desc.metaInfo = "this_key_is_way_too_long"; // 24 bytes, not valid hex
    nixlBackendMD *md = nullptr;

    nixl_status_t status = engine_->registerMem(desc, OBJ_SEG, md);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(md, nullptr);
}

TEST_F(DocMemosBackendTest, DeviceMaxKeyLenAbovePluginCapacity) {
    DocaMockControl::instance().max_key_len = 255;
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 6001;
    desc.metaInfo = hexKey("sixteen_byte_key");
    nixlBackendMD *md = nullptr;

    ASSERT_EQ(engine_->registerMem(desc, OBJ_SEG, md), NIXL_SUCCESS);
    engine_->deregisterMem(md);
}

TEST_F(DocMemosBackendTest, KeyLongerThanDeviceMaxIsRejected) {
    DocaMockControl::instance().max_key_len = 8;
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 6002;
    desc.metaInfo = hexKey("eight_by");
    nixlBackendMD *md = nullptr;
    ASSERT_EQ(engine_->registerMem(desc, OBJ_SEG, md), NIXL_SUCCESS);
    engine_->deregisterMem(md);

    desc.metaInfo = hexKey("nine_byte");
    md = nullptr;
    EXPECT_EQ(engine_->registerMem(desc, OBJ_SEG, md), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(md, nullptr);

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    reg_dlist.addDesc(desc);
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_ERR_BACKEND);
}

TEST_F(DocMemosBackendTest, HashModeRequiresFullWidthDeviceKeys) {
    DocaMockControl::instance().max_key_len = 8;
    (*backend_params_)["convert_key_to_128bit"] = "true";
    EXPECT_THROW(createEngine(), std::runtime_error);
}

// Verify the plugin propagates DOCA_ERROR_TOO_BIG from a STORE task as a
// backend error, mirroring how a real device would reject an oversized value.
TEST_F(DocMemosBackendTest, StoreReturnsTooBig) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.force_task_error = true;
    mock.forced_task_error_code = DOCA_ERROR_TOO_BIG;
    mock.auto_complete_tasks = true;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 6100;
    remote_desc.addr = 0;
    remote_desc.len = 1024;
    remote_desc.metadataP = registerMemory(remote_desc.devId, "too_big_key");

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    nixl_status_t status = NIXL_IN_PROG;
    for (int i = 0; i < 100 && status == NIXL_IN_PROG; i++) {
        status = engine_->checkXfer(handle);
    }
    EXPECT_EQ(status, NIXL_ERR_BACKEND);

    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// ============================================================================
// Request Queueing Tests - Comprehensive coverage for resource exhaustion
// ============================================================================

// Test that a request completes successfully after temporary resource exhaustion
TEST_F(DocMemosBackendTest, QueuedRequestSucceedsAfterResourcesFreed) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Configure mock to fail allocation after 1 task (simulating resource exhaustion)
    mock.task_alloc_call_count = 0;
    mock.task_alloc_fail_after_n = 1; // Allow 1 allocation, then fail

    // Create request with 3 tasks - first will succeed, rest will be queued
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    for (int i = 0; i < 3; i++) {
        nixlMetaDesc local_desc, remote_desc;
        local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
        local_desc.len = 1024;
        remote_desc.devId = 7000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    // Request should be partially submitted (1 task) and queued (2 remaining)

    // Now allow unlimited allocations
    mock.task_alloc_fail_after_n = -1; // Stop failing (negative = disabled)
    mock.task_alloc_call_count = 0; // Reset counter

    // Call checkXfer to progress and complete the queued request
    nixl_status_t status = NIXL_IN_PROG;
    for (int i = 0; i < 50; i++) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Request should complete successfully
    EXPECT_EQ(status, NIXL_SUCCESS);

    // Cleanup
    for (int i = 0; i < 3; i++) {
        free(reinterpret_cast<void *>(local_dlist[i].addr));
        engine_->deregisterMem(remote_dlist[i].metadataP);
    }
    engine_->releaseReqH(handle);
}

// Test FIFO ordering with multiple queued requests
TEST_F(DocMemosBackendTest, MultipleQueuedRequestsFIFOOrdering) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Allow only 2 task allocations initially
    mock.task_alloc_fail_after_n = 2;

    // Create 3 requests, each with 2 tasks (total 6 tasks, but only 2 can submit)
    std::vector<nixlBackendReqH *> handles;
    std::vector<nixl_meta_dlist_t> local_dlists;
    std::vector<nixl_meta_dlist_t> remote_dlists;
    std::vector<std::vector<void *>> buffers;
    std::vector<std::vector<nixlBackendMD *>> metadatas;

    for (int req = 0; req < 3; req++) {
        local_dlists.emplace_back(DRAM_SEG);
        remote_dlists.emplace_back(OBJ_SEG);
        buffers.emplace_back();
        metadatas.emplace_back();

        for (int i = 0; i < 2; i++) {
            nixlMetaDesc local_desc, remote_desc;
            void *buf = malloc(1024);
            // Write a unique marker to verify ordering
            memset(buf, 'A' + req, 1024);
            buffers[req].push_back(buf);

            local_desc.addr = reinterpret_cast<uintptr_t>(buf);
            local_desc.len = 1024;
            remote_desc.devId = 8000 + req * 10 + i;
            remote_desc.addr = 0;
            remote_desc.len = 1024;

            std::string key = "fifo_key_" + std::to_string(req) + "_" + std::to_string(i);
            auto *kv_md = registerMemory(remote_desc.devId, key);
            metadatas[req].push_back(kv_md);
            remote_desc.metadataP = kv_md;

            local_dlists[req].addDesc(local_desc);
            remote_dlists[req].addDesc(remote_desc);
        }

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_IN_PROG);
        handles.push_back(handle);
    }

    // At this point:
    // - Request 0: fully submitted (2 tasks)
    // - Request 1: queued (0 tasks submitted due to allocation limit)
    // - Request 2: queued (0 tasks submitted due to allocation limit)

    // Now allow unlimited allocations
    mock.task_alloc_fail_after_n = -1; // Negative = disabled
    mock.task_alloc_call_count = 0; // Reset counter

    // Check status to progress and complete everything
    for (int i = 0; i < 100; i++) { // More iterations for multiple requests
        // Check if all completed
        bool all_done = true;
        for (auto *h : handles) {
            if (engine_->checkXfer(h) == NIXL_IN_PROG) {
                all_done = false;
            }
        }
        if (all_done) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // All requests should complete successfully in FIFO order
    for (auto *h : handles) {
        EXPECT_EQ(engine_->checkXfer(h), NIXL_SUCCESS);
    }

    // Verify the KV store has all keys
    for (int req = 0; req < 3; req++) {
        for (int i = 0; i < 2; i++) {
            std::string key = "fifo_key_" + std::to_string(req) + "_" + std::to_string(i);
            EXPECT_TRUE(mock.kv_store.find(key) != mock.kv_store.end())
                << "Key " << key << " not found in KV store";
        }
    }

    // Cleanup
    for (size_t req = 0; req < handles.size(); req++) {
        for (size_t i = 0; i < buffers[req].size(); i++) {
            free(buffers[req][i]);
            engine_->deregisterMem(metadatas[req][i]);
        }
        engine_->releaseReqH(handles[req]);
    }
}

// Test that queue handles submit failures (DOCA_ERROR_FULL) correctly
TEST_F(DocMemosBackendTest, QueuedRequestsWithSubmitFailures) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Allow task allocation but fail submission after 2 tasks
    mock.task_alloc_fail_after_n = -1; // Allocations succeed (negative = disabled)
    mock.task_submit_fail_after_n = 2; // Submissions fail after 2

    // Create 2 requests, each with 3 tasks
    std::vector<nixlBackendReqH *> handles;
    std::vector<nixl_meta_dlist_t> local_dlists;
    std::vector<nixl_meta_dlist_t> remote_dlists;
    std::vector<std::vector<void *>> buffers;
    std::vector<std::vector<nixlBackendMD *>> metadatas;

    for (int req = 0; req < 2; req++) {
        local_dlists.emplace_back(DRAM_SEG);
        remote_dlists.emplace_back(OBJ_SEG);
        buffers.emplace_back();
        metadatas.emplace_back();

        for (int i = 0; i < 3; i++) {
            nixlMetaDesc local_desc, remote_desc;
            void *buf = malloc(1024);
            buffers[req].push_back(buf);

            local_desc.addr = reinterpret_cast<uintptr_t>(buf);
            local_desc.len = 1024;
            remote_desc.devId = 9000 + req * 10 + i;
            remote_desc.addr = 0;
            remote_desc.len = 1024;

            std::string key = "submit_key_" + std::to_string(req) + "_" + std::to_string(i);
            auto *kv_md = registerMemory(remote_desc.devId, key);
            metadatas[req].push_back(kv_md);
            remote_desc.metadataP = kv_md;

            local_dlists[req].addDesc(local_desc);
            remote_dlists[req].addDesc(remote_desc);
        }

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_IN_PROG);
        handles.push_back(handle);
    }

    // At this point:
    // - Request 0: 2 tasks submitted, 1 queued
    // - Request 1: 0 tasks submitted, 3 queued

    // Allow unlimited submissions
    mock.task_submit_fail_after_n = -1; // Negative = disabled
    mock.task_submit_call_count = 0; // Reset counter

    // Check status to progress and complete everything
    for (int i = 0; i < 100; i++) { // More iterations for multiple requests
        bool all_done = true;
        for (auto *h : handles) {
            if (engine_->checkXfer(h) == NIXL_IN_PROG) {
                all_done = false;
            }
        }
        if (all_done) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // All requests should complete successfully
    for (auto *h : handles) {
        EXPECT_EQ(engine_->checkXfer(h), NIXL_SUCCESS);
    }

    // Cleanup
    for (size_t req = 0; req < handles.size(); req++) {
        for (size_t i = 0; i < buffers[req].size(); i++) {
            free(buffers[req][i]);
            engine_->deregisterMem(metadatas[req][i]);
        }
        engine_->releaseReqH(handles[req]);
    }
}

// Test queue with many concurrent requests
TEST_F(DocMemosBackendTest, ManyQueuedRequestsConcurrent) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Severely limit resources: only 3 tasks can be allocated/submitted initially
    mock.task_alloc_call_count = 0; // Reset counter
    mock.task_alloc_fail_after_n = 3; // Allow 3 allocations, then fail

    const int NUM_REQUESTS = 10;
    const int TASKS_PER_REQUEST = 2;

    std::vector<nixlBackendReqH *> handles;
    std::vector<nixl_meta_dlist_t> local_dlists;
    std::vector<nixl_meta_dlist_t> remote_dlists;
    std::vector<std::vector<void *>> buffers;
    std::vector<std::vector<nixlBackendMD *>> metadatas;

    // Submit many requests quickly
    for (int req = 0; req < NUM_REQUESTS; req++) {
        local_dlists.emplace_back(DRAM_SEG);
        remote_dlists.emplace_back(OBJ_SEG);
        buffers.emplace_back();
        metadatas.emplace_back();

        for (int i = 0; i < TASKS_PER_REQUEST; i++) {
            nixlMetaDesc local_desc, remote_desc;
            void *buf = malloc(1024);
            buffers[req].push_back(buf);

            local_desc.addr = reinterpret_cast<uintptr_t>(buf);
            local_desc.len = 1024;
            remote_desc.devId = 10000 + req * 10 + i;
            remote_desc.addr = 0;
            remote_desc.len = 1024;

            std::string key = "many_key_" + std::to_string(req) + "_" + std::to_string(i);
            auto *kv_md = registerMemory(remote_desc.devId, key);
            metadatas[req].push_back(kv_md);
            remote_desc.metadataP = kv_md;

            local_dlists[req].addDesc(local_desc);
            remote_dlists[req].addDesc(remote_desc);
        }

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_IN_PROG);
        handles.push_back(handle);
    }

    // Most requests should be queued

    // Now allow unlimited resources
    mock.task_alloc_fail_after_n = -1; // Negative = disabled
    mock.task_alloc_call_count = 0; // Reset counter

    // Check status to progress until all complete
    int max_iterations = 400; // More iterations for many requests
    for (int i = 0; i < max_iterations; i++) {
        bool all_done = true;
        for (auto *h : handles) {
            if (engine_->checkXfer(h) == NIXL_IN_PROG) {
                all_done = false;
            }
        }
        if (all_done) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // All requests should complete successfully
    int success_count = 0;
    for (auto *h : handles) {
        if (engine_->checkXfer(h) == NIXL_SUCCESS) {
            success_count++;
        }
    }
    EXPECT_EQ(success_count, NUM_REQUESTS) << "Not all requests completed successfully";

    // Cleanup
    for (size_t req = 0; req < handles.size(); req++) {
        for (size_t i = 0; i < buffers[req].size(); i++) {
            free(buffers[req][i]);
            engine_->deregisterMem(metadatas[req][i]);
        }
        engine_->releaseReqH(handles[req]);
    }
}

// ============================================================================
// High Queue Depth Tests - Partial Submission and Forward Progress
// ============================================================================

// Test 1: New request can't submit any tasks - whole request gets queued
// After resources become available, the request is fully submitted
TEST_F(DocMemosBackendTest, QueuedRequestNoPartialSubmission) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Configure mock to fail ALL task allocations (simulating complete resource exhaustion)
    mock.task_alloc_call_count = 0;
    mock.task_alloc_fail_after_n = 0; // Fail immediately (0 = fail first call)

    // Create request with 5 tasks - none should be able to submit
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> metadatas;

    for (int i = 0; i < 5; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);

        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 11000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;

        std::string key = "no_partial_key_" + std::to_string(i);
        auto *kv_md = registerMemory(remote_desc.devId, key);
        metadatas.push_back(kv_md);
        remote_desc.metadataP = kv_md;

        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);

    // Post transfer - should return IN_PROG even though no tasks submitted (request is queued)
    nixl_status_t status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_IN_PROG) << "Request should be queued even if no tasks submitted";

    // Verify no tasks were actually submitted (allocation failed immediately on first attempt)
    // The code stops trying after the first allocation fails with DOCA_ERROR_FULL
    EXPECT_EQ(taskAllocCallCount(), 1)
        << "Should have attempted 1 allocation (stops on first failure)";
    EXPECT_EQ(submittedTaskCount(), 0) << "No tasks should be submitted";

    // Now allow unlimited allocations (simulating resources becoming available)
    mock.task_alloc_fail_after_n = -1; // Disable failure
    mock.task_alloc_call_count = 0; // Reset counter

    // Call checkXfer to trigger retry of queued request
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Request should complete successfully after resources become available
    EXPECT_EQ(status, NIXL_SUCCESS) << "Request should succeed after resources become available";
    EXPECT_EQ(submittedTaskCount(), 0) << "All tasks should have completed";

    // Cleanup
    for (size_t i = 0; i < buffers.size(); i++) {
        free(buffers[i]);
        engine_->deregisterMem(metadatas[i]);
    }
    engine_->releaseReqH(handle);
}

// Test 2: Queued request, when retried, can still only be partially submitted
// This tests that we make forward progress even when retrying queued requests
TEST_F(DocMemosBackendTest, QueuedRequestPartialRetryForwardProgress) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Configure mock to allow 2 task allocations initially, then fail
    mock.task_alloc_call_count = 0;
    mock.task_alloc_fail_after_n = 2; // Allow 2 allocations, then fail

    // Create request with 5 tasks
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> metadatas;

    for (int i = 0; i < 5; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);

        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 13000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;

        std::string key = "rp_key_" + std::to_string(i);
        auto *kv_md = registerMemory(remote_desc.devId, key);
        metadatas.push_back(kv_md);
        remote_desc.metadataP = kv_md;

        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);

    // Post transfer - should partially submit (2 tasks) and queue the rest (3 tasks)
    nixl_status_t status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Verify initial partial submission: 2 tasks submitted
    EXPECT_EQ(submittedTaskCount(), 2) << "Initial: 2 tasks should be submitted";

    // Reset counter but keep fail_after_n at 2 to simulate limited resources
    // When we retry, we should be able to submit 2 more tasks (partial progress)
    mock.task_alloc_call_count = 0; // Reset for retry attempt

    // Call checkXfer to trigger retry of queued tasks
    // This should submit 2 more tasks (partial progress), leaving 1 still queued
    status = engine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_IN_PROG) << "Should still be in progress after partial retry";

    // Verify partial retry: should have submitted 2 more tasks (total 4 now)
    // Note: The first 2 tasks may have completed, so we check that we made progress
    // by verifying that more allocations were attempted
    EXPECT_GE(taskAllocCallCount(), 2)
        << "Should have attempted at least 2 more allocations on retry";

    // Now allow unlimited allocations for final submission
    mock.task_alloc_fail_after_n = -1; // Disable failure
    mock.task_alloc_call_count = 0; // Reset counter

    // Call checkXfer again to complete remaining tasks
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Request should complete successfully
    EXPECT_EQ(status, NIXL_SUCCESS) << "Request should succeed after all tasks are submitted";

    // Cleanup
    for (size_t i = 0; i < buffers.size(); i++) {
        free(buffers[i]);
        engine_->deregisterMem(metadatas[i]);
    }
    engine_->releaseReqH(handle);
}

TEST_F(DocMemosBackendTest, PrepXferRejectsRemoteAgent) {
    createEngine();

    xferBuffers x;
    buildXfer(x, 1, 29000, "remote_agent_key_");

    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "other_agent", handle),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);

    free(x.buffers[0]);
    engine_->deregisterMem(x.mds[0]);
}

// ============================================================================
// Release While In-Flight Tests
//
// DOCA cannot abort a submitted task, and the device may still be reading or
// writing the caller's buffer, so releaseReqH() returns NIXL_ERR_REPOST_ACTIVE
// and leaves the handle with the caller until every submitted task completes.
// It drops whatever has not been submitted yet.
// ============================================================================

TEST_F(DocMemosBackendTest, ReleaseInFlightRequest) {
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;

    xferBuffers x;
    buildXfer(x, 3, 30000, "cancel_key_");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);
    EXPECT_EQ(submittedTaskCount(), 3u);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_ERR_REPOST_ACTIVE);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_ERR_REPOST_ACTIVE);

    mock.auto_complete_tasks = true;
    EXPECT_EQ(waitForCompletion(handle), NIXL_ERR_CANCELED);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    for (int i = 0; i < 3; i++) {
        free(x.buffers[i]);
        engine_->deregisterMem(x.mds[i]);
    }
}

TEST_F(DocMemosBackendTest, ReleasePendingRequest) {
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;
    mock.task_alloc_fail_after_n = 2;
    mock.task_alloc_call_count = 0;

    xferBuffers x;
    buildXfer(x, 5, 31000, "pend_key_");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);

    // 2 tasks submitted, 3 queued in pendingRequests_.
    EXPECT_EQ(submittedTaskCount(), 2u);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_ERR_REPOST_ACTIVE);

    mock.auto_complete_tasks = true;
    mock.task_alloc_fail_after_n = -1;
    EXPECT_EQ(waitForCompletion(handle), NIXL_ERR_CANCELED);
    EXPECT_EQ(mock.kv_store.count("pend_key_1"), 1u);
    EXPECT_EQ(mock.kv_store.count("pend_key_2"), 0u) << "unsubmitted tasks must be dropped";
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    for (int i = 0; i < 5; i++) {
        free(x.buffers[i]);
        engine_->deregisterMem(x.mds[i]);
    }
}

TEST_F(DocMemosBackendTest, ReleaseQueuedRequestWithNothingInFlight) {
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;
    mock.task_alloc_fail_after_n = 0;

    xferBuffers x;
    buildXfer(x, 2, 31500, "queued_key_");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);
    EXPECT_EQ(submittedTaskCount(), 0u);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    for (int i = 0; i < 2; i++) {
        free(x.buffers[i]);
        engine_->deregisterMem(x.mds[i]);
    }
}

TEST_F(DocMemosBackendTest, ReleaseCompletedRequest) {
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    xferBuffers x;
    buildXfer(x, 1, 32000, "done_key");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);
    EXPECT_EQ(waitForCompletion(handle), NIXL_SUCCESS);

    ASSERT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    free(x.buffers[0]);
    engine_->deregisterMem(x.mds[0]);
}

TEST_F(DocMemosBackendTest, ThreadedCancelBeforeProgressPickup) {
    setupInitParams(true);
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;
    // Fail every task allocation so the progress thread can never hand buf to
    // DOCA, regardless of whether it picks the request up before releaseReqH().
    // Without this the test races: a winning progress thread submits the store
    // task and the free(buf) below becomes a use-after-free.
    mock.task_alloc_fail_after_n = 0;

    createEngine();

    xferBuffers x;
    buildXfer(x, 1, 33000, "cancel_qd_key_0");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);

    // Depending on whether the progress thread has picked the request up, this
    // is an immediate delete or an abort the thread reconciles first.
    ASSERT_EQ(releaseWhenDrained(handle), NIXL_SUCCESS);

    free(x.buffers[0]);
    engine_->deregisterMem(x.mds[0]);
}

// A fatal allocation failure after 2 submitted tasks caps totalTasks_ at the
// in-flight count plus the failure. Releasing must neither free the request
// before those callbacks run nor lose the failure status.
TEST_F(DocMemosBackendTest, CancelAfterFatalSubmissionError_NoThread) {
    setupInitParams(false);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;
    mock.task_alloc_fail_after_n = 2;
    mock.task_alloc_fail_error = DOCA_ERROR_BAD_STATE;

    xferBuffers x;
    buildXfer(x, 3, 40000, "fatal_key_");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);
    EXPECT_EQ(submittedTaskCount(), 2u);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_ERR_REPOST_ACTIVE);

    mock.auto_complete_tasks = true;
    EXPECT_EQ(waitForCompletion(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(submittedTaskCount(), 0u);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    for (int i = 0; i < 3; i++) {
        free(x.buffers[i]);
        engine_->deregisterMem(x.mds[i]);
    }
}

TEST_F(DocMemosBackendTest, CancelAfterFatalSubmissionError_Threaded) {
    setupInitParams(true);
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;
    mock.task_alloc_fail_after_n = 2;
    mock.task_alloc_fail_error = DOCA_ERROR_BAD_STATE;

    createEngine();

    xferBuffers x;
    buildXfer(x, 3, 41000, "thr_fatal_key_");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);

    for (int i = 0; i < 200; i++) {
        if (taskAllocCallCount() >= 3) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_GE(taskAllocCallCount(), 3) << "Progress thread should have attempted 3 allocations";
    EXPECT_EQ(submittedTaskCount(), 2u);

    // Release while callbacks start firing, so the abort interleaves with them.
    {
        auto lk = DocaMockControl::lock();
        mock.auto_complete_tasks = true;
    }
    ASSERT_EQ(releaseWhenDrained(handle), NIXL_SUCCESS);
    EXPECT_EQ(submittedTaskCount(), 0u);

    for (int i = 0; i < 3; i++) {
        free(x.buffers[i]);
        engine_->deregisterMem(x.mds[i]);
    }
}

// Allocations keep failing with FULL after 2 tasks, so the request can only
// complete if the abort drops its 3 queued descriptors.
TEST_F(DocMemosBackendTest, CancelWithInflightAndPendingTasks_Threaded) {
    setupInitParams(true);
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;
    mock.task_alloc_fail_after_n = 2;
    mock.task_alloc_fail_error = DOCA_ERROR_FULL;

    createEngine();

    xferBuffers x;
    buildXfer(x, 5, 42000, "pend_infl_key_");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);

    for (int i = 0; i < 200; i++) {
        if (submittedTaskCount() >= 2) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(submittedTaskCount(), 2u);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_ERR_REPOST_ACTIVE);

    {
        auto lk = DocaMockControl::lock();
        mock.auto_complete_tasks = true;
    }
    EXPECT_EQ(waitForCompletion(handle), NIXL_ERR_CANCELED);
    {
        auto lk = DocaMockControl::lock();
        EXPECT_EQ(mock.kv_store.count("pend_infl_key_2"), 0u);
    }
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    for (int i = 0; i < 5; i++) {
        free(x.buffers[i]);
        engine_->deregisterMem(x.mds[i]);
    }
}

// doca_pe_progress() returning 0 doesn't mean nothing is in flight. The
// destructor must keep polling until the context reports no tasks.
TEST_F(DocMemosBackendTest, DestructorDrainsPastIdleProgress) {
    setupInitParams(false);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;

    xferBuffers x;
    buildXfer(x, 3, 43000, "dtor_key_");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_ERR_REPOST_ACTIVE);

    mock.auto_complete_tasks = true;
    mock.pe_progress_stall_calls = 5;
    for (auto *md : x.mds) {
        engine_->deregisterMem(md);
    }
    engine_.reset();

    EXPECT_EQ(submittedTaskCount(), 0u);
    EXPECT_EQ(mock.kv_store.count("dtor_key_2"), 1u);

    // The engine never took ownership of the aborted handle.
    delete handle;
    for (auto *buf : x.buffers) {
        free(buf);
    }
}

TEST_F(DocMemosBackendTest, ThreadedDestructorDrainsAbortedRequest) {
    setupInitParams(true);
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;

    createEngine();

    xferBuffers x;
    buildXfer(x, 3, 44000, "thr_dtor_key_");

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, x.local, x.remote, "", handle), NIXL_IN_PROG);

    for (int i = 0; i < 200; i++) {
        if (submittedTaskCount() >= 3) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_ERR_REPOST_ACTIVE);

    {
        auto lk = DocaMockControl::lock();
        mock.auto_complete_tasks = true;
    }
    for (auto *md : x.mds) {
        engine_->deregisterMem(md);
    }
    engine_.reset();

    EXPECT_EQ(submittedTaskCount(), 0u);

    delete handle;
    for (auto *buf : x.buffers) {
        free(buf);
    }
}

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
