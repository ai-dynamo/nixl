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

// ScalityObjEngineImpl with a mock REST client and a fake RDMA descriptor
// provider: registration, how a transfer becomes requests, and how their outcomes
// become the transfer's status. Local buffers are never touched, so their
// addresses are arbitrary.

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "nixl_types.h"
#include "rest_accel/scality_ai_connector/engine_impl.h"

namespace gtest::obj {
namespace {

    // One request issued by the engine, with the callback that completes it.
    struct restCall {
        enum class kind { PUT, GET, BODY };
        kind method;
        std::string key;
        uintptr_t addr = 0;
        size_t len = 0;
        size_t offset = 0;
        std::string rdmaDesc;
        bool pastEndOk = false;
        std::function<void(bool)> done;
    };

    // Records every request; the test completes them when and how it wants.
    // HEAD answers come from `exists`; a key missing from it is an error.
    class mockRestClient : public iRestClient {
    public:
        std::vector<restCall> calls;
        std::map<std::string, std::optional<bool>> exists;

        void
        putObjectRdmaAsync(std::string_view key,
                           uintptr_t data_ptr,
                           size_t data_len,
                           size_t offset,
                           std::string_view rdma_desc,
                           put_object_callback_t callback) override {
            calls.push_back({restCall::kind::PUT,
                             std::string(key),
                             data_ptr,
                             data_len,
                             offset,
                             std::string(rdma_desc),
                             false,
                             std::move(callback)});
        }

        void
        getObjectRdmaAsync(std::string_view key,
                           uintptr_t data_ptr,
                           size_t data_len,
                           size_t offset,
                           std::string_view rdma_desc,
                           bool past_end_ok,
                           get_object_callback_t callback) override {
            calls.push_back({restCall::kind::GET,
                             std::string(key),
                             data_ptr,
                             data_len,
                             offset,
                             std::string(rdma_desc),
                             past_end_ok,
                             std::move(callback)});
        }

        void
        getObjectBodyAsync(std::string_view key,
                           void *dst,
                           size_t data_len,
                           size_t offset,
                           get_object_callback_t callback) override {
            calls.push_back({restCall::kind::BODY,
                             std::string(key),
                             reinterpret_cast<uintptr_t>(dst),
                             data_len,
                             offset,
                             "",
                             false,
                             std::move(callback)});
        }

        void
        checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) override {
            auto it = exists.find(std::string(key));
            callback(it == exists.end() ? std::nullopt : it->second);
        }

        void
        complete(size_t i, bool ok) {
            calls.at(i).done(ok);
        }

        void
        completeAll(bool ok) {
            for (auto &call : calls) {
                call.done(ok);
            }
        }
    };

    // Registers nothing for real. A descriptor names the range it describes, so
    // a test can check which bytes each request covers.
    class fakeDescriptorProvider : public iRdmaDescriptorProvider {
    public:
        struct registration {
            uintptr_t ptr;
            size_t size;
            int devId;
        };

        std::vector<registration> registered;
        std::vector<uintptr_t> deregistered;
        uintptr_t failAddr = 0; ///< makeDescriptor fails for a range starting here

        static std::string
        descriptorFor(uintptr_t ptr, size_t size) {
            std::ostringstream os;
            os << "desc@" << std::hex << ptr << std::dec << "+" << size;
            return os.str();
        }

        bool
        isConnected() const override {
            return true;
        }

        nixl_status_t
        registerMemory(void *ptr, size_t size, int dev_id) override {
            registered.push_back({reinterpret_cast<uintptr_t>(ptr), size, dev_id});
            return NIXL_SUCCESS;
        }

        nixl_status_t
        deregisterMemory(void *ptr) override {
            deregistered.push_back(reinterpret_cast<uintptr_t>(ptr));
            return NIXL_SUCCESS;
        }

        std::string
        makeDescriptor(void *ptr, size_t size) override {
            const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            return (failAddr != 0 && addr == failAddr) ? std::string() : descriptorFor(addr, size);
        }
    };

    constexpr uintptr_t local_addr = 0x100000;
    constexpr uint64_t obj_dev = 7;
    const std::string test_agent = "agent";

    struct range {
        uintptr_t addr;
        size_t len;
        uint64_t devId;
    };

    class scalityEngineTest : public testing::Test {
    protected:
        // split_size is small so a split transfer stays small.
        void
        makeEngine(const nixl_b_params_t &extra = {}) {
            params_ = {{"split_size", "4096"}};
            for (const auto &kv : extra) {
                params_[kv.first] = kv.second;
            }
            init_.localAgent = test_agent;
            init_.customParams = &params_;
            rest_ = std::make_shared<mockRestClient>();
            provider_ = std::make_shared<fakeDescriptorProvider>();
            engine_ = std::make_unique<ScalityObjEngineImpl>(&init_, rest_, provider_);
        }

        void
        TearDown() override {
            for (nixlBackendMD *md : mds_) {
                engine_->deregisterMem(md);
            }
        }

        nixl_status_t
        registerMem(nixl_mem_t mem,
                    uintptr_t addr,
                    size_t len,
                    uint64_t dev_id,
                    const std::string &meta_info = "") {
            nixlBlobDesc desc = {};
            desc.addr = addr;
            desc.len = len;
            desc.devId = dev_id;
            desc.metaInfo = meta_info;
            nixlBackendMD *md = nullptr;
            const nixl_status_t st = engine_->registerMem(desc, mem, md);
            if (st == NIXL_SUCCESS) {
                mds_.push_back(md);
            }
            return st;
        }

        // Registers the object segment (key "obj") and a local buffer.
        void
        registerDefaults(nixl_mem_t local_mem = DRAM_SEG, uint64_t local_dev = 0) {
            ASSERT_EQ(registerMem(OBJ_SEG, 0, 0, obj_dev, "obj"), NIXL_SUCCESS);
            ASSERT_EQ(registerMem(local_mem, local_addr, 1 << 20, local_dev), NIXL_SUCCESS);
        }

        static nixl_meta_dlist_t
        dlist(nixl_mem_t mem, const std::vector<range> &ranges) {
            nixl_meta_dlist_t list(mem);
            for (const range &r : ranges) {
                list.addDesc(nixlMetaDesc(r.addr, r.len, r.devId));
            }
            return list;
        }

        nixl_status_t
        prep(nixl_xfer_op_t op, const nixl_meta_dlist_t &local, const nixl_meta_dlist_t &remote) {
            return engine_->prepXfer(op, local, remote, test_agent, test_agent, handle_, nullptr);
        }

        nixl_status_t
        post(nixl_xfer_op_t op, const nixl_meta_dlist_t &local, const nixl_meta_dlist_t &remote) {
            return engine_->postXfer(op, local, remote, test_agent, handle_, nullptr);
        }

        void
        releaseHandle() {
            if (handle_ != nullptr) {
                EXPECT_EQ(engine_->releaseReqH(handle_), NIXL_SUCCESS);
                handle_ = nullptr;
            }
        }

        nixl_b_params_t params_;
        nixlBackendInitParams init_;
        std::shared_ptr<mockRestClient> rest_;
        std::shared_ptr<fakeDescriptorProvider> provider_;
        std::unique_ptr<ScalityObjEngineImpl> engine_;
        std::vector<nixlBackendMD *> mds_;
        nixlBackendReqH *handle_ = nullptr;
    };

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

TEST_F(scalityEngineTest, VramPassesGpuOrdinalAsAffinityHint) {
    makeEngine();
    ASSERT_EQ(registerMem(VRAM_SEG, local_addr, 4096, 3), NIXL_SUCCESS);
    ASSERT_EQ(registerMem(DRAM_SEG, local_addr + 0x10000, 4096, 0), NIXL_SUCCESS);

    ASSERT_EQ(provider_->registered.size(), 2u);
    EXPECT_EQ(provider_->registered[0].devId, 3) << "VRAM must pass its GPU ordinal";
    EXPECT_EQ(provider_->registered[1].devId, -1) << "host memory has no affinity";
    EXPECT_EQ(provider_->registered[0].size, 4096u);
}

TEST_F(scalityEngineTest, DeregisterReleasesTheRegistration) {
    makeEngine();
    ASSERT_EQ(registerMem(DRAM_SEG, local_addr, 4096, 0), NIXL_SUCCESS);
    ASSERT_EQ(engine_->deregisterMem(mds_.back()), NIXL_SUCCESS);
    mds_.pop_back();

    ASSERT_EQ(provider_->deregistered.size(), 1u);
    EXPECT_EQ(provider_->deregistered[0], local_addr);
}

// An object segment's key is its metaInfo, or its devId when no key was given.
TEST_F(scalityEngineTest, ObjectKeyIsMetaInfoElseDevId) {
    makeEngine();
    ASSERT_EQ(registerMem(OBJ_SEG, 0, 0, 5, ""), NIXL_SUCCESS);
    ASSERT_EQ(registerMem(OBJ_SEG, 0, 0, 6, "named"), NIXL_SUCCESS);
    ASSERT_EQ(registerMem(DRAM_SEG, local_addr, 1 << 20, 0), NIXL_SUCCESS);

    const auto local = dlist(DRAM_SEG, {{local_addr, 100, 0}, {local_addr + 100, 100, 0}});
    const auto remote = dlist(OBJ_SEG, {{0, 100, 5}, {0, 100, 6}});
    ASSERT_EQ(prep(NIXL_READ, local, remote), NIXL_SUCCESS);
    ASSERT_EQ(post(NIXL_READ, local, remote), NIXL_IN_PROG);

    ASSERT_EQ(rest_->calls.size(), 2u);
    EXPECT_EQ(rest_->calls[0].key, "5");
    EXPECT_EQ(rest_->calls[1].key, "named");
    rest_->completeAll(true);
    releaseHandle();
}

// ---------------------------------------------------------------------------
// How a transfer becomes requests
// ---------------------------------------------------------------------------

// A READ is cut on split_size boundaries of the object's own offset space: a
// short head up to the first boundary, then aligned pieces. Every piece gets its
// own RDMA descriptor, and all but the first may start past the end of the object.
TEST_F(scalityEngineTest, ReadIsSplitOnObjectAlignedBoundaries) {
    makeEngine();
    registerDefaults();

    const auto local = dlist(DRAM_SEG, {{local_addr, 10000, 0}});
    const auto remote = dlist(OBJ_SEG, {{1000, 10000, obj_dev}});
    ASSERT_EQ(prep(NIXL_READ, local, remote), NIXL_SUCCESS);
    ASSERT_EQ(post(NIXL_READ, local, remote), NIXL_IN_PROG);

    struct expected {
        uintptr_t addr;
        size_t len;
        size_t offset;
        bool pastEndOk;
    };

    const std::vector<expected> want = {
        {local_addr, 3096, 1000, false},
        {local_addr + 3096, 4096, 4096, true},
        {local_addr + 7192, 2808, 8192, true},
    };
    ASSERT_EQ(rest_->calls.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i) {
        const restCall &call = rest_->calls[i];
        EXPECT_EQ(call.method, restCall::kind::GET) << "piece " << i;
        EXPECT_EQ(call.key, "obj") << "piece " << i;
        EXPECT_EQ(call.addr, want[i].addr) << "piece " << i;
        EXPECT_EQ(call.len, want[i].len) << "piece " << i;
        EXPECT_EQ(call.offset, want[i].offset) << "piece " << i;
        EXPECT_EQ(call.pastEndOk, want[i].pastEndOk) << "piece " << i;
        EXPECT_EQ(call.rdmaDesc, fakeDescriptorProvider::descriptorFor(want[i].addr, want[i].len))
            << "piece " << i;
    }
    rest_->completeAll(true);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_SUCCESS);
    releaseHandle();
}

TEST_F(scalityEngineTest, SplitSizeZeroDisablesSplitting) {
    makeEngine({{"split_size", "0"}});
    registerDefaults();

    const auto local = dlist(DRAM_SEG, {{local_addr, 10000, 0}});
    const auto remote = dlist(OBJ_SEG, {{1000, 10000, obj_dev}});
    ASSERT_EQ(prep(NIXL_READ, local, remote), NIXL_SUCCESS);
    ASSERT_EQ(post(NIXL_READ, local, remote), NIXL_IN_PROG);

    ASSERT_EQ(rest_->calls.size(), 1u);
    EXPECT_EQ(rest_->calls[0].len, 10000u);
    EXPECT_FALSE(rest_->calls[0].pastEndOk);
    rest_->completeAll(true);
    releaseHandle();
}

// A PUT writes the whole object, so a WRITE is one request whatever split_size.
TEST_F(scalityEngineTest, WriteIsNeverSplit) {
    makeEngine();
    registerDefaults();

    const auto local = dlist(DRAM_SEG, {{local_addr, 10000, 0}});
    const auto remote = dlist(OBJ_SEG, {{0, 10000, obj_dev}});
    ASSERT_EQ(prep(NIXL_WRITE, local, remote), NIXL_SUCCESS);
    ASSERT_EQ(post(NIXL_WRITE, local, remote), NIXL_IN_PROG);

    ASSERT_EQ(rest_->calls.size(), 1u);
    EXPECT_EQ(rest_->calls[0].method, restCall::kind::PUT);
    EXPECT_EQ(rest_->calls[0].len, 10000u);
    EXPECT_EQ(rest_->calls[0].rdmaDesc, fakeDescriptorProvider::descriptorFor(local_addr, 10000));
    rest_->completeAll(true);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_SUCCESS);
    releaseHandle();
}

TEST_F(scalityEngineTest, WriteAtNonZeroObjectOffsetIsRejected) {
    makeEngine();
    registerDefaults();

    const auto local = dlist(DRAM_SEG, {{local_addr, 100, 0}});
    const auto remote = dlist(OBJ_SEG, {{4096, 100, obj_dev}});
    EXPECT_EQ(prep(NIXL_WRITE, local, remote), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_TRUE(rest_->calls.empty());
}

// An empty READ descriptor needs no request; an empty WRITE is refused.
TEST_F(scalityEngineTest, ZeroLengthDescriptors) {
    makeEngine();
    registerDefaults();

    const auto local = dlist(DRAM_SEG, {{local_addr, 0, 0}, {local_addr + 100, 100, 0}});
    const auto remote = dlist(OBJ_SEG, {{0, 0, obj_dev}, {0, 100, obj_dev}});
    ASSERT_EQ(prep(NIXL_READ, local, remote), NIXL_SUCCESS);
    ASSERT_EQ(post(NIXL_READ, local, remote), NIXL_IN_PROG);
    ASSERT_EQ(rest_->calls.size(), 1u) << "the empty descriptor must not become a request";
    EXPECT_EQ(rest_->calls[0].len, 100u);
    rest_->completeAll(true);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_SUCCESS);
    releaseHandle();

    const auto empty_local = dlist(DRAM_SEG, {{local_addr, 0, 0}});
    const auto empty_remote = dlist(OBJ_SEG, {{0, 0, obj_dev}});
    ASSERT_EQ(prep(NIXL_READ, empty_local, empty_remote), NIXL_SUCCESS);
    post(NIXL_READ, empty_local, empty_remote);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_SUCCESS) << "an all-empty READ completes at once";
    releaseHandle();

    EXPECT_EQ(prep(NIXL_WRITE, empty_local, empty_remote), NIXL_ERR_NOT_SUPPORTED);
}

TEST_F(scalityEngineTest, InvalidTransfersAreRejected) {
    makeEngine();
    registerDefaults();

    const auto local = dlist(DRAM_SEG, {{local_addr, 100, 0}});
    EXPECT_EQ(prep(NIXL_READ, local, dlist(OBJ_SEG, {{0, 100, 99}})), NIXL_ERR_INVALID_PARAM)
        << "an unregistered object segment must be rejected";
    EXPECT_EQ(
        prep(NIXL_READ, dlist(OBJ_SEG, {{0, 100, obj_dev}}), dlist(OBJ_SEG, {{0, 100, obj_dev}})),
        NIXL_ERR_INVALID_PARAM)
        << "the local side must be DRAM or VRAM";
    EXPECT_EQ(prep(NIXL_READ, local, dlist(DRAM_SEG, {{local_addr, 100, 0}})),
              NIXL_ERR_INVALID_PARAM)
        << "the remote side must be an object segment";
    EXPECT_EQ(prep(NIXL_READ, local, dlist(OBJ_SEG, {{0, 100, obj_dev}, {0, 100, obj_dev}})),
              NIXL_ERR_INVALID_PARAM)
        << "descriptor counts must match";
    EXPECT_TRUE(rest_->calls.empty());
}

TEST_F(scalityEngineTest, MissingRdmaDescriptorFailsPreparation) {
    makeEngine();
    registerDefaults();
    provider_->failAddr = local_addr;

    const auto local = dlist(DRAM_SEG, {{local_addr, 100, 0}});
    const auto remote = dlist(OBJ_SEG, {{0, 100, obj_dev}});
    EXPECT_EQ(prep(NIXL_READ, local, remote), NIXL_ERR_BACKEND);
}

// ---------------------------------------------------------------------------
// Transfer status
// ---------------------------------------------------------------------------

// The transfer stays in progress until every request has finished, even after
// one failed, and then reports the first error. Reporting it earlier would let
// the caller reuse the buffer while the others still write into it.
TEST_F(scalityEngineTest, StatusWaitsForEveryRequestThenReportsFirstError) {
    makeEngine();
    registerDefaults();

    const auto local = dlist(DRAM_SEG, {{local_addr, 3 * 4096, 0}});
    const auto remote = dlist(OBJ_SEG, {{0, 3 * 4096, obj_dev}});
    ASSERT_EQ(prep(NIXL_READ, local, remote), NIXL_SUCCESS);
    ASSERT_EQ(post(NIXL_READ, local, remote), NIXL_IN_PROG);
    ASSERT_EQ(rest_->calls.size(), 3u);

    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_IN_PROG);
    rest_->complete(1, false);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_IN_PROG) << "two requests are still running";
    rest_->complete(0, true);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_IN_PROG) << "one request is still running";
    rest_->complete(2, true);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_ERR_BACKEND) << "the error must stay reported";
    releaseHandle();
}

// A prepared handle can be posted again; each post reports its own outcome.
TEST_F(scalityEngineTest, RepostReportsItsOwnOutcome) {
    makeEngine();
    registerDefaults();

    const auto local = dlist(DRAM_SEG, {{local_addr, 100, 0}});
    const auto remote = dlist(OBJ_SEG, {{0, 100, obj_dev}});
    ASSERT_EQ(prep(NIXL_READ, local, remote), NIXL_SUCCESS);

    ASSERT_EQ(post(NIXL_READ, local, remote), NIXL_IN_PROG);
    rest_->complete(0, false);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_ERR_BACKEND);

    ASSERT_EQ(post(NIXL_READ, local, remote), NIXL_IN_PROG);
    ASSERT_EQ(rest_->calls.size(), 2u);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_IN_PROG);
    rest_->complete(1, true);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_SUCCESS)
        << "the earlier failure leaked into a repost";
    releaseHandle();
}

// Releasing a handle waits for its requests: each writes into the caller's
// buffer, which the caller frees once the handle is released.
TEST_F(scalityEngineTest, ReleaseWaitsForOutstandingRequests) {
    makeEngine();
    registerDefaults();

    const auto local = dlist(DRAM_SEG, {{local_addr, 100, 0}});
    const auto remote = dlist(OBJ_SEG, {{0, 100, obj_dev}});
    ASSERT_EQ(prep(NIXL_READ, local, remote), NIXL_SUCCESS);
    ASSERT_EQ(post(NIXL_READ, local, remote), NIXL_IN_PROG);

    std::atomic<bool> released{false};
    nixlBackendReqH *handle = handle_;
    handle_ = nullptr;
    std::thread releaser([&] {
        engine_->releaseReqH(handle);
        released = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_FALSE(released.load()) << "released while a request was still running";
    rest_->complete(0, true);
    releaser.join();
    EXPECT_TRUE(released.load());
}

// ---------------------------------------------------------------------------
// dram_rdma=false: host memory read over plain HTTP
// ---------------------------------------------------------------------------

TEST_F(scalityEngineTest, DramWithoutRdmaPinsNothingAndReadsOverHttp) {
    makeEngine({{"dram_rdma", "false"}});
    registerDefaults();
    EXPECT_TRUE(provider_->registered.empty()) << "DRAM must not be registered for RDMA";

    // Not split either: there are no rails to spread a body read over.
    const auto local = dlist(DRAM_SEG, {{local_addr, 10000, 0}});
    const auto remote = dlist(OBJ_SEG, {{1000, 10000, obj_dev}});
    ASSERT_EQ(prep(NIXL_READ, local, remote), NIXL_SUCCESS);
    ASSERT_EQ(post(NIXL_READ, local, remote), NIXL_IN_PROG);

    ASSERT_EQ(rest_->calls.size(), 1u);
    EXPECT_EQ(rest_->calls[0].method, restCall::kind::BODY);
    EXPECT_EQ(rest_->calls[0].addr, local_addr);
    EXPECT_EQ(rest_->calls[0].len, 10000u);
    EXPECT_EQ(rest_->calls[0].offset, 1000u);
    rest_->completeAll(true);
    EXPECT_EQ(engine_->checkXfer(handle_), NIXL_SUCCESS);
    releaseHandle();

    EXPECT_EQ(prep(NIXL_WRITE, local, dlist(OBJ_SEG, {{0, 10000, obj_dev}})),
              NIXL_ERR_NOT_SUPPORTED)
        << "there is no plain-HTTP upload path";
}

TEST_F(scalityEngineTest, DramWithoutRdmaLeavesVramOnRdma) {
    makeEngine({{"dram_rdma", "false"}});
    ASSERT_EQ(registerMem(VRAM_SEG, local_addr, 4096, 2), NIXL_SUCCESS);
    ASSERT_EQ(provider_->registered.size(), 1u);
    EXPECT_EQ(provider_->registered[0].devId, 2);
}

// ---------------------------------------------------------------------------
// queryMem: object existence through HEAD
// ---------------------------------------------------------------------------

TEST_F(scalityEngineTest, QueryMemReportsWhichObjectsExist) {
    makeEngine();
    rest_->exists = {{"present", true}, {"absent", false}};

    nixl_reg_dlist_t descs(OBJ_SEG);
    for (const char *key : {"present", "absent"}) {
        nixlBlobDesc desc = {};
        desc.metaInfo = key;
        descs.addDesc(desc);
    }
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(descs, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 2u);
    EXPECT_TRUE(resp[0].has_value()) << "an existing object must be reported";
    EXPECT_FALSE(resp[1].has_value()) << "a missing object must not be reported";
}

TEST_F(scalityEngineTest, QueryMemFailsWhenACheckFails) {
    makeEngine();
    rest_->exists = {{"present", true}}; // "broken" is missing: its HEAD fails

    nixl_reg_dlist_t descs(OBJ_SEG);
    for (const char *key : {"present", "broken"}) {
        nixlBlobDesc desc = {};
        desc.metaInfo = key;
        descs.addDesc(desc);
    }
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(descs, resp), NIXL_ERR_BACKEND);
    ASSERT_EQ(resp.size(), 2u);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_FALSE(resp[1].has_value());
}

} // namespace gtest::obj
