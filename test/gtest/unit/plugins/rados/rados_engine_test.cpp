/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "rados_backend.h"
#include "rados_capi.h"

namespace gtest::rados {

namespace {

    enum class stage { pending, complete_only, retired };

    struct fakeCCompletion {
        stage completion_stage = stage::pending;
        int rc = 0;
        bool waited = false;
        int releases = 0;
        bool released_unretired = false;
        bool submitted = false;
    };

    struct fakeCppCompletion {
        stage completion_stage = stage::pending;
        int rc = 0;
        bool waited = false;
        int releases = 0;
        bool released_unretired = false;
        bool submitted = false;
    };

    struct submission {
        bool is_write;
        std::string oid;
        uint64_t offset;
        uintptr_t buf;
        uint64_t len;
        fakeCCompletion *c_comp;
        fakeCppCompletion *cpp_comp;
        const librados::bufferlist *payload;
        std::string payload_at_release;
        uintptr_t payload_alias;
    };

    struct wait_gate {
        std::mutex mtx;
        std::condition_variable cv;
        bool armed = false;
        bool entered = false;
        bool released = false;
    };

    class fakeCApi final : public nixl::rados::cApi {
    public:
        fakeCApi()
            : log(std::make_shared<std::vector<std::string>>()),
              view_bind_order(std::make_shared<std::vector<librados::IoCtx *>>()),
              view_close_order(std::make_shared<std::vector<librados::IoCtx *>>()) {}

        ~fakeCApi() override {
            EXPECT_TRUE(violations.empty()) << violations;
            for (const auto &comp : c_completions) {
                EXPECT_EQ(comp->releases, 1) << "C completion leaked or double-released";
            }
            for (const auto &comp : cpp_completions) {
                EXPECT_EQ(comp->releases, 1) << "C++ completion leaked or double-released";
            }
            for (const auto &rec : clusters) {
                EXPECT_FALSE(rec.live) << "cluster handle left open at fake teardown";
            }
            for (const auto &rec : ioctxs) {
                EXPECT_FALSE(rec.live) << "raw ioctx left open at fake teardown";
            }
            for (auto *view : *view_bind_order) {
                EXPECT_EQ(std::count(view_close_order->begin(), view_close_order->end(), view), 1)
                    << "IoCtx view not closed exactly once";
            }
        }

        std::shared_ptr<std::vector<std::string>> log;
        std::shared_ptr<std::vector<librados::IoCtx *>> view_bind_order;
        std::shared_ptr<std::vector<librados::IoCtx *>> view_close_order;
        std::vector<std::pair<std::string, std::string>> options;
        std::string violations;

        int connect_rc = 0;

        int c_create_completion_fail_after = -1;
        int c_submit_fail_after = -1;
        int cpp_create_completion_fail_after = -1;
        int cpp_submit_fail_after = -1;
        int cpp_write_throw_after = -1;

        std::vector<submission> submissions;
        int c_waits = 0;
        int cpp_waits = 0;
        int c_releases = 0;
        int cpp_releases = 0;
        int c_rc_reads = 0;
        int cpp_rc_reads = 0;
        std::vector<std::string> create2_names;
        std::vector<std::string> conf_paths;
        std::vector<std::string> pools;
        std::string object_namespace;
        std::vector<librados::IoCtx *> write_ctxs;
        std::vector<size_t> read_ioctx_ordinals;
        wait_gate gate;

        void
        armWaitGate() {
            const std::lock_guard<std::mutex> lock(gate.mtx);
            gate.armed = true;
        }

        bool
        waitGateEntered() {
            std::unique_lock<std::mutex> lock(gate.mtx);
            return gate.cv.wait_for(
                lock, std::chrono::seconds(30), [this] { return gate.entered; });
        }

        void
        releaseWaitGate() {
            const std::lock_guard<std::mutex> lock(gate.mtx);
            gate.released = true;
            gate.cv.notify_all();
        }

        void
        completeSubmission(size_t i, int rc) {
            ASSERT_LT(i, submissions.size());
            submissions[i].payload_at_release.clear();
            if (submissions[i].is_write) {
                ASSERT_NE(submissions[i].cpp_comp, nullptr);
                submissions[i].cpp_comp->rc = rc;
                ASSERT_NE(submissions[i].payload, nullptr);
                submissions[i].payload_at_release = submissions[i].payload->to_str();
                submissions[i].cpp_comp->completion_stage = stage::retired;
            } else {
                ASSERT_NE(submissions[i].c_comp, nullptr);
                submissions[i].c_comp->rc = rc;
                submissions[i].c_comp->completion_stage = stage::retired;
            }
        }

        void
        completeSubmission(size_t i) {
            ASSERT_LT(i, submissions.size());
            completeSubmission(i,
                               submissions[i].is_write ? 0 : static_cast<int>(submissions[i].len));
        }

        void
        completeButUnretired(size_t i) {
            ASSERT_LT(i, submissions.size());
            if (submissions[i].is_write) {
                submissions[i].cpp_comp->completion_stage = stage::complete_only;
            } else {
                submissions[i].c_comp->completion_stage = stage::complete_only;
            }
        }

        void
        retire(size_t i) {
            ASSERT_LT(i, submissions.size());
            if (submissions[i].is_write) {
                submissions[i].cpp_comp->completion_stage = stage::retired;
            } else {
                submissions[i].c_comp->completion_stage = stage::retired;
            }
        }

        int
        create2(rados_t *cluster, const char *, const char *name, uint64_t) override {
            log->push_back("create2");
            create2_names.push_back(name);
            *cluster = makeClusterToken(clusters.size());
            clusters.push_back({*cluster, true});
            return 0;
        }

        int
        conf_read_file(rados_t cluster, const char *path) override {
            log->push_back("conf_read_file");
            conf_paths.push_back(path);
            if (!requireLiveCluster(cluster)) {
                return -EINVAL;
            }
            return 0;
        }

        int
        conf_set(rados_t cluster, const char *option, const char *value) override {
            log->push_back(std::string("conf_set:") + option);
            options.emplace_back(option, value);
            if (!requireLiveCluster(cluster)) {
                return -EINVAL;
            }
            return 0;
        }

        int
        connect(rados_t cluster) override {
            log->push_back("connect");
            if (!requireLiveCluster(cluster)) {
                return -EINVAL;
            }
            if (connect_rc != 0) {
                return connect_rc;
            }
            return 0;
        }

        int
        ioctx_create(rados_t cluster, const char *pool, rados_ioctx_t *ioctx) override {
            log->push_back("ioctx_create");
            pools.push_back(pool);
            if (!requireLiveCluster(cluster)) {
                return -EINVAL;
            }
            *ioctx = makeIoctxToken(ioctxs.size());
            ioctxs.push_back({*ioctx, true});
            return 0;
        }

        void
        ioctx_set_namespace(rados_ioctx_t ioctx, const char *nspace) override {
            log->push_back("ioctx_set_namespace");
            if (!requireLiveIoctx(ioctx)) {
                return;
            }
            EXPECT_NE(nspace, nullptr);
            object_namespace = nspace;
        }

        void
        ioctx_destroy(rados_ioctx_t ioctx) override {
            log->push_back("ioctx_destroy");
            for (auto &rec : ioctxs) {
                if (rec.token == ioctx) {
                    if (!rec.live) {
                        violations += "ioctx destroyed more than once\n";
                    }
                    rec.live = false;
                    return;
                }
            }
            violations += "ioctx destroyed without a created handle\n";
        }

        void
        shutdown(rados_t cluster) override {
            log->push_back("shutdown");
            for (auto &rec : clusters) {
                if (rec.token == cluster) {
                    if (!rec.live) {
                        violations += "cluster shut down more than once\n";
                    }
                    rec.live = false;
                    return;
                }
            }
            violations += "cluster shut down without a created handle\n";
        }

        void
        ioctxFromRaw(rados_ioctx_t raw, librados::IoCtx *ctx) override {
            log->push_back("ioctx_from_raw");
            if (!requireLiveIoctx(raw)) {
                return;
            }
            EXPECT_NE(ctx, nullptr);
            view_bind_order->push_back(ctx);
        }

        void
        ioCtxClose(librados::IoCtx *ctx) override {
            log->push_back("ioctx_close");
            EXPECT_NE(ctx, nullptr);
            if (std::find(view_bind_order->begin(), view_bind_order->end(), ctx) ==
                view_bind_order->end()) {
                violations += "IoCtx view closed without having been bound\n";
            } else if (std::find(view_close_order->begin(), view_close_order->end(), ctx) !=
                       view_close_order->end()) {
                violations += "IoCtx view closed more than once\n";
            } else {
                view_close_order->push_back(ctx);
            }
            for (const auto &sub : submissions) {
                if (sub.cpp_comp != nullptr && sub.cpp_comp->releases == 0) {
                    violations += "IoCtx closed with an unreleased C++ completion\n";
                }
                if (sub.c_comp != nullptr && sub.c_comp->releases == 0) {
                    violations += "IoCtx closed with an unreleased C completion\n";
                }
            }
        }

        int
        cCreateCompletion(rados_completion_t *pc) override {
            log->push_back("c_create_completion");
            *pc = nullptr;
            if (c_create_completion_fail_after >= 0 &&
                static_cast<int>(c_completions.size()) >= c_create_completion_fail_after) {
                return -ENOMEM;
            }
            c_completions.push_back(std::make_unique<fakeCCompletion>());
            *pc = reinterpret_cast<rados_completion_t>(c_completions.back().get());
            return 0;
        }

        int
        cRead(rados_ioctx_t ioctx,
              const char *oid,
              rados_completion_t completion,
              char *buf,
              size_t len,
              uint64_t off) override {
            log->push_back("c_read");
            if (!requireLiveIoctx(ioctx)) {
                violations += "C read submitted on an unavailable ioctx\n";
                return -EINVAL;
            }
            for (size_t i = 0; i < ioctxs.size(); ++i) {
                if (ioctxs[i].token == ioctx) {
                    read_ioctx_ordinals.push_back(i);
                }
            }
            fakeCCompletion *comp = asC(completion);
            if (comp == nullptr) {
                violations += "C read submitted with a foreign completion token\n";
                return -EINVAL;
            }
            if (c_submit_fail_after >= 0 &&
                static_cast<int>(submissions.size()) >= c_submit_fail_after) {
                return -EIO;
            }
            submissions.push_back(
                {false, oid, off, reinterpret_cast<uintptr_t>(buf), len, comp, nullptr, nullptr,
                 {}, 0});
            comp->submitted = true;
            return 0;
        }

        int
        cIsCompleteAndCb(rados_completion_t completion) override {
            log->push_back("c_is_complete_and_cb");
            fakeCCompletion *comp = asC(completion);
            if (comp == nullptr) {
                violations += "C poll on a foreign completion token\n";
                return 0;
            }
            return comp->completion_stage == stage::retired ? 1 : 0;
        }

        int
        cGetReturnValue(rados_completion_t completion) override {
            log->push_back("c_get_return_value");
            fakeCCompletion *comp = asC(completion);
            if (comp == nullptr) {
                violations += "C return value read on a foreign completion token\n";
                return -EINVAL;
            }
            ++c_rc_reads;
            if (comp->completion_stage != stage::retired) {
                violations += "C return value read before complete-and-cb\n";
            }
            return comp->rc;
        }

        int
        cWaitForCompleteAndCb(rados_completion_t completion) override {
            log->push_back("c_wait_for_complete_and_cb");
            fakeCCompletion *comp = asC(completion);
            if (comp == nullptr) {
                violations += "C wait on a foreign completion token\n";
                return -EINVAL;
            }
            ++c_waits;
            comp->waited = true;
            blockOnGateIfNeeded();
            comp->completion_stage = stage::retired;
            return 0;
        }

        void
        cRelease(rados_completion_t completion) override {
            log->push_back("c_release");
            fakeCCompletion *comp = asC(completion);
            if (comp == nullptr) {
                violations += "C release on a foreign completion token\n";
                return;
            }
            ++c_releases;
            ++comp->releases;
            if (comp->releases > 1) {
                violations += "C completion released more than once\n";
            }
            if (comp->submitted && comp->completion_stage != stage::retired) {
                violations += "submitted C completion released without wait/proof\n";
                comp->released_unretired = true;
            }
        }

        int
        cppCreateCompletion(librados::AioCompletion **pc) override {
            log->push_back("cpp_create_completion");
            *pc = nullptr;
            if (cpp_create_completion_fail_after >= 0 &&
                static_cast<int>(cpp_completions.size()) >= cpp_create_completion_fail_after) {
                return -ENOMEM;
            }
            cpp_completions.push_back(std::make_unique<fakeCppCompletion>());
            *pc = reinterpret_cast<librados::AioCompletion *>(cpp_completions.back().get());
            return 0;
        }

        int
        cppWrite(librados::IoCtx *ctx,
                 const char *oid,
                 librados::AioCompletion *completion,
                 const librados::bufferlist &bl,
                 size_t len,
                 uint64_t off) override {
            log->push_back("cpp_write");
            EXPECT_NE(ctx, nullptr);
            write_ctxs.push_back(ctx);
            fakeCppCompletion *comp = asCpp(completion);
            if (comp == nullptr) {
                violations += "C++ write submitted with a foreign completion token\n";
                return -EINVAL;
            }
            if (bl.length() != len) {
                violations += "C++ write payload length does not match the request\n";
            }
            if (cpp_write_throw_after >= 0 &&
                static_cast<int>(submissions.size()) >= cpp_write_throw_after) {
                throw std::bad_alloc();
            }
            if (cpp_submit_fail_after >= 0 &&
                static_cast<int>(submissions.size()) >= cpp_submit_fail_after) {
                return -EIO;
            }
            submissions.push_back(
                {true, oid, off, 0, len, nullptr, comp, &bl, {}, PayloadAddr(bl)});
            comp->submitted = true;
            return 0;
        }

        int
        cppIsCompleteAndCb(librados::AioCompletion *completion) override {
            log->push_back("cpp_is_complete_and_cb");
            fakeCppCompletion *comp = asCpp(completion);
            if (comp == nullptr) {
                violations += "C++ poll on a foreign completion token\n";
                return 0;
            }
            return comp->completion_stage == stage::retired ? 1 : 0;
        }

        int
        cppGetReturnValue(librados::AioCompletion *completion) override {
            log->push_back("cpp_get_return_value");
            fakeCppCompletion *comp = asCpp(completion);
            if (comp == nullptr) {
                violations += "C++ return value read on a foreign completion token\n";
                return -EINVAL;
            }
            ++cpp_rc_reads;
            if (comp->completion_stage != stage::retired) {
                violations += "C++ return value read before complete-and-cb\n";
            }
            return comp->rc;
        }

        int
        cppWaitForCompleteAndCb(librados::AioCompletion *completion) override {
            log->push_back("cpp_wait_for_complete_and_cb");
            fakeCppCompletion *comp = asCpp(completion);
            if (comp == nullptr) {
                violations += "C++ wait on a foreign completion token\n";
                return -EINVAL;
            }
            ++cpp_waits;
            comp->waited = true;
            blockOnGateIfNeeded();
            comp->completion_stage = stage::retired;
            return 0;
        }

        void
        cppRelease(librados::AioCompletion *completion) override {
            log->push_back("cpp_release");
            fakeCppCompletion *comp = asCpp(completion);
            if (comp == nullptr) {
                violations += "C++ release on a foreign completion token\n";
                return;
            }
            ++cpp_releases;
            ++comp->releases;
            if (comp->releases > 1) {
                violations += "C++ completion released more than once\n";
            }
            if (comp->submitted && comp->completion_stage != stage::retired) {
                violations += "submitted C++ completion released without wait/proof\n";
                comp->released_unretired = true;
            }
            for (auto &sub : submissions) {
                if (sub.cpp_comp == comp) {
                    sub.payload_at_release =
                        sub.payload == nullptr ? std::string{} : sub.payload->to_str();
                }
            }
        }

    private:
        void
        blockOnGateIfNeeded() {
            std::unique_lock<std::mutex> lock(gate.mtx);
            if (!gate.armed) {
                return;
            }
            gate.entered = true;
            gate.cv.notify_all();
            gate.cv.wait(lock, [this] { return gate.released; });
        }

        static uintptr_t
        PayloadAddr(const librados::bufferlist &bl) {
            for (const auto &b : bl.buffers()) {
                return reinterpret_cast<uintptr_t>(b.c_str());
            }
            return 0;
        }

        fakeCCompletion *
        asC(rados_completion_t completion) {
            auto *raw = reinterpret_cast<fakeCCompletion *>(completion);
            for (const auto &comp : c_completions) {
                if (comp.get() == raw) {
                    return raw;
                }
            }
            return nullptr;
        }

        fakeCppCompletion *
        asCpp(librados::AioCompletion *completion) {
            auto *raw = reinterpret_cast<fakeCppCompletion *>(completion);
            for (const auto &comp : cpp_completions) {
                if (comp.get() == raw) {
                    return raw;
                }
            }
            return nullptr;
        }

        std::vector<std::unique_ptr<fakeCCompletion>> c_completions;
        std::vector<std::unique_ptr<fakeCppCompletion>> cpp_completions;

        struct cluster_rec {
            rados_t token;
            bool live;
        };
        struct ioctx_rec {
            rados_ioctx_t token;
            bool live;
        };
        std::vector<cluster_rec> clusters;
        std::vector<ioctx_rec> ioctxs;

        static rados_t
        makeClusterToken(size_t ordinal) {
            return reinterpret_cast<rados_t>(static_cast<uintptr_t>(0xc10e0000u +
                                                                    0x100u * (ordinal + 1)));
        }

        static rados_ioctx_t
        makeIoctxToken(size_t ordinal) {
            return reinterpret_cast<rados_ioctx_t>(static_cast<uintptr_t>(0x10c70002u +
                                                                          0x100u * (ordinal + 1)));
        }

        bool
        requireLiveCluster(rados_t cluster) {
            for (auto &rec : clusters) {
                if (rec.token == cluster) {
                    if (!rec.live) {
                        violations += "cluster operation on a shut-down handle\n";
                    }
                    return rec.live;
                }
            }
            violations += "cluster operation on an unknown handle\n";
            return false;
        }

        bool
        requireLiveIoctx(rados_ioctx_t ioctx) {
            for (auto &rec : ioctxs) {
                if (rec.token == ioctx) {
                    if (!rec.live) {
                        violations += "ioctx operation on a destroyed handle\n";
                    }
                    return rec.live;
                }
            }
            violations += "ioctx operation on an unknown handle\n";
            return false;
        }
    };

    nixl_b_params_t
    validBase() {
        return {{"conf", "fixture.conf"}, {"pool", "qa-pool"}, {"name", "client.qa"}};
    }

    // Survives the engine's adopt-or-drop decision so sleeps and timer
    // retention stay observable either way.
    struct timer_state {
        std::vector<uint64_t> budgets;
        bool destroyed = false;
    };

    // Records every intended sleep and its verbatim budget; never really sleeps.
    class fakeStatusWaitTimer final : public nixl::rados::statusWaitTimer {
    public:
        explicit fakeStatusWaitTimer(std::shared_ptr<timer_state> state)
            : state_(std::move(state)) {}

        fakeStatusWaitTimer(const fakeStatusWaitTimer &) = delete;
        fakeStatusWaitTimer &
        operator=(const fakeStatusWaitTimer &) = delete;

        ~fakeStatusWaitTimer() override {
            state_->destroyed = true;
        }

        void
        sleep(uint64_t budget_us) override {
            state_->budgets.push_back(budget_us);
        }

    private:
        std::shared_ptr<timer_state> state_;
    };

    void
    add_pair(nixl_meta_dlist_t &local, nixl_meta_dlist_t &remote, uintptr_t addr, uint64_t len,
             uint64_t object_offset, uint64_t dev_id) {
        local.addDesc(nixlMetaDesc(addr, len, 0));
        remote.addDesc(nixlMetaDesc(object_offset, len, dev_id));
    }

    // One DRAM/object pair over a fixture-owned buffer; mutate buffer in place so
    // descriptor-captured addresses stay valid.
    struct one_pair {
        one_pair(uint64_t object_offset, uint64_t length, uint64_t dev_id, unsigned char fill)
            : buffer(static_cast<size_t>(length), fill), local(DRAM_SEG), remote(OBJ_SEG) {
            add_pair(local,
                     remote,
                     reinterpret_cast<uintptr_t>(buffer.data()),
                     length,
                     object_offset,
                     dev_id);
        }

        std::vector<unsigned char> buffer;
        nixl_meta_dlist_t local;
        nixl_meta_dlist_t remote;
    };

} // namespace

class radosEngineFixture : public testing::Test {
protected:
    void
    SetUp() override {
        init_params_.localAgent = "test-agent";
        init_params_.type = "RADOS";
        init_params_.customParams = &custom_params_;
        init_params_.enableProgTh = false;
        init_params_.pthrDelay = 0;
        init_params_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    }

    std::shared_ptr<std::vector<std::string>> log;

    std::unique_ptr<nixlRadosEngine>
    makeEngine(fakeCApi **retained_fake = nullptr) {
        auto fake = std::make_unique<fakeCApi>();
        log = fake->log;
        if (retained_fake != nullptr) {
            *retained_fake = fake.get();
        }
        return std::make_unique<nixlRadosEngine>(&init_params_, std::move(fake));
    }

    // Same as makeEngine, but injects the fake status-wait timer.
    std::unique_ptr<nixlRadosEngine>
    makeEngineWithTimer(fakeCApi **retained_fake, const std::shared_ptr<timer_state> &state) {
        auto fake = std::make_unique<fakeCApi>();
        log = fake->log;
        if (retained_fake != nullptr) {
            *retained_fake = fake.get();
        }
        auto timer = std::make_unique<fakeStatusWaitTimer>(state);
        return std::make_unique<nixlRadosEngine>(
            &init_params_, std::move(fake), std::move(timer));
    }

    // Representative successful construction: validBase parameters plus the two client bootstraps.
    std::unique_ptr<nixlRadosEngine>
    makeReady(fakeCApi **retained_fake = nullptr) {
        custom_params_ = validBase();
        auto engine = makeEngine(retained_fake);
        EXPECT_FALSE(engine->getInitErr());
        return engine;
    }

    nixlBackendMD *
    registerObject(nixlRadosEngine &engine, uint64_t dev_id, const std::string &key) {
        nixlBackendMD *md = nullptr;
        const nixlBlobDesc desc(0, 0, dev_id, key);
        EXPECT_EQ(engine.registerMem(desc, OBJ_SEG, md), NIXL_SUCCESS);
        return md;
    }

    nixlBackendInitParams init_params_;
    nixl_b_params_t custom_params_;
};

// radosConfigTest owns the invalid-parameter matrix; representatives prove the
// engine rejects them before touching the librados seam.
TEST_F(radosEngineFixture, RejectsInvalidParamsBeforeAnyAPICall) {
    std::vector<nixl_b_params_t> invalid;
    auto unprefixed = validBase();
    unprefixed["rbd_cache"] = "true";
    invalid.push_back(unprefixed);
    auto empty_ns = validBase();
    empty_ns["namespace"] = "";
    invalid.push_back(empty_ns);
    auto nul_ns = validBase();
    nul_ns["namespace"] = std::string("qa\0ns", 5);
    invalid.push_back(nul_ns);

    for (const auto &params : invalid) {
        custom_params_ = params;
        auto engine = makeEngine();
        EXPECT_TRUE(engine->getInitErr());
        EXPECT_TRUE(log->empty()) << "rejection must not touch the librados seam";
    }

    custom_params_ = validBase();
    nixlRadosEngine null_seam(&init_params_, nullptr);
    EXPECT_TRUE(null_seam.getInitErr());
}

TEST_F(radosEngineFixture, BootstrapsClientAndRetiresResources) {
    auto params = validBase();
    params["ceph.zeta"] = "3";
    params["ceph.alpha"] = "";
    custom_params_ = params;

    const std::vector<std::string> client_bootstrap{"create2",
                                                    "conf_read_file",
                                                    "conf_set:alpha",
                                                    "conf_set:zeta",
                                                    "connect",
                                                    "ioctx_create",
                                                    "ioctx_from_raw"};
    const std::vector<std::string> client_teardown{"ioctx_close", "ioctx_destroy", "shutdown"};

    fakeCApi *fake = nullptr;
    {
        auto engine = makeEngine(&fake);
        ASSERT_FALSE(engine->getInitErr());
        EXPECT_EQ(*log, client_bootstrap);
        EXPECT_EQ(fake->create2_names, (std::vector<std::string>{"client.qa"}));
        EXPECT_EQ(fake->conf_paths, (std::vector<std::string>{"fixture.conf"}));
        EXPECT_EQ(fake->pools, (std::vector<std::string>{"qa-pool"}));
        EXPECT_EQ(fake->options,
                  (std::vector<std::pair<std::string, std::string>>{{"alpha", ""}, {"zeta", "3"}}));
        ASSERT_EQ(fake->view_bind_order->size(), 1U);
    }
    std::vector<std::string> final_log = client_bootstrap;
    final_log.insert(final_log.end(), client_teardown.begin(), client_teardown.end());
    EXPECT_EQ(*log, final_log);
}

TEST_F(radosEngineFixture, NamespaceConfiguredOnceBetweenIoctxCreateAndView) {
    custom_params_ = validBase();
    custom_params_["namespace"] = "qa-ns";
    fakeCApi *fake = nullptr;
    {
        auto engine = makeEngine(&fake);
        ASSERT_FALSE(engine->getInitErr());
        EXPECT_EQ(*log,
                  (std::vector<std::string>{"create2",
                                            "conf_read_file",
                                            "connect",
                                            "ioctx_create",
                                            "ioctx_set_namespace",
                                            "ioctx_from_raw"}));
        EXPECT_EQ(fake->object_namespace, "qa-ns");
    }
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("ioctx_set_namespace")), 1)
        << "the one raw ioctx must be configured exactly once";
}

TEST_F(radosEngineFixture, BootstrapFailureUnwindsAndGatesRegistration) {
    custom_params_ = validBase();
    auto failing = std::make_unique<fakeCApi>();
    log = failing->log;
    failing->connect_rc = -EINVAL;
    {
        nixlRadosEngine engine(&init_params_, std::move(failing));
        EXPECT_TRUE(engine.getInitErr());
        nixlBackendMD *md = nullptr;
        EXPECT_EQ(engine.registerMem(nixlBlobDesc(0x1000, 4096, 0, {}), DRAM_SEG, md),
                  NIXL_ERR_BACKEND);
        EXPECT_EQ(md, nullptr);
        EXPECT_EQ(engine.registerMem(nixlBlobDesc(0, 1024, 1, "obj"), OBJ_SEG, md),
                  NIXL_ERR_BACKEND);
        EXPECT_EQ(*log, (std::vector<std::string>{"create2",
                                                  "conf_read_file",
                                                  "connect",
                                                  "shutdown"}));
    }
    EXPECT_EQ(*log,
              (std::vector<std::string>{"create2",
                                        "conf_read_file",
                                        "connect",
                                        "shutdown"}));
}

TEST_F(radosEngineFixture, AdvertisesLocalOnlyDramAndObjSegments) {
    auto engine = makeReady();
    EXPECT_TRUE(engine->supportsLocal());
    EXPECT_FALSE(engine->supportsRemote());
    EXPECT_FALSE(engine->supportsNotif());
    EXPECT_EQ(engine->getSupportedMems(), (nixl_mem_list_t{DRAM_SEG, OBJ_SEG}));
    EXPECT_EQ(engine->connect("peer"), NIXL_SUCCESS);
    EXPECT_EQ(engine->disconnect("peer"), NIXL_SUCCESS);
    nixlBackendMD *sentinel = reinterpret_cast<nixlBackendMD *>(static_cast<uintptr_t>(0xdead));
    nixlBackendMD *loaded = nullptr;
    EXPECT_EQ(engine->loadLocalMD(sentinel, loaded), NIXL_SUCCESS);
    EXPECT_EQ(loaded, sentinel);
    EXPECT_EQ(engine->unloadMD(sentinel), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, RegistrationBoundarySemantics) {
    fakeCApi *fake = nullptr;
    const auto engine = makeReady(&fake);
    const size_t seam_calls_after_bootstrap = log->size();

    nixlBackendMD *dram_md = reinterpret_cast<nixlBackendMD *>(static_cast<uintptr_t>(0xdead));
    const nixlBlobDesc dram_desc(0x1000, 4096, 0, {});
    EXPECT_EQ(engine->registerMem(dram_desc, DRAM_SEG, dram_md), NIXL_SUCCESS);
    EXPECT_EQ(dram_md, nullptr);
    EXPECT_EQ(engine->deregisterMem(nullptr), NIXL_SUCCESS);

    nixlBackendMD *file_md = nullptr;
    const nixlBlobDesc file_desc(0, 0, 0, {});
    EXPECT_EQ(engine->registerMem(file_desc, FILE_SEG, file_md), NIXL_ERR_NOT_SUPPORTED);

    nixlBackendMD *md_a = nullptr;
    const nixlBlobDesc desc_a(0, 1024, 1, "obj-a");
    EXPECT_EQ(engine->registerMem(desc_a, OBJ_SEG, md_a), NIXL_SUCCESS);
    ASSERT_NE(md_a, nullptr);
    EXPECT_EQ(static_cast<nixlRadosObjectMD *>(md_a)->objectKey, "obj-a");

    nixlBackendMD *md_zero = nullptr;
    const nixlBlobDesc desc_zero(0, 1024, 0, {});
    EXPECT_EQ(engine->registerMem(desc_zero, OBJ_SEG, md_zero), NIXL_SUCCESS);
    EXPECT_EQ(static_cast<nixlRadosObjectMD *>(md_zero)->objectKey, "0");

    nixlBackendMD *md_dup = nullptr;
    const nixlBlobDesc desc_dup(0, 1024, 1, "other");
    EXPECT_EQ(engine->registerMem(desc_dup, OBJ_SEG, md_dup), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(md_dup, nullptr);

    nixlBackendMD *md_alias = nullptr;
    const nixlBlobDesc desc_alias(0, 1024, 2, "obj-a");
    EXPECT_EQ(engine->registerMem(desc_alias, OBJ_SEG, md_alias), NIXL_SUCCESS);

    // A reservation is erased only when devId and key still match; a stale
    // mismatched MD must leave the live reservation intact and consume itself.
    EXPECT_EQ(engine->deregisterMem(new nixlRadosObjectMD(1, "stale-key")), NIXL_SUCCESS);
    nixlBackendMD *md_reject = nullptr;
    EXPECT_EQ(engine->registerMem(desc_a, OBJ_SEG, md_reject), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(md_reject, nullptr);

    EXPECT_EQ(engine->deregisterMem(md_a), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_zero), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_alias), NIXL_SUCCESS);
    EXPECT_EQ(engine->registerMem(desc_a, OBJ_SEG, md_reject), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_reject), NIXL_SUCCESS);
    EXPECT_EQ(log->size(), seam_calls_after_bootstrap)
        << "registration and deregistration must not reach the librados seam";
}

TEST_F(radosEngineFixture, RegisterRejectsEmbeddedNulMetaInfoWithoutReservation) {
    fakeCApi *fake = nullptr;
    const auto engine = makeReady(&fake);
    const size_t seam_calls_after_bootstrap = log->size();

    const nixlBlobDesc nul_desc(0, 1024, 1, std::string("bad\0key", 7));
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(engine->registerMem(nul_desc, OBJ_SEG, md), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(md, nullptr);

    nixlBackendMD *retry_md = nullptr;
    const nixlBlobDesc retry_desc(0, 1024, 1, "good-key");
    EXPECT_EQ(engine->registerMem(retry_desc, OBJ_SEG, retry_md), NIXL_SUCCESS);
    ASSERT_NE(retry_md, nullptr);
    EXPECT_EQ(static_cast<nixlRadosObjectMD *>(retry_md)->objectKey, "good-key");

    nixlBackendMD *fallback_md = nullptr;
    const nixlBlobDesc fallback_desc(0, 1024, 0, {});
    EXPECT_EQ(engine->registerMem(fallback_desc, OBJ_SEG, fallback_md), NIXL_SUCCESS);
    ASSERT_NE(fallback_md, nullptr);
    EXPECT_EQ(static_cast<nixlRadosObjectMD *>(fallback_md)->objectKey, "0");

    nixlBackendMD *trailing_md = nullptr;
    const nixlBlobDesc trailing_desc(0, 1024, 3, std::string("key\0", 4));
    EXPECT_EQ(engine->registerMem(trailing_desc, OBJ_SEG, trailing_md), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(trailing_md, nullptr);

    EXPECT_EQ(engine->deregisterMem(retry_md), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(fallback_md), NIXL_SUCCESS);
    EXPECT_EQ(log->size(), seam_calls_after_bootstrap)
        << "registration and its NUL rejection must not reach the librados seam";
}

TEST_F(radosEngineFixture, PrepValidationGateTable) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    [[maybe_unused]] nixlBackendMD *md_1 = registerObject(*engine, 1, "obj");
    [[maybe_unused]] nixlBackendMD *md_2 = registerObject(*engine, 2, "obj");
    [[maybe_unused]] nixlBackendMD *md_3 = registerObject(*engine, 3, "obj-3");

    const uintptr_t base = 0x10000;
    const uintptr_t top = std::numeric_limits<uintptr_t>::max();
    const uint64_t otop = std::numeric_limits<uint64_t>::max();
    using pair_spec = std::tuple<uintptr_t, uint64_t, uint64_t, uint64_t>;  // addr,len,off,dev
    struct prep_case {
        const char *name;
        nixl_xfer_op_t op;
        nixl_mem_t local_type;
        nixl_mem_t remote_type;
        const char *agent;
        std::vector<pair_spec> pairs;
        nixl_status_t expected;
    };
    const std::vector<prep_case> cases = {
        {"unsupported op",
         static_cast<nixl_xfer_op_t>(99), DRAM_SEG, OBJ_SEG, "test-agent", {{base, 64, 0, 1}},
         NIXL_ERR_NOT_SUPPORTED},
        {"local segment",
         NIXL_WRITE, FILE_SEG, OBJ_SEG, "test-agent", {{base, 64, 0, 1}}, NIXL_ERR_NOT_SUPPORTED},
        {"remote segment",
         NIXL_WRITE, DRAM_SEG, DRAM_SEG, "test-agent", {{base, 64, 0, 1}}, NIXL_ERR_NOT_SUPPORTED},
        {"foreign agent",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG, "other-agent", {{base, 64, 0, 1}}, NIXL_ERR_NOT_SUPPORTED},
        {"empty list", NIXL_WRITE, DRAM_SEG, OBJ_SEG, "test-agent", {}, NIXL_ERR_INVALID_PARAM},
        {"zero length",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG, "test-agent", {{base, 0, 0, 1}}, NIXL_ERR_INVALID_PARAM},
        {"length above INT_MAX",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG, "test-agent", {{base, uint64_t(INT_MAX) + 1, 0, 1}},
         NIXL_ERR_INVALID_PARAM},
        {"null dram address",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG, "test-agent", {{0, 64, 0, 1}}, NIXL_ERR_INVALID_PARAM},
        {"local range wrap",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG,
         "test-agent", {{top - 8, 64, 0, 1}}, NIXL_ERR_INVALID_PARAM},
        {"object offset wrap",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG, "test-agent", {{base, 64, otop - 8, 1}},
         NIXL_ERR_INVALID_PARAM},
        {"unregistered devId",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG, "test-agent", {{base, 64, 0, 42}}, NIXL_ERR_INVALID_PARAM},
        {"overlapping writes on one devId",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG,
         "test-agent", {{base, 64, 0, 1}, {base + 64, 64, 32, 1}}, NIXL_ERR_INVALID_PARAM},
        {"overlapping writes across aliased devIds",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG,
         "test-agent", {{base, 64, 0, 1}, {base + 64, 64, 32, 2}}, NIXL_ERR_INVALID_PARAM},
        {"overlapping reads are legal",
         NIXL_READ, DRAM_SEG, OBJ_SEG,
         "test-agent", {{base, 64, 0, 1}, {base + 64, 64, 32, 1}}, NIXL_SUCCESS},
        {"adjacent writes across aliased devIds",
         NIXL_WRITE, DRAM_SEG, OBJ_SEG,
         "test-agent", {{base, 64, 0, 1}, {base + 64, 64, 64, 2}}, NIXL_SUCCESS},
    };
    for (const auto &item : cases) {
        nixl_meta_dlist_t local(item.local_type);
        nixl_meta_dlist_t remote(item.remote_type);
        for (const auto &[addr, len, off, dev] : item.pairs) {
            add_pair(local, remote, addr, len, off, dev);
        }
        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(engine->prepXfer(item.op, local, remote, item.agent, handle), item.expected)
            << item.name;
        if (item.expected == NIXL_SUCCESS) {
            ASSERT_NE(handle, nullptr) << item.name;
            EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS) << item.name;
        } else {
            EXPECT_EQ(handle, nullptr) << item.name;
        }
    }

    {
        nixl_meta_dlist_t local(DRAM_SEG);
        nixl_meta_dlist_t remote(OBJ_SEG);
        add_pair(local, remote, base, 64, 0, 1);
        remote.addDesc(nixlMetaDesc(64, 64, 2));  // descriptor-count mismatch
        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, "test-agent", handle),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(handle, nullptr);
    }
    {
        const one_pair single(0, 64, 1, 0x00);
        nixl_meta_dlist_t short_remote(OBJ_SEG);
        short_remote.addDesc(nixlMetaDesc(0, 63, 1));  // unequal pair lengths
        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(engine->prepXfer(NIXL_WRITE, single.local, short_remote, "test-agent", handle),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(handle, nullptr);
    }
    EXPECT_TRUE(fake->submissions.empty());
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("c_read")), 0);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("cpp_write")), 0);

    const nixl_meta_dlist_t empty_local(DRAM_SEG);
    const nixl_meta_dlist_t empty_remote(OBJ_SEG);
    nixlBackendReqH *foreign = reinterpret_cast<nixlBackendReqH *>(static_cast<uintptr_t>(0x1));
    EXPECT_EQ(engine->postXfer(NIXL_WRITE, empty_local, empty_remote, "test-agent", foreign),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->checkXfer(nullptr), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->checkXfer(foreign), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->releaseReqH(nullptr), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->releaseReqH(foreign), NIXL_ERR_INVALID_PARAM);
}

// prepXfer snapshots key/pointer/offset/length/direction; postXfer is driven solely
// by that state, ignoring its operation, descriptor, and agent arguments.
TEST_F(radosEngineFixture, PreparedRequestKeepsSnapshottedKeyAfterDeregistration) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md = registerObject(*engine, 1, "obj-a");
    ASSERT_NE(md, nullptr);

    const one_pair pair(0, 64, 1, 0x00);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
              NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);

    nixlBackendReqH *refetch = nullptr;
    EXPECT_EQ(engine->prepXfer(NIXL_READ, pair.local, pair.remote, "test-agent", refetch),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(refetch, nullptr);

    nixl_meta_dlist_t junk_local(FILE_SEG);
    nixl_meta_dlist_t junk_remote(FILE_SEG);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, junk_local, junk_remote, "other-agent", handle),
              NIXL_IN_PROG);
    ASSERT_EQ(fake->submissions.size(), 1U);
    EXPECT_FALSE(fake->submissions[0].is_write);
    EXPECT_EQ(fake->submissions[0].oid, "obj-a");
    EXPECT_EQ(fake->submissions[0].offset, 0U);
    EXPECT_EQ(fake->submissions[0].buf, reinterpret_cast<uintptr_t>(pair.buffer.data()));
    fake->completeSubmission(0, static_cast<int>(pair.buffer.size()));
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
}

// C reads describe caller destination memory directly; results map per rc.
TEST_F(radosEngineFixture, ReadResultMappingAndDirectCallerDestination) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md = registerObject(*engine, 1, "obj");

    constexpr uint64_t kLen = 128;
    const one_pair pair(0x1000, kLen, 1, 0x2a);
    const std::vector<std::tuple<int, nixl_status_t>> cases = {
        {static_cast<int>(kLen), NIXL_SUCCESS},
        {static_cast<int>(kLen) - 1, NIXL_ERR_MISMATCH},
        {0, NIXL_ERR_MISMATCH},
        {static_cast<int>(kLen) + 1, NIXL_ERR_BACKEND},
        {-ENOENT, NIXL_ERR_NOT_FOUND},
        {-EIO, NIXL_ERR_BACKEND},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto &[rc, expected] = cases[i];
        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine->prepXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine->postXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
                  NIXL_IN_PROG);
        const submission &sub = fake->submissions[i];
        if (i == 0) {
            EXPECT_FALSE(sub.is_write);
            EXPECT_EQ(sub.oid, "obj");
            EXPECT_EQ(sub.offset, 0x1000U);
            EXPECT_EQ(sub.len, kLen);
            EXPECT_EQ(sub.buf, reinterpret_cast<uintptr_t>(pair.buffer.data()))
                << "reads must land directly in caller destination memory";
            EXPECT_EQ(std::count(log->begin(), log->end(), std::string("cpp_write")), 0);
        }
        fake->completeSubmission(i, rc);
        EXPECT_EQ(engine->checkXfer(handle), expected) << "rc=" << rc;
        EXPECT_EQ(engine->checkXfer(handle), expected);
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    }
    EXPECT_EQ(fake->c_waits, 0) << "poll-driven harvest must never wait";
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

// Writes borrow caller memory through a non-owning static view (never copied by
// the plugin); results map per rc, with only exact zero being success.
TEST_F(radosEngineFixture, WriteResultMappingAndNonOwningPayload) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md = registerObject(*engine, 1, "obj");

    const one_pair pair(0x2000, 64, 1, 0x11);
    const std::vector<std::tuple<int, nixl_status_t>> cases = {
        {0, NIXL_SUCCESS},
        {1, NIXL_ERR_BACKEND},
        {-EIO, NIXL_ERR_BACKEND},
        {-ENOENT, NIXL_ERR_BACKEND},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto &[rc, expected] = cases[i];
        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine->prepXfer(NIXL_WRITE, pair.local, pair.remote, "test-agent", handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine->postXfer(NIXL_WRITE, pair.local, pair.remote, "test-agent", handle),
                  NIXL_IN_PROG);
        const submission &sub = fake->submissions[i];
        if (i == 0) {
            EXPECT_TRUE(sub.is_write);
            EXPECT_EQ(sub.oid, "obj");
            EXPECT_EQ(sub.offset, 0x2000U);
            EXPECT_EQ(sub.len, 64U);
            EXPECT_EQ(sub.payload_alias, reinterpret_cast<uintptr_t>(pair.buffer.data()))
                << "the payload must alias caller DRAM, not a plugin copy";
            EXPECT_EQ(std::count(log->begin(), log->end(), std::string("c_read")), 0);
            EXPECT_EQ(std::count(log->begin(), log->end(), std::string("c_create_completion")), 0);
        }
        fake->completeSubmission(i, rc);
        EXPECT_EQ(engine->checkXfer(handle), expected) << "rc=" << rc;
        if (i == 0) {
            EXPECT_EQ(sub.payload_at_release, std::string(64, 0x11));
        }
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    }
    EXPECT_EQ(fake->cpp_waits, 0) << "poll-driven harvest must never wait";
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

// Three ordered pairs against one object; rounds: zero-accepted terminal,
// partial rejection stops later submissions, completion-factory rejection stops.
TEST_F(radosEngineFixture, WriteSubmissionStopsAtFirstRejectedPairAndOwnsCompletions) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md = registerObject(*engine, 1, "obj");
    const one_pair pair(0, 192, 1, 0x33);
    const uintptr_t base = reinterpret_cast<uintptr_t>(pair.buffer.data());
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    for (int i = 0; i < 3; ++i) {
        add_pair(
            local, remote, base + static_cast<uintptr_t>(i) * 64, 64, static_cast<uint64_t>(i) * 64,
            1);
    }
    nixlBackendReqH *handle = nullptr;

    fake->cpp_submit_fail_after = 0;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->postXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_TRUE(fake->submissions.empty());
    EXPECT_EQ(fake->cpp_releases, 1);
    EXPECT_EQ(fake->cpp_waits, 0);
    EXPECT_EQ(fake->cpp_rc_reads, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);

    fake->cpp_submit_fail_after = 1;
    const int writes_before_second_round =
        std::count(log->begin(), log->end(), std::string("cpp_write"));
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->postXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_IN_PROG);
    ASSERT_EQ(fake->submissions.size(), 1U);
    EXPECT_EQ(fake->submissions[0].offset, 0U) << "pairs submit in descriptor order";
    EXPECT_EQ(fake->cpp_releases, 2) << "the rejected pair's completion is released without wait";
    EXPECT_EQ(fake->cpp_waits, 0);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("cpp_write")),
              writes_before_second_round + 2)
        << "the third pair is never attempted";
    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    fake->completeSubmission(0);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);

    fake->cpp_submit_fail_after = -1;
    const int creates_before =
        std::count(log->begin(), log->end(), std::string("cpp_create_completion"));
    const int releases_before = fake->cpp_releases;
    fake->cpp_create_completion_fail_after = creates_before + 1;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->postXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_IN_PROG);
    ASSERT_EQ(fake->submissions.size(), 2U);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("cpp_create_completion")),
              creates_before + 2)
        << "the factory-failed pair reserves no completion";
    EXPECT_EQ(fake->cpp_releases, releases_before)
        << "a factory-failed pair leaves nothing to release";
    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    fake->completeSubmission(1);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, ReadSubmissionStopsAtFirstRejectedPairAndOwnsCompletions) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md = registerObject(*engine, 1, "obj");
    const one_pair pair(0, 192, 1, 0x00);
    const uintptr_t base = reinterpret_cast<uintptr_t>(pair.buffer.data());
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    for (int i = 0; i < 3; ++i) {
        add_pair(
            local, remote, base + static_cast<uintptr_t>(i) * 64, 64, static_cast<uint64_t>(i) * 64,
            1);
    }
    nixlBackendReqH *handle = nullptr;

    fake->c_submit_fail_after = 0;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->postXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_TRUE(fake->submissions.empty());
    EXPECT_EQ(fake->c_releases, 1);
    EXPECT_EQ(fake->c_waits, 0);
    EXPECT_EQ(fake->c_rc_reads, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);

    fake->c_submit_fail_after = 1;
    const int reads_before_second_round =
        std::count(log->begin(), log->end(), std::string("c_read"));
    ASSERT_EQ(engine->prepXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->postXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_IN_PROG);
    ASSERT_EQ(fake->submissions.size(), 1U);
    EXPECT_EQ(fake->submissions[0].offset, 0U) << "pairs submit in descriptor order";
    EXPECT_EQ(fake->c_releases, 2) << "the rejected pair's completion is released without wait";
    EXPECT_EQ(fake->c_waits, 0);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("c_read")),
              reads_before_second_round + 2)
        << "the third pair is never attempted";
    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    fake->completeSubmission(0);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);

    fake->c_submit_fail_after = -1;
    const int creates_before =
        std::count(log->begin(), log->end(), std::string("c_create_completion"));
    const int releases_before = fake->c_releases;
    fake->c_create_completion_fail_after = creates_before + 1;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->postXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_IN_PROG);
    ASSERT_EQ(fake->submissions.size(), 2U);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("c_create_completion")),
              creates_before + 2)
        << "the factory-failed pair reserves no completion";
    EXPECT_EQ(fake->c_releases, releases_before)
        << "a factory-failed pair leaves nothing to release";
    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    fake->completeSubmission(1);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, CppWriteThrowAfterAcceptedPrefixReleasesWithoutWait) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md = registerObject(*engine, 1, "obj");
    const one_pair pair(0, 192, 1, 0x3a);
    const uintptr_t base = reinterpret_cast<uintptr_t>(pair.buffer.data());
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    for (int i = 0; i < 3; ++i) {
        add_pair(
            local, remote, base + static_cast<uintptr_t>(i) * 64, 64, static_cast<uint64_t>(i) * 64,
            1);
    }

    fake->cpp_write_throw_after = 1;
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_SUCCESS);
    EXPECT_NO_THROW(
        EXPECT_EQ(engine->postXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_IN_PROG));
    EXPECT_EQ(fake->submissions.size(), 1U);
    EXPECT_EQ(fake->cpp_releases, 1);
    EXPECT_EQ(fake->cpp_waits, 0);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("cpp_write")), 2);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("cpp_create_completion")), 2);

    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    fake->completeSubmission(0);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(fake->cpp_waits, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_TRUE(fake->violations.empty());
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, OutOfOrderRetirementKeepsRequestInProgressUntilEarliestPairResolves) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md_a = registerObject(*engine, 1, "obj-a");
    nixlBackendMD *md_b = registerObject(*engine, 2, "obj-b");

    const one_pair pair(0, 128, 1, 0x00);
    const uintptr_t base = reinterpret_cast<uintptr_t>(pair.buffer.data());
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    add_pair(local, remote, base, 64, 0, 1);
    add_pair(local, remote, base + 64, 64, 64, 2);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_IN_PROG);
    ASSERT_EQ(fake->submissions.size(), 2U);

    fake->completeSubmission(1, -EIO);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);

    fake->completeSubmission(0, -ENOENT);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_NOT_FOUND)
        << "terminal status is the first non-success in descriptor order";
    EXPECT_EQ(fake->c_rc_reads, 2);
    EXPECT_EQ(fake->c_waits, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("ioctx_destroy")), 0);
    EXPECT_EQ(engine->deregisterMem(md_a), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_b), NIXL_SUCCESS);
}

// A complete-but-callback-unretired completion is unproven: reads never harvest it,
// a release drain waits for retirement before reading and releasing.
TEST_F(radosEngineFixture, CompleteButUnretiredGatesHarvestAndReleaseInBothFamilies) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md_r = registerObject(*engine, 1, "obj-r");
    nixlBackendMD *md_w = registerObject(*engine, 2, "obj-w");

    const one_pair read(0, 64, 1, 0x00);
    nixlBackendReqH *read_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_IN_PROG);
    fake->completeButUnretired(0);
    EXPECT_EQ(engine->checkXfer(read_handle), NIXL_IN_PROG);
    EXPECT_EQ(fake->c_rc_reads, 0);
    EXPECT_EQ(fake->c_releases, 0);
    EXPECT_EQ(engine->releaseReqH(read_handle), NIXL_SUCCESS);
    EXPECT_EQ(fake->c_waits, 1);
    EXPECT_EQ(fake->c_rc_reads, 1);
    EXPECT_TRUE(fake->submissions[0].c_comp->waited);
    EXPECT_FALSE(fake->submissions[0].c_comp->released_unretired);

    const one_pair write(0, 64, 2, 0x11);
    nixlBackendReqH *write_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
              NIXL_IN_PROG);
    fake->completeButUnretired(1);
    EXPECT_EQ(engine->checkXfer(write_handle), NIXL_IN_PROG);
    EXPECT_EQ(fake->cpp_rc_reads, 0);
    fake->retire(1);
    EXPECT_EQ(engine->checkXfer(write_handle), NIXL_SUCCESS);
    EXPECT_EQ(fake->cpp_waits, 0);
    EXPECT_EQ(engine->releaseReqH(write_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_r), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_w), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, EarlyReleaseBlocksOnWithheldWaitThenDrainsBothFamilies) {
    custom_params_ = validBase();
    std::vector<unsigned char> src(64, 0x44);
    std::vector<unsigned char> dest(64, 0x00);
    fakeCApi *fake = nullptr;
    auto engine = makeEngine(&fake);
    ASSERT_FALSE(engine->getInitErr());
    nixlBackendMD *md_w = registerObject(*engine, 1, "obj-w");
    nixlBackendMD *md_r = registerObject(*engine, 2, "obj-r");

    nixl_meta_dlist_t write_local(DRAM_SEG);
    nixl_meta_dlist_t write_remote(OBJ_SEG);
    write_local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(src.data()), src.size(), 0));
    write_remote.addDesc(nixlMetaDesc(0, src.size(), 1));
    nixl_meta_dlist_t read_local(DRAM_SEG);
    nixl_meta_dlist_t read_remote(OBJ_SEG);
    read_local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(dest.data()), dest.size(), 0));
    read_remote.addDesc(nixlMetaDesc(0, dest.size(), 2));

    nixlBackendReqH *write_handle = nullptr;
    nixlBackendReqH *read_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, write_local, write_remote, "test-agent", write_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, write_local, write_remote, "test-agent", write_handle),
              NIXL_IN_PROG);
    ASSERT_EQ(engine->prepXfer(NIXL_READ, read_local, read_remote, "test-agent", read_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, read_local, read_remote, "test-agent", read_handle),
              NIXL_IN_PROG);
    ASSERT_EQ(fake->submissions.size(), 2U);

    fake->armWaitGate();
    const size_t log_before = log->size();
    std::promise<nixl_status_t> write_release;
    std::future<nixl_status_t> write_status = write_release.get_future();
    std::thread releaser(
        [&write_release, &engine, write_handle]() {
            write_release.set_value(engine->releaseReqH(write_handle));
        });

    const bool gate_entered = fake->waitGateEntered();
    bool withheld = false;
    int cpp_waits_at_gate = -1;
    int cpp_releases_at_gate = -1;
    int c_releases_at_gate = -1;
    if (gate_entered) {
        withheld =
            write_status.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
        cpp_waits_at_gate = fake->cpp_waits;
        cpp_releases_at_gate = fake->cpp_releases;
        c_releases_at_gate = fake->c_releases;
    }

    fake->releaseWaitGate();
    releaser.join();

    ASSERT_TRUE(gate_entered) << "the drain never reached the completion wait";
    EXPECT_TRUE(withheld) << "releaseReqH returned while its completion wait was withheld";
    EXPECT_EQ(cpp_waits_at_gate, 1);
    EXPECT_EQ(cpp_releases_at_gate, 0);
    EXPECT_EQ(c_releases_at_gate, 0);
    EXPECT_EQ(write_status.get(), NIXL_SUCCESS);

    const std::vector<std::string> write_tail(
        log->begin() + static_cast<std::ptrdiff_t>(log_before), log->end());
    const auto w_wait = std::find(
        write_tail.begin(), write_tail.end(), std::string("cpp_wait_for_complete_and_cb"));
    const auto w_release =
        std::find(write_tail.begin(), write_tail.end(), std::string("cpp_release"));
    ASSERT_NE(w_wait, write_tail.end());
    ASSERT_NE(w_release, write_tail.end());
    EXPECT_LT(w_wait, w_release);
    EXPECT_TRUE(fake->submissions[0].cpp_comp->waited);
    EXPECT_FALSE(fake->submissions[0].cpp_comp->released_unretired);
    EXPECT_EQ(fake->c_releases, 0) << "the read request must stay untouched";

    EXPECT_EQ(engine->releaseReqH(read_handle), NIXL_SUCCESS);
    EXPECT_TRUE(fake->submissions[1].c_comp->waited);
    EXPECT_FALSE(fake->submissions[1].c_comp->released_unretired);
    EXPECT_EQ(fake->c_releases, 1);

    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("ioctx_destroy")), 0);
    EXPECT_EQ(engine->deregisterMem(md_w), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_r), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, TeardownDrainsBothFamiliesBeforeResourceDeath) {
    custom_params_ = validBase();
    std::vector<unsigned char> src(64, 0x77);
    std::vector<unsigned char> dest(64, 0x00);
    {
        fakeCApi *fake = nullptr;
        auto engine = makeEngine(&fake);
        ASSERT_FALSE(engine->getInitErr());
        [[maybe_unused]] nixlBackendMD *md_w = registerObject(*engine, 1, "obj-w");
        [[maybe_unused]] nixlBackendMD *md_r = registerObject(*engine, 2, "obj-r");

        nixl_meta_dlist_t write_local(DRAM_SEG);
        nixl_meta_dlist_t write_remote(OBJ_SEG);
        write_local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(src.data()), src.size(), 0));
        write_remote.addDesc(nixlMetaDesc(0, src.size(), 1));
        nixl_meta_dlist_t read_local(DRAM_SEG);
        nixl_meta_dlist_t read_remote(OBJ_SEG);
        read_local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(dest.data()), dest.size(), 0));
        read_remote.addDesc(nixlMetaDesc(0, dest.size(), 2));

        nixlBackendReqH *write_handle = nullptr;
        nixlBackendReqH *read_handle = nullptr;
        ASSERT_EQ(
            engine->prepXfer(NIXL_WRITE, write_local, write_remote, "test-agent", write_handle),
            NIXL_SUCCESS);
        ASSERT_EQ(
            engine->postXfer(NIXL_WRITE, write_local, write_remote, "test-agent", write_handle),
            NIXL_IN_PROG);
        ASSERT_EQ(engine->prepXfer(NIXL_READ, read_local, read_remote, "test-agent", read_handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine->postXfer(NIXL_READ, read_local, read_remote, "test-agent", read_handle),
                  NIXL_IN_PROG);

        fake->completeSubmission(0, -EIO);
        ASSERT_EQ(engine->checkXfer(write_handle), NIXL_ERR_BACKEND);
    }
    const auto &final_log = *log;
    const auto last_release =
        std::find_if(final_log.rbegin(), final_log.rend(), [](const std::string &entry) {
            return entry == "c_release" || entry == "cpp_release";
        });
    const auto ioctx_close =
        std::find(final_log.begin(), final_log.end(), std::string("ioctx_close"));
    const auto ioctx_destroy =
        std::find(final_log.begin(), final_log.end(), std::string("ioctx_destroy"));
    const auto shutdown = std::find(final_log.begin(), final_log.end(), std::string("shutdown"));
    ASSERT_NE(last_release, final_log.rend());
    ASSERT_NE(ioctx_close, final_log.end());
    ASSERT_NE(ioctx_destroy, final_log.end());
    ASSERT_NE(shutdown, final_log.end());
    EXPECT_TRUE(last_release.base() <= ioctx_close);
    EXPECT_LT(ioctx_close, ioctx_destroy);
    EXPECT_LT(ioctx_destroy, shutdown);
    EXPECT_EQ(
        std::count(final_log.begin(), final_log.end(), std::string("cpp_wait_for_complete_and_cb")),
        0);
    EXPECT_EQ(
        std::count(final_log.begin(), final_log.end(), std::string("c_wait_for_complete_and_cb")),
        1);
    EXPECT_EQ(std::count(final_log.begin(), final_log.end(), std::string("cpp_release")), 1);
    EXPECT_EQ(std::count(final_log.begin(), final_log.end(), std::string("c_release")), 1);
}

// Repost is legal only from terminal: a fresh completion and buffer state is
// rebuilt from the snapshotted pointer, re-staging the caller's current bytes.
TEST_F(radosEngineFixture, RepostAfterTerminalResubmitsResetStateWithCurrentSourceBytes) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md_w = registerObject(*engine, 1, "obj-w");
    nixlBackendMD *md_r = registerObject(*engine, 2, "obj-r");

    one_pair write(0, 64, 1, 0x99);
    nixlBackendReqH *write_handle = nullptr;
    ASSERT_EQ(
        engine->prepXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
        NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
              NIXL_IN_PROG);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
              NIXL_IN_PROG);
    EXPECT_EQ(fake->submissions.size(), 1U) << "repost during ACTIVE is a no-op";
    fake->completeSubmission(0);
    ASSERT_EQ(engine->checkXfer(write_handle), NIXL_SUCCESS);

    std::fill(write.buffer.begin(), write.buffer.end(), 0x6b);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
              NIXL_IN_PROG);
    ASSERT_EQ(fake->submissions.size(), 2U);
    EXPECT_NE(fake->submissions[1].cpp_comp, fake->submissions[0].cpp_comp);
    fake->completeSubmission(1);
    EXPECT_EQ(engine->checkXfer(write_handle), NIXL_SUCCESS);
    EXPECT_EQ(fake->submissions[0].payload_at_release, std::string(64, 0x99));
    EXPECT_EQ(fake->submissions[1].payload_at_release, std::string(64, 0x6b))
        << "the reposted round restages the current caller bytes";
    EXPECT_EQ(engine->releaseReqH(write_handle), NIXL_SUCCESS);

    const one_pair read(0, 64, 2, 0x00);
    nixlBackendReqH *read_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_IN_PROG);
    fake->completeSubmission(2, -EIO);
    ASSERT_EQ(engine->checkXfer(read_handle), NIXL_ERR_BACKEND);
    ASSERT_EQ(engine->postXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_IN_PROG);
    ASSERT_EQ(fake->submissions.size(), 4U);
    EXPECT_NE(fake->submissions[3].c_comp, fake->submissions[2].c_comp);
    fake->completeSubmission(3, 64);
    EXPECT_EQ(engine->checkXfer(read_handle), NIXL_SUCCESS)
        << "the repost must reset the stale failed result cell";
    EXPECT_EQ(engine->releaseReqH(read_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_w), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_r), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, QueryMemUnsupportedUntilBatchedQueries) {
    fakeCApi *fake = nullptr;
    const auto engine = makeReady(&fake);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 0, 1, "obj"));
    std::vector<nixl_query_resp_t> resp(1, nixl_query_resp_t{nixl_b_params_t{}});
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_ERR_NOT_SUPPORTED);
    ASSERT_EQ(resp.size(), 1u);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_EQ(fake->submissions.size(), 0U);
}

// C reads submit on the raw ioctx and C++ writes on the bound IoCtx view, so
// the static payload and the direct destination ride their required handles.
TEST_F(radosEngineFixture, ReadsUseRawIoctxWritesUseBoundView) {
    fakeCApi *fake = nullptr;
    auto engine = makeReady(&fake);
    ASSERT_EQ(fake->view_bind_order->size(), 1U);
    const librados::IoCtx *view = (*fake->view_bind_order)[0];

    nixlBackendMD *md_w = registerObject(*engine, 1, "iso-w");
    nixlBackendMD *md_r = registerObject(*engine, 2, "iso-r");
    const one_pair write(0, 64, 1, 0x77);
    nixlBackendReqH *write_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
              NIXL_IN_PROG);
    ASSERT_EQ(fake->write_ctxs.size(), 1U);
    EXPECT_EQ(fake->write_ctxs[0], view) << "writes must submit on the bound view";
    EXPECT_TRUE(fake->read_ioctx_ordinals.empty());
    fake->completeSubmission(0);
    EXPECT_EQ(engine->checkXfer(write_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->releaseReqH(write_handle), NIXL_SUCCESS);

    const one_pair read(0, 64, 2, 0x00);
    nixlBackendReqH *read_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_IN_PROG);
    ASSERT_EQ(fake->read_ioctx_ordinals.size(), 1U);
    EXPECT_EQ(fake->read_ioctx_ordinals[0], 0U) << "reads must use the raw ioctx";
    EXPECT_EQ(fake->write_ctxs[0], view) << "the write view must stay the bound view";
    fake->completeSubmission(1, 64);
    EXPECT_EQ(engine->checkXfer(read_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->releaseReqH(read_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_w), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_r), NIXL_SUCCESS);
}

// Representative rejections; the full table lives in radosConfigTest.
TEST_F(radosEngineFixture, InvalidStatusWaitUsFailsInitWithoutCephAccess) {
    for (const char *bad : {"", "-1", "abc", "1000001"}) {
        custom_params_ = validBase();
        custom_params_["status_wait_us"] = bad;
        auto engine = makeEngine();
        EXPECT_TRUE(engine->getInitErr()) << "value '" << bad << "'";
        EXPECT_TRUE(log->empty()) << "rejection must not touch the librados seam";
    }
}

TEST_F(radosEngineFixture, ZeroStatusWaitUsRetainsNoTimerAndNeverSleeps) {
    std::vector<nixl_b_params_t> variants;
    variants.push_back(validBase()); // absent
    auto zero = validBase();
    zero["status_wait_us"] = "0";
    variants.push_back(zero);
    auto padded = validBase();
    padded["status_wait_us"] = "00000"; // leading-zero zero
    variants.push_back(padded);

    for (const auto &variant : variants) {
        custom_params_ = variant;
        auto tstate = std::make_shared<timer_state>();
        fakeCApi *fake = nullptr;
        auto engine = makeEngineWithTimer(&fake, tstate);
        ASSERT_FALSE(engine->getInitErr());
        EXPECT_TRUE(tstate->destroyed) << "zero mode must not retain the injected timer";

        nixlBackendMD *md = registerObject(*engine, 1, "obj");
        const one_pair pair(0, 64, 1, 0x00);

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine->prepXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine->postXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
                  NIXL_IN_PROG);
        EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
        EXPECT_TRUE(tstate->budgets.empty());
        fake->completeSubmission(0, static_cast<int>(pair.buffer.size()));
        EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
        EXPECT_TRUE(tstate->budgets.empty());
        EXPECT_EQ(fake->c_waits, 0);
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
        EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
    }
}

TEST_F(radosEngineFixture, FirstUnprovenPairSleepsOncePerCallAndDefersHarvest) {
    custom_params_ = validBase();
    custom_params_["status_wait_us"] = "777";
    auto tstate = std::make_shared<timer_state>();
    fakeCApi *fake = nullptr;
    auto engine = makeEngineWithTimer(&fake, tstate);
    ASSERT_FALSE(engine->getInitErr());
    EXPECT_FALSE(tstate->destroyed) << "a positive budget must adopt the injected timer";
    nixlBackendMD *md = registerObject(*engine, 1, "obj");

    const one_pair pair(0, 128, 1, 0x00);
    const uintptr_t base = reinterpret_cast<uintptr_t>(pair.buffer.data());
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    for (int i = 0; i < 2; ++i) {
        add_pair(local,
                 remote,
                 base + static_cast<uintptr_t>(i) * 64,
                 64,
                 static_cast<uint64_t>(i) * 64,
                 1);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, local, remote, "test-agent", handle), NIXL_IN_PROG);

    const auto poll_count = [this]() {
        return std::count(log->begin(), log->end(), std::string("c_is_complete_and_cb"));
    };

    // Pair 0 is operationally done but callback-unretired (unproven through the
    // combined seam) and pair 1 is retired: one sleep per call, no rescan, so
    // the sleeping poll never reaches pair 1 nor harvests anything.
    fake->submissions[0].c_comp->rc = 64;
    fake->completeButUnretired(0);
    fake->completeSubmission(1, 64);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    EXPECT_EQ(tstate->budgets, (std::vector<uint64_t>{777}));
    EXPECT_EQ(poll_count(), 1);
    EXPECT_EQ(fake->c_rc_reads, 0);
    EXPECT_EQ(fake->c_releases, 0);

    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    EXPECT_EQ(tstate->budgets, (std::vector<uint64_t>{777, 777}));

    // The next poll after retirement harvests both in order, sleep-free.
    fake->retire(0);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(tstate->budgets.size(), 2U);
    EXPECT_EQ(fake->c_rc_reads, 2);
    EXPECT_EQ(fake->c_releases, 2);
    EXPECT_EQ(fake->c_waits, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, WriteCompletionSleepsOnceAndHarvestsOnNextPoll) {
    custom_params_ = validBase();
    custom_params_["status_wait_us"] = "2400";
    auto tstate = std::make_shared<timer_state>();
    fakeCApi *fake = nullptr;
    auto engine = makeEngineWithTimer(&fake, tstate);
    ASSERT_FALSE(engine->getInitErr());
    nixlBackendMD *md = registerObject(*engine, 1, "obj");

    const one_pair pair(0, 64, 1, 0x63);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, pair.local, pair.remote, "test-agent", handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, pair.local, pair.remote, "test-agent", handle),
              NIXL_IN_PROG);

    const auto poll_count = [this]() {
        return std::count(log->begin(), log->end(), std::string("cpp_is_complete_and_cb"));
    };

    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    EXPECT_EQ(tstate->budgets, (std::vector<uint64_t>{2400}));
    EXPECT_EQ(poll_count(), 1);
    EXPECT_EQ(fake->cpp_rc_reads, 0);
    EXPECT_EQ(fake->cpp_releases, 0);

    fake->completeSubmission(0);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(tstate->budgets.size(), 1U);
    EXPECT_EQ(poll_count(), 2);
    EXPECT_EQ(fake->cpp_rc_reads, 1);
    EXPECT_EQ(fake->cpp_releases, 1);
    EXPECT_EQ(fake->cpp_waits, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, WaitNeverSleepsOutsideActivePairScans) {
    custom_params_ = validBase();
    custom_params_["status_wait_us"] = "3000";
    auto tstate = std::make_shared<timer_state>();
    fakeCApi *fake = nullptr;
    auto engine = makeEngineWithTimer(&fake, tstate);
    ASSERT_FALSE(engine->getInitErr());
    [[maybe_unused]] nixlBackendMD *md = registerObject(*engine, 1, "obj");

    const one_pair pair(0, 64, 1, 0x00);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
              NIXL_SUCCESS);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_NOT_POSTED);
    EXPECT_EQ(engine->checkXfer(nullptr), NIXL_ERR_INVALID_PARAM);
    nixlBackendReqH *foreign = reinterpret_cast<nixlBackendReqH *>(static_cast<uintptr_t>(0x1));
    EXPECT_EQ(engine->checkXfer(foreign), NIXL_ERR_INVALID_PARAM);
    EXPECT_TRUE(tstate->budgets.empty());

    // Release and teardown drain through the seam's blocking waits, never the
    // timer; the fake seam dies with the engine, so post-reset assertions run
    // on the shared log.
    ASSERT_EQ(engine->postXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
              NIXL_IN_PROG);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_TRUE(tstate->budgets.empty());

    nixlBackendReqH *abandoned = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, pair.local, pair.remote, "test-agent", abandoned),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, pair.local, pair.remote, "test-agent", abandoned),
              NIXL_IN_PROG);
    engine.reset();
    EXPECT_TRUE(tstate->budgets.empty());
    EXPECT_TRUE(tstate->destroyed);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("c_wait_for_complete_and_cb")), 2);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("c_release")), 2);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("ioctx_close")), 1);
}

} // namespace gtest::rados
