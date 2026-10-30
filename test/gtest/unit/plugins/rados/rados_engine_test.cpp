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
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "rados_backend.h"
#include "rados_api.h"

namespace gtest::rados {

namespace {

    enum class stage { pending, retired };

    // Byte the fake writes into a read destination when modeling delivered payload.
    constexpr unsigned char kDeliveredByte = 0x5a;

    // Pre-set caller-destination byte proving that no read path wrote the buffer.
    constexpr unsigned char kSentinelByte = 0xc7;

    struct fakeCompletion {
        stage completion_stage = stage::pending;
        int rc = 0;
        bool waited = false;
        int releases = 0;
        bool released_unretired = false;
        bool submitted = false;
        bool is_stat = false;
        int stat_rc = 0;
        int stat_prval_rc = 0;
        uint64_t stat_size = 0;
        std::optional<uint64_t> *size_cell = nullptr;
        std::optional<int> *prval_cell = nullptr;
        std::optional<librados::bufferlist> *pbl_cell = nullptr;
    };

    struct submission {
        bool is_write;
        std::string oid;
        uint64_t offset;
        uintptr_t buf;
        uint64_t len;
        fakeCompletion *comp;
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

    class fakeRadosApi final : public nixl::rados::radosApi {
    public:
        fakeRadosApi()
            : log(std::make_shared<std::vector<std::string>>()),
              view_bind_order(std::make_shared<std::vector<librados::IoCtx *>>()),
              view_close_order(std::make_shared<std::vector<librados::IoCtx *>>()) {}

        ~fakeRadosApi() override {
            EXPECT_TRUE(violations.empty()) << violations;
            for (const auto &comp : completions) {
                EXPECT_EQ(comp->releases, 1) << "completion leaked or double-released";
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

        // Per-method call ordinals (0-based) that fail with -EINVAL, used to make
        // the second (metadata) client's bootstrap fail while the data client succeeds.
        int create2_fail_ordinal = -1;
        int conf_read_file_fail_ordinal = -1;
        int conf_set_fail_ordinal = -1;
        int connect_fail_ordinal = -1;
        int ioctx_create_fail_ordinal = -1;

        int create_completion_fail_after = -1;
        int submit_fail_after = -1;
        int write_throw_after = -1;
        int aio_stat_create_fail_at = -1;
        int aio_stat_submit_fail_at = -1;

        std::vector<submission> submissions;
        int waits = 0;
        int releases = 0;
        int rc_reads = 0;
        int create_attempts = 0;
        int stat_submits = 0;
        std::vector<fakeCompletion *> stat_comps;
        std::map<std::string, uint64_t> stat_sizes;
        std::map<std::string, int> stat_errors;
        std::vector<std::string> stat_oids;
        std::vector<std::string> create2_names;
        std::vector<std::string> conf_paths;
        std::vector<std::string> pools;
        std::string object_namespace;
        std::vector<std::string> namespaces;
        std::vector<librados::IoCtx *> stat_ctxs;
        std::vector<librados::IoCtx *> write_ctxs;
        std::vector<librados::IoCtx *> read_ctxs;
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
            ASSERT_NE(submissions[i].comp, nullptr);
            submissions[i].comp->rc = rc;
            if (submissions[i].is_write) {
                if (submissions[i].comp->is_stat) {
                    fillStatCells(submissions[i].comp);
                    submissions[i].comp->completion_stage = stage::retired;
                    return;
                }
                ASSERT_NE(submissions[i].payload, nullptr);
                submissions[i].payload_at_release = submissions[i].payload->to_str();
            } else {
                modelReadDelivery(submissions[i], rc);
            }
            submissions[i].comp->completion_stage = stage::retired;
        }

        void
        completeSubmission(size_t i) {
            ASSERT_LT(i, submissions.size());
            completeSubmission(i,
                               submissions[i].is_write ? 0 : static_cast<int>(submissions[i].len));
        }

        // Model librados replacing read output with a single owned buffer or split fragments.
        void
        replaceReadOutput(size_t i, const std::string &data, size_t chunk) {
            ASSERT_LT(i, submissions.size());
            ASSERT_FALSE(submissions[i].is_write);
            ASSERT_NE(submissions[i].payload, nullptr);
            ASSERT_FALSE(submissions[i].comp->completion_stage == stage::retired)
                << "replacement output must be configured before retirement";
            auto *bl = const_cast<librados::bufferlist *>(submissions[i].payload);
            ASSERT_NE(bl->get_num_buffers(), 0U);
            const uintptr_t caller_dram = PayloadAddr(*bl);
            bl->clear();
            if (chunk == 0 || chunk >= data.size()) {
                bl->append(data.data(), static_cast<unsigned>(data.size()));
                ASSERT_EQ(bl->get_num_buffers(), 1U);
            } else {
                for (size_t off = 0; off < data.size(); off += chunk) {
                    librados::bufferlist part;
                    part.append(data.data() + off,
                                static_cast<unsigned>(std::min(chunk, data.size() - off)));
                    bl->claim_append(part);
                }
                ASSERT_GT(bl->get_num_buffers(), 1U);
            }
            ASSERT_NE(PayloadAddr(*bl), caller_dram) << "replacement must not alias caller DRAM";
        }

        int
        create2(rados_t *cluster, const char *, const char *name, uint64_t) override {
            log->push_back("create2");
            create2_names.push_back(name);
            if (create2_fail_ordinal >= 0 &&
                static_cast<size_t>(create2_fail_ordinal) == clusters.size()) {
                return -EINVAL;
            }
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
            const size_t ordinal = conf_read_file_calls++;
            if (conf_read_file_fail_ordinal >= 0 &&
                static_cast<size_t>(conf_read_file_fail_ordinal) == ordinal) {
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
            const size_t ordinal = conf_set_calls++;
            if (conf_set_fail_ordinal >= 0 &&
                static_cast<size_t>(conf_set_fail_ordinal) == ordinal) {
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
            const size_t ordinal = connect_calls++;
            if (connect_fail_ordinal >= 0 && static_cast<size_t>(connect_fail_ordinal) == ordinal) {
                return -EINVAL;
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
            const size_t ordinal = ioctx_create_calls++;
            if (ioctx_create_fail_ordinal >= 0 &&
                static_cast<size_t>(ioctx_create_fail_ordinal) == ordinal) {
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
            if (nspace != nullptr) {
                namespaces.emplace_back(nspace);
            }
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
        ioctxClose(librados::IoCtx *ctx) override {
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
                if (sub.comp != nullptr && sub.comp->releases == 0) {
                    violations += "IoCtx closed with an unreleased completion\n";
                }
            }
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

        librados::AioCompletion *
        createCompletion() noexcept override {
            log->push_back("create_completion");
            const int attempt = create_attempts++;
            if (aio_stat_create_fail_at >= 0 && attempt == aio_stat_create_fail_at) {
                return nullptr;
            }
            if (create_completion_fail_after >= 0 &&
                static_cast<int>(completions.size()) >= create_completion_fail_after) {
                return nullptr;
            }
            completions.push_back(std::make_unique<fakeCompletion>());
            return reinterpret_cast<librados::AioCompletion *>(completions.back().get());
        }

        int
        aioStat(librados::IoCtx *ctx,
                const char *oid,
                librados::AioCompletion *completion,
                std::optional<uint64_t> *psize,
                std::optional<int> *prval,
                std::optional<librados::bufferlist> *pbl) override {
            log->push_back("aio_stat");
            EXPECT_NE(ctx, nullptr);
            stat_ctxs.push_back(ctx);
            fakeCompletion *comp = asFakeCompletion(completion);
            if (comp == nullptr) {
                violations += "async stat submitted with a foreign completion token\n";
                return -EINVAL;
            }
            if (psize == nullptr || prval == nullptr || pbl == nullptr) {
                violations += "async stat submitted with null output cells\n";
                return -EINVAL;
            }
            psize->emplace(0);
            prval->emplace(0);
            pbl->emplace();
            stat_oids.emplace_back(oid);
            comp->is_stat = true;
            comp->size_cell = psize;
            comp->prval_cell = prval;
            comp->pbl_cell = pbl;
            const auto err = stat_errors.find(oid);
            const auto found = stat_sizes.find(oid);
            if (err != stat_errors.end()) {
                comp->stat_rc = err->second;
                comp->stat_prval_rc = err->second;
            } else if (found != stat_sizes.end()) {
                comp->stat_rc = 0;
                comp->stat_prval_rc = 0;
                comp->stat_size = found->second;
            } else {
                comp->stat_rc = -ENOENT;
                comp->stat_prval_rc = -EIO;
            }
            stat_comps.push_back(comp);
            const int ordinal = stat_submits++;
            if (aio_stat_submit_fail_at >= 0 && ordinal == aio_stat_submit_fail_at) {
                return -EIO;
            }
            submissions.push_back({true, oid, 0, 0, 0, comp, nullptr, {}, 0});
            comp->submitted = true;
            return 0;
        }

        int
        aioRead(librados::IoCtx *ctx,
                const char *oid,
                librados::AioCompletion *completion,
                librados::bufferlist *bl,
                size_t len,
                uint64_t off) override {
            log->push_back("aio_read");
            EXPECT_NE(ctx, nullptr);
            read_ctxs.push_back(ctx);
            fakeCompletion *comp = asFakeCompletion(completion);
            if (comp == nullptr || bl == nullptr) {
                violations += "read submitted with invalid arguments\n";
                return -EINVAL;
            }
            if (bl->length() != len || PayloadAddr(*bl) == 0) {
                violations += "read payload does not describe caller DRAM\n";
            }
            if (submit_fail_after >= 0 &&
                static_cast<int>(submissions.size()) >= submit_fail_after) {
                return -EIO;
            }
            submissions.push_back(
                {false, oid, off, PayloadAddr(*bl), len, comp, bl, {}, PayloadAddr(*bl)});
            comp->submitted = true;
            return 0;
        }

        int
        aioWrite(librados::IoCtx *ctx,
                 const char *oid,
                 librados::AioCompletion *completion,
                 const librados::bufferlist &bl,
                 size_t len,
                 uint64_t off) override {
            log->push_back("aio_write");
            EXPECT_NE(ctx, nullptr);
            write_ctxs.push_back(ctx);
            fakeCompletion *comp = asFakeCompletion(completion);
            if (comp == nullptr) {
                violations += "write submitted with a foreign completion token\n";
                return -EINVAL;
            }
            if (bl.length() != len) {
                violations += "write payload length does not match the request\n";
            }
            if (write_throw_after >= 0 &&
                static_cast<int>(submissions.size()) >= write_throw_after) {
                throw std::bad_alloc();
            }
            if (submit_fail_after >= 0 &&
                static_cast<int>(submissions.size()) >= submit_fail_after) {
                return -EIO;
            }
            submissions.push_back({true, oid, off, 0, len, comp, &bl, {}, PayloadAddr(bl)});
            comp->submitted = true;
            return 0;
        }

        int
        isCompleteAndCb(librados::AioCompletion *completion) override {
            log->push_back("is_complete_and_cb");
            fakeCompletion *comp = asFakeCompletion(completion);
            if (comp == nullptr) {
                violations += "poll on a foreign completion token\n";
                return 0;
            }
            return comp->completion_stage == stage::retired ? 1 : 0;
        }

        int
        getReturnValue(librados::AioCompletion *completion) override {
            log->push_back("get_return_value");
            fakeCompletion *comp = asFakeCompletion(completion);
            if (comp == nullptr) {
                violations += "return value read on a foreign completion token\n";
                return -EINVAL;
            }
            ++rc_reads;
            if (comp->completion_stage != stage::retired) {
                violations += "return value read before complete-and-cb\n";
            }
            return comp->rc;
        }

        int
        waitForCompleteAndCb(librados::AioCompletion *completion) override {
            log->push_back("wait_for_complete_and_cb");
            fakeCompletion *comp = asFakeCompletion(completion);
            if (comp == nullptr) {
                violations += "wait on a foreign completion token\n";
                return -EINVAL;
            }
            ++waits;
            comp->waited = true;
            blockOnGateIfNeeded();
            if (comp->completion_stage != stage::retired) {
                fillStatCells(comp);
            }
            comp->completion_stage = stage::retired;
            return 0;
        }

        void
        release(librados::AioCompletion *completion) override {
            log->push_back("release");
            fakeCompletion *comp = asFakeCompletion(completion);
            if (comp == nullptr) {
                violations += "release on a foreign completion token\n";
                return;
            }
            ++releases;
            ++comp->releases;
            if (comp->releases > 1) {
                violations += "completion released more than once\n";
            }
            if (comp->submitted && comp->completion_stage != stage::retired) {
                violations += "submitted completion released without wait/proof\n";
                comp->released_unretired = true;
            }
            for (auto &sub : submissions) {
                if (sub.comp == comp) {
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

        // Direct fill only for unreplaced full reads; short reads deliver owned output.
        static void
        modelReadDelivery(const submission &sub, int rc) {
            if (rc < 0 || static_cast<uint64_t>(rc) > sub.len) {
                return;
            }
            auto *bl = const_cast<librados::bufferlist *>(sub.payload);
            if (!bl->is_provided_buffer(reinterpret_cast<const char *>(sub.buf))) {
                return;
            }
            if (static_cast<uint64_t>(rc) == sub.len) {
                std::fill_n(reinterpret_cast<unsigned char *>(sub.buf),
                            static_cast<size_t>(rc),
                            kDeliveredByte);
                return;
            }
            const std::string delivered(static_cast<size_t>(rc), kDeliveredByte);
            bl->clear();
            bl->append(delivered.data(), static_cast<unsigned>(rc));
        }

        static void
        fillStatCells(fakeCompletion *comp) {
            if (!comp->is_stat || comp->prval_cell == nullptr) {
                return;
            }
            *comp->prval_cell = comp->stat_prval_rc;
            *comp->size_cell = comp->stat_size;
            *comp->pbl_cell = librados::bufferlist{};
            comp->rc = comp->stat_rc;
        }

        static uintptr_t
        PayloadAddr(const librados::bufferlist &bl) {
            for (const auto &b : bl.buffers()) {
                return reinterpret_cast<uintptr_t>(b.c_str());
            }
            return 0;
        }

        fakeCompletion *
        asFakeCompletion(librados::AioCompletion *completion) {
            auto *raw = reinterpret_cast<fakeCompletion *>(completion);
            for (const auto &comp : completions) {
                if (comp.get() == raw) {
                    return raw;
                }
            }
            return nullptr;
        }

        std::vector<std::unique_ptr<fakeCompletion>> completions;

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
        size_t conf_read_file_calls = 0;
        size_t conf_set_calls = 0;
        size_t connect_calls = 0;
        size_t ioctx_create_calls = 0;

        static rados_t
        makeClusterToken(size_t ordinal) {
            return reinterpret_cast<rados_t>(
                static_cast<uintptr_t>(0xc10e0000u + 0x100u * (ordinal + 1)));
        }

        static rados_ioctx_t
        makeIoctxToken(size_t ordinal) {
            return reinterpret_cast<rados_ioctx_t>(
                static_cast<uintptr_t>(0x10c70002u + 0x100u * (ordinal + 1)));
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
    add_pair(nixl_meta_dlist_t &local,
             nixl_meta_dlist_t &remote,
             uintptr_t addr,
             uint64_t len,
             uint64_t object_offset,
             uint64_t dev_id) {
        local.addDesc(nixlMetaDesc(addr, len, 0));
        remote.addDesc(nixlMetaDesc(object_offset, len, dev_id));
    }

    // One DRAM/object pair over a fixture-owned buffer; mutate buffer in place so
    // descriptor-captured addresses stay valid.
    struct one_pair {
        one_pair(uint64_t object_offset, uint64_t length, uint64_t dev_id, unsigned char fill)
            : buffer(static_cast<size_t>(length), fill),
              local(DRAM_SEG),
              remote(OBJ_SEG) {
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
    makeEngine(fakeRadosApi **retained_fake = nullptr) {
        auto fake = std::make_unique<fakeRadosApi>();
        log = fake->log;
        if (retained_fake != nullptr) {
            *retained_fake = fake.get();
        }
        return std::make_unique<nixlRadosEngine>(&init_params_, std::move(fake));
    }

    // Same as makeEngine, but injects the fake status-wait timer.
    std::unique_ptr<nixlRadosEngine>
    makeEngineWithTimer(fakeRadosApi **retained_fake, const std::shared_ptr<timer_state> &state) {
        auto fake = std::make_unique<fakeRadosApi>();
        log = fake->log;
        if (retained_fake != nullptr) {
            *retained_fake = fake.get();
        }
        auto timer = std::make_unique<fakeStatusWaitTimer>(state);
        return std::make_unique<nixlRadosEngine>(&init_params_, std::move(fake), std::move(timer));
    }

    // Representative successful construction: validBase parameters plus the two client bootstraps.
    std::unique_ptr<nixlRadosEngine>
    makeReady(fakeRadosApi **retained_fake = nullptr) {
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

    // Three ordered pairs, one object; three rejection rounds; shared by write and read tests.
    void
    expectSubmissionStopsAtFirstRejectedPair(nixl_xfer_op_t op) {
        const bool expect_write = (op == NIXL_WRITE);
        const std::string verb = expect_write ? "aio_write" : "aio_read";
        const char *direction = expect_write ? "write" : "read";
        SCOPED_TRACE(direction);
        fakeRadosApi *fake = nullptr;
        auto engine = makeReady(&fake);
        nixlBackendMD *md = registerObject(*engine, 1, "obj");
        const one_pair pair(0, 192, 1, expect_write ? 0x33 : 0x00);
        const uintptr_t base = reinterpret_cast<uintptr_t>(pair.buffer.data());
        nixl_meta_dlist_t local(DRAM_SEG);
        nixl_meta_dlist_t remote(OBJ_SEG);
        for (int i = 0; i < 3; ++i) {
            add_pair(local,
                     remote,
                     base + static_cast<uintptr_t>(i) * 64,
                     64,
                     static_cast<uint64_t>(i) * 64,
                     1);
        }
        nixlBackendReqH *handle = nullptr;

        fake->submit_fail_after = 0;
        ASSERT_EQ(engine->prepXfer(op, local, remote, "test-agent", handle), NIXL_SUCCESS);
        EXPECT_EQ(engine->postXfer(op, local, remote, "test-agent", handle), NIXL_ERR_BACKEND);
        EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
        EXPECT_TRUE(fake->submissions.empty());
        EXPECT_EQ(fake->releases, 1);
        EXPECT_EQ(fake->waits, 0);
        EXPECT_EQ(fake->rc_reads, 0);
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);

        fake->submit_fail_after = 1;
        const int submits_before_second_round = std::count(log->begin(), log->end(), verb);
        ASSERT_EQ(engine->prepXfer(op, local, remote, "test-agent", handle), NIXL_SUCCESS);
        EXPECT_EQ(engine->postXfer(op, local, remote, "test-agent", handle), NIXL_IN_PROG);
        ASSERT_EQ(fake->submissions.size(), 1U);
        EXPECT_EQ(fake->submissions[0].offset, 0U) << "pairs submit in descriptor order";
        EXPECT_EQ(fake->submissions[0].is_write, expect_write)
            << "the accepted pair is a " << direction;
        EXPECT_EQ(fake->releases, 2) << "the rejected pair's completion is released without wait";
        EXPECT_EQ(fake->waits, 0);
        EXPECT_EQ(std::count(log->begin(), log->end(), verb), submits_before_second_round + 2)
            << "the third pair is never attempted";
        EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
        fake->completeSubmission(0);
        EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);

        fake->submit_fail_after = -1;
        const int creates_before =
            std::count(log->begin(), log->end(), std::string("create_completion"));
        const int releases_before = fake->releases;
        fake->create_completion_fail_after = creates_before + 1;
        ASSERT_EQ(engine->prepXfer(op, local, remote, "test-agent", handle), NIXL_SUCCESS);
        EXPECT_EQ(engine->postXfer(op, local, remote, "test-agent", handle), NIXL_IN_PROG);
        ASSERT_EQ(fake->submissions.size(), 2U);
        EXPECT_EQ(std::count(log->begin(), log->end(), std::string("create_completion")),
                  creates_before + 2)
            << "the factory-failed pair reserves no completion";
        EXPECT_EQ(fake->releases, releases_before)
            << "a factory-failed pair leaves nothing to release";
        EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
        fake->completeSubmission(1);
        EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);

        fake->create_completion_fail_after = 0;
        const int creates_at_factory_round =
            std::count(log->begin(), log->end(), std::string("create_completion"));
        const int releases_at_factory_round = fake->releases;
        const size_t submissions_at_factory_round = fake->submissions.size();
        const int verbs_at_factory_round = std::count(log->begin(), log->end(), verb);
        ASSERT_EQ(engine->prepXfer(op, local, remote, "test-agent", handle), NIXL_SUCCESS);
        EXPECT_EQ(engine->postXfer(op, local, remote, "test-agent", handle), NIXL_ERR_BACKEND)
            << "a factory failure with nothing accepted terminates directly";
        EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
        EXPECT_EQ(fake->submissions.size(), submissions_at_factory_round)
            << "the factory-failed request submits nothing";
        EXPECT_EQ(fake->releases, releases_at_factory_round)
            << "a failed factory allocation leaves nothing to release";
        EXPECT_EQ(fake->waits, 0);
        EXPECT_EQ(std::count(log->begin(), log->end(), std::string("create_completion")),
                  creates_at_factory_round + 1)
            << "later pairs are never attempted after the factory fails";
        EXPECT_EQ(std::count(log->begin(), log->end(), verb), verbs_at_factory_round)
            << "the factory-failed pair never reaches submission";
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
        EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
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

TEST_F(radosEngineFixture, BootstrapsTwoClientsAndRetiresMetadataFirst) {
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

    fakeRadosApi *fake = nullptr;
    std::shared_ptr<std::vector<librados::IoCtx *>> binds;
    std::shared_ptr<std::vector<librados::IoCtx *>> closes;
    {
        auto engine = makeEngine(&fake);
        ASSERT_FALSE(engine->getInitErr());
        std::vector<std::string> running = client_bootstrap;
        running.insert(running.end(), client_bootstrap.begin(), client_bootstrap.end());
        EXPECT_EQ(*log, running);
        EXPECT_EQ(fake->create2_names, (std::vector<std::string>{"client.qa", "client.qa"}));
        EXPECT_EQ(fake->conf_paths, (std::vector<std::string>{"fixture.conf", "fixture.conf"}));
        EXPECT_EQ(fake->pools, (std::vector<std::string>{"qa-pool", "qa-pool"}));
        EXPECT_EQ(fake->options,
                  (std::vector<std::pair<std::string, std::string>>{
                      {"alpha", ""}, {"zeta", "3"}, {"alpha", ""}, {"zeta", "3"}}));
        ASSERT_EQ(fake->view_bind_order->size(), 2U);
        EXPECT_NE((*fake->view_bind_order)[0], (*fake->view_bind_order)[1])
            << "data and metadata views must be distinct IoCtx objects";
        binds = fake->view_bind_order;
        closes = fake->view_close_order;
    }
    std::vector<std::string> final_log = client_bootstrap;
    final_log.insert(final_log.end(), client_bootstrap.begin(), client_bootstrap.end());
    final_log.insert(final_log.end(), client_teardown.begin(), client_teardown.end());
    final_log.insert(final_log.end(), client_teardown.begin(), client_teardown.end());
    EXPECT_EQ(*log, final_log);
    ASSERT_EQ(closes->size(), 2U);
    EXPECT_EQ((*closes)[0], (*binds)[1]) << "the metadata view must close first";
    EXPECT_EQ((*closes)[1], (*binds)[0]) << "the data view must close last";
}

TEST_F(radosEngineFixture, NamespaceConfiguredOncePerClientBetweenIoctxCreateAndView) {
    custom_params_ = validBase();
    custom_params_["namespace"] = "qa-ns";
    fakeRadosApi *fake = nullptr;
    {
        auto engine = makeEngine(&fake);
        ASSERT_FALSE(engine->getInitErr());
        const std::vector<std::string> client_bootstrap{"create2",
                                                        "conf_read_file",
                                                        "connect",
                                                        "ioctx_create",
                                                        "ioctx_set_namespace",
                                                        "ioctx_from_raw"};
        std::vector<std::string> running = client_bootstrap;
        running.insert(running.end(), client_bootstrap.begin(), client_bootstrap.end());
        EXPECT_EQ(*log, running);
        EXPECT_EQ(fake->object_namespace, "qa-ns");
        EXPECT_EQ(fake->namespaces, (std::vector<std::string>{"qa-ns", "qa-ns"}));
        EXPECT_EQ(std::count(log->begin(), log->end(), std::string("ioctx_set_namespace")), 2)
            << "each client's raw ioctx must be configured exactly once";
    }
}

TEST_F(radosEngineFixture, BootstrapFailureUnwindsAndGatesRegistration) {
    custom_params_ = validBase();
    auto failing = std::make_unique<fakeRadosApi>();
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
        EXPECT_EQ(*log,
                  (std::vector<std::string>{"create2", "conf_read_file", "connect", "shutdown"}));
    }
    EXPECT_EQ(*log, (std::vector<std::string>{"create2", "conf_read_file", "connect", "shutdown"}));
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
    fakeRadosApi *fake = nullptr;
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
    fakeRadosApi *fake = nullptr;
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
    fakeRadosApi *fake = nullptr;
    auto engine = makeReady(&fake);
    [[maybe_unused]] nixlBackendMD *md_1 = registerObject(*engine, 1, "obj");
    [[maybe_unused]] nixlBackendMD *md_2 = registerObject(*engine, 2, "obj");
    [[maybe_unused]] nixlBackendMD *md_3 = registerObject(*engine, 3, "obj-3");

    const uintptr_t base = 0x10000;
    const uintptr_t top = std::numeric_limits<uintptr_t>::max();
    const uint64_t otop = std::numeric_limits<uint64_t>::max();
    using pair_spec = std::tuple<uintptr_t, uint64_t, uint64_t, uint64_t>; // addr,len,off,dev

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
         static_cast<nixl_xfer_op_t>(99),
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, 64, 0, 1}},
         NIXL_ERR_NOT_SUPPORTED},
        {"local segment",
         NIXL_WRITE,
         FILE_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, 64, 0, 1}},
         NIXL_ERR_NOT_SUPPORTED},
        {"remote segment",
         NIXL_WRITE,
         DRAM_SEG,
         DRAM_SEG,
         "test-agent",
         {{base, 64, 0, 1}},
         NIXL_ERR_NOT_SUPPORTED},
        {"foreign agent",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "other-agent",
         {{base, 64, 0, 1}},
         NIXL_ERR_NOT_SUPPORTED},
        {"empty list", NIXL_WRITE, DRAM_SEG, OBJ_SEG, "test-agent", {}, NIXL_ERR_INVALID_PARAM},
        {"zero length",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, 0, 0, 1}},
         NIXL_ERR_INVALID_PARAM},
        {"length above INT_MAX",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, uint64_t(INT_MAX) + 1, 0, 1}},
         NIXL_ERR_INVALID_PARAM},
        {"null dram address",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{0, 64, 0, 1}},
         NIXL_ERR_INVALID_PARAM},
        {"local range wrap",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{top - 8, 64, 0, 1}},
         NIXL_ERR_INVALID_PARAM},
        {"object offset wrap",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, 64, otop - 8, 1}},
         NIXL_ERR_INVALID_PARAM},
        {"unregistered devId",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, 64, 0, 42}},
         NIXL_ERR_INVALID_PARAM},
        {"overlapping writes on one devId",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, 64, 0, 1}, {base + 64, 64, 32, 1}},
         NIXL_ERR_INVALID_PARAM},
        {"overlapping writes across aliased devIds",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, 64, 0, 1}, {base + 64, 64, 32, 2}},
         NIXL_ERR_INVALID_PARAM},
        {"overlapping reads are legal",
         NIXL_READ,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, 64, 0, 1}, {base + 64, 64, 32, 1}},
         NIXL_SUCCESS},
        {"adjacent writes across aliased devIds",
         NIXL_WRITE,
         DRAM_SEG,
         OBJ_SEG,
         "test-agent",
         {{base, 64, 0, 1}, {base + 64, 64, 64, 2}},
         NIXL_SUCCESS},
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
        remote.addDesc(nixlMetaDesc(64, 64, 2)); // descriptor-count mismatch
        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, "test-agent", handle),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(handle, nullptr);
    }
    {
        const one_pair single(0, 64, 1, 0x00);
        nixl_meta_dlist_t short_remote(OBJ_SEG);
        short_remote.addDesc(nixlMetaDesc(0, 63, 1)); // unequal pair lengths
        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(engine->prepXfer(NIXL_WRITE, single.local, short_remote, "test-agent", handle),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(handle, nullptr);
    }
    EXPECT_TRUE(fake->submissions.empty());
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("aio_read")), 0);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("aio_write")), 0);

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
    fakeRadosApi *fake = nullptr;
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

// reads describe caller destination memory directly; results map per rc.
TEST_F(radosEngineFixture, ReadResultMappingAndDirectCallerDestination) {
    fakeRadosApi *fake = nullptr;
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
            EXPECT_EQ(std::count(log->begin(), log->end(), std::string("aio_write")), 0);
        }
        fake->completeSubmission(i, rc);
        EXPECT_EQ(engine->checkXfer(handle), expected) << "rc=" << rc;
        EXPECT_EQ(engine->checkXfer(handle), expected);
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    }
    EXPECT_EQ(fake->waits, 0) << "poll-driven harvest must never wait";
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

// A single-buffer replaced read output is copied into the caller destination before release.
TEST_F(radosEngineFixture, SingleBufferReplacementReadOutputCopiesIntoCallerDestination) {
    fakeRadosApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md = registerObject(*engine, 1, "obj");
    const one_pair pair(0, 64, 1, kSentinelByte);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
              NIXL_IN_PROG);
    fake->replaceReadOutput(0, std::string(64, 0x5c), 0);
    fake->completeSubmission(0, 64);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(pair.buffer, (std::vector<unsigned char>(64, 0x5c)))
        << "the single-buffer replacement must be copied into the caller destination";
    EXPECT_EQ(fake->releases, 1) << "the settled read releases its completion";
    EXPECT_EQ(fake->waits, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

// A fragmented replacement is copied fragment-by-fragment, never rejected for shape.
TEST_F(radosEngineFixture, FragmentedReplacementReadOutputCopiesIntoCallerDestination) {
    fakeRadosApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md = registerObject(*engine, 1, "obj");
    const one_pair pair(0, 64, 1, kSentinelByte);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, pair.local, pair.remote, "test-agent", handle),
              NIXL_IN_PROG);
    fake->replaceReadOutput(0, std::string(64, 0x5c), 16);
    fake->completeSubmission(0, 64);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(pair.buffer, (std::vector<unsigned char>(64, 0x5c)))
        << "the fragmented replacement must be copied into the caller destination";
    EXPECT_EQ(fake->releases, 1) << "the settled read releases its completion";
    EXPECT_EQ(fake->waits, 0);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS) << "terminal status stays sticky";
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

// Writes borrow caller memory through a non-owning static view; only zero succeeds.
TEST_F(radosEngineFixture, WriteResultMappingAndNonOwningPayload) {
    fakeRadosApi *fake = nullptr;
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
            EXPECT_EQ(std::count(log->begin(), log->end(), std::string("aio_read")), 0);
        }
        fake->completeSubmission(i, rc);
        EXPECT_EQ(engine->checkXfer(handle), expected) << "rc=" << rc;
        if (i == 0) {
            EXPECT_EQ(sub.payload_at_release, std::string(64, 0x11));
        }
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    }
    EXPECT_EQ(fake->waits, 0) << "poll-driven harvest must never wait";
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

// The two directions share the rounds; see expectSubmissionStopsAtFirstRejectedPair.
TEST_F(radosEngineFixture, WriteSubmissionStopsAtFirstRejectedPairAndOwnsCompletions) {
    expectSubmissionStopsAtFirstRejectedPair(NIXL_WRITE);
}

TEST_F(radosEngineFixture, ReadSubmissionStopsAtFirstRejectedPairAndOwnsCompletions) {
    expectSubmissionStopsAtFirstRejectedPair(NIXL_READ);
}

TEST_F(radosEngineFixture, AioWriteThrowAfterAcceptedPrefixReleasesWithoutWait) {
    fakeRadosApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md = registerObject(*engine, 1, "obj");
    const one_pair pair(0, 192, 1, 0x3a);
    const uintptr_t base = reinterpret_cast<uintptr_t>(pair.buffer.data());
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    for (int i = 0; i < 3; ++i) {
        add_pair(local,
                 remote,
                 base + static_cast<uintptr_t>(i) * 64,
                 64,
                 static_cast<uint64_t>(i) * 64,
                 1);
    }

    fake->write_throw_after = 1;
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_SUCCESS);
    EXPECT_NO_THROW(
        EXPECT_EQ(engine->postXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_IN_PROG));
    EXPECT_EQ(fake->submissions.size(), 1U);
    EXPECT_EQ(fake->releases, 1);
    EXPECT_EQ(fake->waits, 0);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("aio_write")), 2);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("create_completion")), 2);

    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    fake->completeSubmission(0);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_ERR_BACKEND);
    EXPECT_EQ(fake->waits, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_TRUE(fake->violations.empty());
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, OutOfOrderRetirementKeepsRequestInProgressUntilEarliestPairResolves) {
    fakeRadosApi *fake = nullptr;
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
    EXPECT_EQ(fake->rc_reads, 2);
    EXPECT_EQ(fake->waits, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("ioctx_destroy")), 0);
    EXPECT_EQ(engine->deregisterMem(md_a), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_b), NIXL_SUCCESS);
}

// Pending pairs stay IN_PROG for the wait-driven release drain; retired pairs harvest poll-only.
TEST_F(radosEngineFixture, PollAndDrainHarvestCoverBothDirections) {
    fakeRadosApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md_r = registerObject(*engine, 1, "obj-r");
    nixlBackendMD *md_w = registerObject(*engine, 2, "obj-w");

    const one_pair read(0, 64, 1, 0x00);
    nixlBackendReqH *read_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_IN_PROG);
    EXPECT_EQ(engine->checkXfer(read_handle), NIXL_IN_PROG);
    EXPECT_EQ(fake->rc_reads, 0);
    EXPECT_EQ(fake->releases, 0);
    EXPECT_FALSE(fake->submissions[0].is_write) << "pair 0 is the read";
    EXPECT_EQ(engine->releaseReqH(read_handle), NIXL_SUCCESS);
    EXPECT_EQ(fake->waits, 1);
    EXPECT_EQ(fake->rc_reads, 1);
    EXPECT_EQ(fake->releases, 1);
    EXPECT_TRUE(fake->submissions[0].comp->waited);
    EXPECT_FALSE(fake->submissions[0].comp->released_unretired);

    const one_pair write(0, 64, 2, 0x11);
    nixlBackendReqH *write_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
              NIXL_IN_PROG);
    EXPECT_TRUE(fake->submissions[1].is_write) << "pair 1 is the write";
    fake->completeSubmission(1);
    EXPECT_EQ(engine->checkXfer(write_handle), NIXL_SUCCESS);
    EXPECT_EQ(fake->rc_reads, 2) << "the write harvest adds exactly one read";
    EXPECT_EQ(fake->waits, 1) << "poll-driven write harvest never waits";
    EXPECT_FALSE(fake->submissions[1].comp->waited) << "the write completion was never waited";
    EXPECT_EQ(engine->releaseReqH(write_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_r), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_w), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, EarlyReleaseBlocksOnWithheldWaitThenDrainsBothDirections) {
    custom_params_ = validBase();
    std::vector<unsigned char> src(64, 0x44);
    std::vector<unsigned char> dest(64, 0x00);
    fakeRadosApi *fake = nullptr;
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
    std::thread releaser([&write_release, &engine, write_handle]() {
        write_release.set_value(engine->releaseReqH(write_handle));
    });

    const bool gate_entered = fake->waitGateEntered();
    bool withheld = false;
    int waits_at_gate = -1;
    int releases_at_gate = -1;
    bool read_drain_touched_at_gate = false;
    if (gate_entered) {
        withheld = write_status.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
        waits_at_gate = fake->waits;
        releases_at_gate = fake->releases;
        auto *read_comp = fake->submissions[1].comp;
        read_drain_touched_at_gate = read_comp->waited || read_comp->releases != 0;
    }

    fake->releaseWaitGate();
    releaser.join();

    ASSERT_TRUE(gate_entered) << "the drain never reached the completion wait";
    EXPECT_TRUE(withheld) << "releaseReqH returned while its completion wait was withheld";
    EXPECT_EQ(waits_at_gate, 1);
    EXPECT_EQ(releases_at_gate, 0);
    EXPECT_FALSE(read_drain_touched_at_gate) << "the read request must stay untouched";
    EXPECT_EQ(write_status.get(), NIXL_SUCCESS);

    const std::vector<std::string> write_tail(
        log->begin() + static_cast<std::ptrdiff_t>(log_before), log->end());
    const auto w_wait =
        std::find(write_tail.begin(), write_tail.end(), std::string("wait_for_complete_and_cb"));
    const auto w_release = std::find(write_tail.begin(), write_tail.end(), std::string("release"));
    ASSERT_NE(w_wait, write_tail.end());
    ASSERT_NE(w_release, write_tail.end());
    EXPECT_LT(w_wait, w_release);
    EXPECT_TRUE(fake->submissions[0].is_write) << "the released handle is the write request";
    EXPECT_FALSE(fake->submissions[1].is_write) << "the untouched handle is the read request";
    EXPECT_TRUE(fake->submissions[0].comp->waited);
    EXPECT_FALSE(fake->submissions[0].comp->released_unretired);
    EXPECT_EQ(fake->releases, 1) << "only the write request drained";

    EXPECT_EQ(engine->releaseReqH(read_handle), NIXL_SUCCESS);
    EXPECT_TRUE(fake->submissions[1].comp->waited);
    EXPECT_FALSE(fake->submissions[1].comp->released_unretired);
    EXPECT_EQ(fake->releases, 2) << "the read drain adds exactly one release";
    EXPECT_EQ(fake->submissions[0].comp->releases, 1)
        << "the read drain must not touch the write completion";

    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("ioctx_destroy")), 0);
    EXPECT_EQ(engine->deregisterMem(md_w), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_r), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, TeardownDrainsBothDirectionsBeforeResourceDeath) {
    custom_params_ = validBase();
    std::vector<unsigned char> src(64, 0x77);
    std::vector<unsigned char> dest(64, 0x00);
    {
        fakeRadosApi *fake = nullptr;
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
            return entry == "release";
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
        std::count(final_log.begin(), final_log.end(), std::string("wait_for_complete_and_cb")), 1);
    EXPECT_EQ(std::count(final_log.begin(), final_log.end(), std::string("release")), 2);
}

// Repost is legal only from terminal: a fresh completion and buffer state is
// rebuilt from the snapshotted pointer, re-staging the caller's current bytes.
TEST_F(radosEngineFixture, RepostAfterTerminalResubmitsResetStateWithCurrentSourceBytes) {
    fakeRadosApi *fake = nullptr;
    auto engine = makeReady(&fake);
    nixlBackendMD *md_w = registerObject(*engine, 1, "obj-w");
    nixlBackendMD *md_r = registerObject(*engine, 2, "obj-r");

    one_pair write(0, 64, 1, 0x99);
    nixlBackendReqH *write_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, write.local, write.remote, "test-agent", write_handle),
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
    EXPECT_NE(fake->submissions[1].comp, fake->submissions[0].comp);
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
    EXPECT_NE(fake->submissions[3].comp, fake->submissions[2].comp);
    fake->completeSubmission(3, 64);
    EXPECT_EQ(engine->checkXfer(read_handle), NIXL_SUCCESS)
        << "the repost must reset the stale failed result cell";
    EXPECT_EQ(engine->releaseReqH(read_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_w), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md_r), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, QueryResolvesOidFromMetaInfoThenDecimalDevId) {
    custom_params_ = validBase();
    fakeRadosApi *fake = nullptr;
    auto engine = makeEngine(&fake);
    ASSERT_FALSE(engine->getInitErr());

    // Registration-independent: no registerMem call precedes the query.
    fake->stat_sizes["named-obj"] = 1;
    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 0, 7, "named-obj"));
    descs.addDesc(nixlBlobDesc(0, 0, 42, ""));
    std::vector<nixl_query_resp_t> resp(4, nixl_query_resp_t{nixl_b_params_t{}});
    ASSERT_EQ(engine->queryMem(descs, resp), NIXL_SUCCESS);
    EXPECT_EQ(fake->stat_oids, (std::vector<std::string>{"named-obj", "42"}));
    ASSERT_EQ(resp.size(), 2u);
    ASSERT_TRUE(resp[0].has_value());
    EXPECT_EQ(resp[0]->at("size"), "1");
    EXPECT_FALSE(resp[1].has_value());
}

TEST_F(radosEngineFixture, QueryContinuesAfterErrorAndPopulatesLaterSuccess) {
    custom_params_ = validBase();
    fakeRadosApi *fake = nullptr;
    auto engine = makeEngine(&fake);
    ASSERT_FALSE(engine->getInitErr());

    fake->stat_sizes["zero-obj"] = 0;
    fake->stat_errors["bad-obj"] = -EIO;
    fake->stat_sizes["later-obj"] = 30;
    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 0, 1, "zero-obj"));
    descs.addDesc(nixlBlobDesc(0, 0, 2, "bad-obj"));
    descs.addDesc(nixlBlobDesc(0, 0, 3, "later-obj"));
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_ERR_BACKEND);
    ASSERT_EQ(resp.size(), 3u);
    ASSERT_TRUE(resp[0].has_value());
    EXPECT_EQ(resp[0]->at("size"), "0");
    EXPECT_FALSE(resp[1].has_value());
    ASSERT_TRUE(resp[2].has_value());
    EXPECT_EQ(resp[2]->at("size"), "30");
    EXPECT_EQ(fake->stat_oids, (std::vector<std::string>{"zero-obj", "bad-obj", "later-obj"}));
}

TEST_F(radosEngineFixture, QueryGatesClearAndResizeResponseWithoutSeamCall) {
    struct gate_case {
        const char *name;
        bool init_err;
        nixl_mem_t list_type;
        int desc_count;
        nixl_status_t expected;
    };

    const gate_case cases[] = {
        {"initialization error", true, OBJ_SEG, 1, NIXL_ERR_BACKEND},
        {"non-OBJ_SEG list", false, DRAM_SEG, 1, NIXL_ERR_NOT_SUPPORTED},
        {"empty OBJ_SEG list", false, OBJ_SEG, 0, NIXL_SUCCESS},
    };
    for (const auto &item : cases) {
        custom_params_ = validBase();
        if (item.init_err) {
            custom_params_.erase("pool");
        }
        fakeRadosApi *fake = nullptr;
        auto engine = makeEngine(&fake);
        EXPECT_EQ(engine->getInitErr(), item.init_err) << item.name;

        nixl_reg_dlist_t descs(item.list_type);
        for (int i = 0; i < item.desc_count; ++i) {
            descs.addDesc(nixlBlobDesc(0, 0, 7, "obj-a"));
        }
        std::vector<nixl_query_resp_t> resp(2, nixl_query_resp_t{nixl_b_params_t{}});
        EXPECT_EQ(engine->queryMem(descs, resp), item.expected) << item.name;
        EXPECT_EQ(resp.size(), static_cast<size_t>(item.desc_count)) << item.name;
        for (const auto &entry : resp) {
            EXPECT_FALSE(entry.has_value()) << item.name;
        }
        EXPECT_TRUE(fake->stat_oids.empty()) << item.name;
        EXPECT_EQ(std::count(log->begin(), log->end(), std::string("aio_stat")), 0) << item.name;
    }
}

TEST_F(radosEngineFixture, QueryRejectsEmbeddedNulMetaInfoBeforeAnyStat) {
    custom_params_ = validBase();
    fakeRadosApi *fake = nullptr;
    auto engine = makeEngine(&fake);
    ASSERT_FALSE(engine->getInitErr());

    // Statable prefixes: any submitted stat on them would be a NUL-truncation alias.
    fake->stat_sizes["bad"] = 7;
    fake->stat_sizes["key"] = 9;
    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 0, 1, "untouched"));
    descs.addDesc(nixlBlobDesc(0, 0, 2, std::string("bad\0key", 7)));
    descs.addDesc(nixlBlobDesc(0, 0, 3, std::string("key\0", 4)));
    std::vector<nixl_query_resp_t> resp(1, nixl_query_resp_t{nixl_b_params_t{}});
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_ERR_INVALID_PARAM);
    ASSERT_EQ(resp.size(), 3u);
    for (const auto &entry : resp) {
        EXPECT_FALSE(entry.has_value());
    }
    EXPECT_TRUE(fake->stat_oids.empty()) << "an embedded-NUL list must not alias-stat its prefix";
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("aio_stat")), 0);

    // Rejection is per-call: a clean list resolves normally afterwards.
    nixl_reg_dlist_t clean(OBJ_SEG);
    clean.addDesc(nixlBlobDesc(0, 0, 4, "bad"));
    EXPECT_EQ(engine->queryMem(clean, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 1u);
    ASSERT_TRUE(resp[0].has_value());
    EXPECT_EQ(resp[0]->at("size"), "7");
    EXPECT_EQ(fake->stat_oids, (std::vector<std::string>{"bad"}));
}

TEST_F(radosEngineFixture, QueryWindowsPreserveFoundMissingAndErrorOrder) {
    custom_params_ = validBase();
    fakeRadosApi *fake = nullptr;
    auto engine = makeEngine(&fake);
    ASSERT_FALSE(engine->getInitErr());

    // 300 descriptors span two 256-wide stat windows; found (including decimal
    // zero), missing, and failed stats interleave across the boundary.
    constexpr int kCount = 300;
    nixl_reg_dlist_t descs(OBJ_SEG);
    std::vector<std::string> expected_oids;
    for (int i = 0; i < kCount; ++i) {
        const std::string key = "win-" + std::to_string(i);
        if (i % 5 == 0) {
            fake->stat_sizes[key] = static_cast<uint64_t>(i);
        } else if (i % 7 == 0) {
            fake->stat_errors[key] = -EIO;
        }
        expected_oids.push_back(key);
        descs.addDesc(nixlBlobDesc(0, 0, static_cast<uint64_t>(i), key));
    }

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_ERR_BACKEND);
    ASSERT_EQ(resp.size(), static_cast<size_t>(kCount));
    for (int i = 0; i < kCount; ++i) {
        const nixl_query_resp_t &entry = resp[static_cast<size_t>(i)];
        if (i % 5 == 0) {
            ASSERT_TRUE(entry.has_value()) << i;
            EXPECT_EQ(entry->at("size"), std::to_string(i)) << i;
        } else {
            EXPECT_FALSE(entry.has_value()) << i;
        }
    }
    EXPECT_EQ(fake->stat_oids, expected_oids);
    EXPECT_EQ(fake->submissions.size(), static_cast<size_t>(kCount));
    EXPECT_EQ(fake->releases, kCount);
    EXPECT_EQ(fake->waits, kCount);
    EXPECT_EQ(fake->rc_reads, kCount);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("aio_stat")), kCount);

    size_t stats_before_first_wait = 0;
    for (const auto &entry : *log) {
        if (entry == "aio_stat") {
            ++stats_before_first_wait;
        } else if (entry == "wait_for_complete_and_cb") {
            break;
        }
    }
    EXPECT_EQ(stats_before_first_wait, 256U) << "the in-flight stat window must cap at 256";
}

TEST_F(radosEngineFixture, QueryCreateFailureMidWindowLeaksNothingAndLaterStatsSucceed) {
    custom_params_ = validBase();
    fakeRadosApi *fake = nullptr;
    auto engine = makeEngine(&fake);
    ASSERT_FALSE(engine->getInitErr());

    constexpr int kCount = 40;
    nixl_reg_dlist_t descs(OBJ_SEG);
    for (int i = 0; i < kCount; ++i) {
        const std::string key = "cf-" + std::to_string(i);
        fake->stat_sizes[key] = static_cast<uint64_t>(i) + 1;
        descs.addDesc(nixlBlobDesc(0, 0, static_cast<uint64_t>(i), key));
    }
    fake->aio_stat_create_fail_at = 3;

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_ERR_BACKEND);
    ASSERT_EQ(resp.size(), static_cast<size_t>(kCount));
    for (int i = 0; i < kCount; ++i) {
        const nixl_query_resp_t &entry = resp[static_cast<size_t>(i)];
        if (i == 3) {
            EXPECT_FALSE(entry.has_value());
            continue;
        }
        ASSERT_TRUE(entry.has_value()) << i;
        EXPECT_EQ(entry->at("size"), std::to_string(i + 1)) << i;
    }
    EXPECT_EQ(fake->create_attempts, kCount);
    EXPECT_EQ(fake->submissions.size(), static_cast<size_t>(kCount - 1));
    EXPECT_EQ(fake->releases, kCount - 1);
    EXPECT_EQ(fake->waits, kCount - 1);
}

TEST_F(radosEngineFixture, QuerySubmitFailureMidWindowReleasesOnceAndLaterStatsSucceed) {
    custom_params_ = validBase();
    fakeRadosApi *fake = nullptr;
    auto engine = makeEngine(&fake);
    ASSERT_FALSE(engine->getInitErr());

    constexpr int kCount = 40;
    nixl_reg_dlist_t descs(OBJ_SEG);
    for (int i = 0; i < kCount; ++i) {
        const std::string key = "sf-" + std::to_string(i);
        fake->stat_sizes[key] = static_cast<uint64_t>(i) + 1;
        descs.addDesc(nixlBlobDesc(0, 0, static_cast<uint64_t>(i), key));
    }
    fake->aio_stat_submit_fail_at = 35;

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_ERR_BACKEND);
    ASSERT_EQ(resp.size(), static_cast<size_t>(kCount));
    for (int i = 0; i < kCount; ++i) {
        const nixl_query_resp_t &entry = resp[static_cast<size_t>(i)];
        if (i == 35) {
            EXPECT_FALSE(entry.has_value());
            continue;
        }
        ASSERT_TRUE(entry.has_value()) << i;
        EXPECT_EQ(entry->at("size"), std::to_string(i + 1)) << i;
    }
    EXPECT_EQ(fake->submissions.size(), static_cast<size_t>(kCount - 1));
    EXPECT_EQ(fake->releases, kCount);
    EXPECT_EQ(fake->waits, kCount - 1);
    ASSERT_EQ(fake->stat_comps.size(), static_cast<size_t>(kCount));
    fakeCompletion *rejected = fake->stat_comps[35];
    EXPECT_FALSE(rejected->submitted);
    EXPECT_FALSE(rejected->waited);
    EXPECT_FALSE(rejected->released_unretired);
    EXPECT_EQ(rejected->releases, 1);
}

TEST_F(radosEngineFixture, QueryResolvesBeforeReturnWithoutStatusWaitTimer) {
    custom_params_ = validBase();
    custom_params_["status_wait_us"] = "4000";
    auto tstate = std::make_shared<timer_state>();
    fakeRadosApi *fake = nullptr;
    auto engine = makeEngineWithTimer(&fake, tstate);
    ASSERT_FALSE(engine->getInitErr());

    nixl_reg_dlist_t descs(OBJ_SEG);
    for (int i = 0; i < 36; ++i) {
        descs.addDesc(nixlBlobDesc(0, 0, static_cast<uint64_t>(i), "qt-" + std::to_string(i)));
    }
    fake->stat_sizes["qt-7"] = 77;

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 36u);
    ASSERT_TRUE(resp[7].has_value());
    EXPECT_EQ(resp[7]->at("size"), "77");
    EXPECT_TRUE(tstate->budgets.empty()) << "query must never touch the status wait timer";
    EXPECT_EQ(fake->releases, 36);
    EXPECT_EQ(fake->waits, 36);
}

// Direct reads and writes submit on the bound IoCtx view.
TEST_F(radosEngineFixture, ReadsAndWritesSubmitOnBoundIoCtxView) {
    fakeRadosApi *fake = nullptr;
    auto engine = makeReady(&fake);
    ASSERT_EQ(fake->view_bind_order->size(), 2U);
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
    EXPECT_TRUE(fake->read_ctxs.empty());
    fake->completeSubmission(0);
    EXPECT_EQ(engine->checkXfer(write_handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->releaseReqH(write_handle), NIXL_SUCCESS);

    const one_pair read(0, 64, 2, 0x00);
    nixlBackendReqH *read_handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_READ, read.local, read.remote, "test-agent", read_handle),
              NIXL_IN_PROG);
    ASSERT_EQ(fake->read_ctxs.size(), 1U);
    EXPECT_EQ(fake->read_ctxs[0], view) << "reads must submit on the bound view";
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
        fakeRadosApi *fake = nullptr;
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
        EXPECT_EQ(fake->waits, 0);
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
        EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
    }
}

TEST_F(radosEngineFixture, FirstUnprovenPairSleepsOncePerCallAndDefersHarvest) {
    custom_params_ = validBase();
    custom_params_["status_wait_us"] = "777";
    auto tstate = std::make_shared<timer_state>();
    fakeRadosApi *fake = nullptr;
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
        return std::count(log->begin(), log->end(), std::string("is_complete_and_cb"));
    };

    // Pair 0 pending, pair 1 retired: one sleep per call, no rescan, no harvest.
    fake->completeSubmission(1, 64);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    EXPECT_EQ(tstate->budgets, (std::vector<uint64_t>{777}));
    EXPECT_EQ(poll_count(), 1);
    EXPECT_EQ(fake->rc_reads, 0);
    EXPECT_EQ(fake->releases, 0);

    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    EXPECT_EQ(tstate->budgets, (std::vector<uint64_t>{777, 777}));

    // The next poll after pair 0 completes harvests both in order, sleep-free.
    fake->completeSubmission(0, 64);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(tstate->budgets.size(), 2U);
    EXPECT_EQ(fake->rc_reads, 2);
    EXPECT_EQ(fake->releases, 2);
    EXPECT_EQ(fake->waits, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, WriteCompletionSleepsOnceAndHarvestsOnNextPoll) {
    custom_params_ = validBase();
    custom_params_["status_wait_us"] = "2400";
    auto tstate = std::make_shared<timer_state>();
    fakeRadosApi *fake = nullptr;
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
        return std::count(log->begin(), log->end(), std::string("is_complete_and_cb"));
    };

    EXPECT_EQ(engine->checkXfer(handle), NIXL_IN_PROG);
    EXPECT_EQ(tstate->budgets, (std::vector<uint64_t>{2400}));
    EXPECT_EQ(poll_count(), 1);
    EXPECT_EQ(fake->rc_reads, 0);
    EXPECT_EQ(fake->releases, 0);

    fake->completeSubmission(0);
    EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(tstate->budgets.size(), 1U);
    EXPECT_EQ(poll_count(), 2);
    EXPECT_EQ(fake->rc_reads, 1);
    EXPECT_EQ(fake->releases, 1);
    EXPECT_EQ(fake->waits, 0);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

TEST_F(radosEngineFixture, WaitNeverSleepsOutsideActivePairScans) {
    custom_params_ = validBase();
    custom_params_["status_wait_us"] = "3000";
    auto tstate = std::make_shared<timer_state>();
    fakeRadosApi *fake = nullptr;
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
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("wait_for_complete_and_cb")), 2);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("release")), 2);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("ioctx_close")), 2);
}

TEST_F(radosEngineFixture, MetadataBootstrapFailureUnwindsWithoutLeak) {
    struct entry {
        const char *name;
        int fakeRadosApi::*fail_field;
        std::vector<std::string> metadata_attempt;
    };

    // A metadata-client stage failure must leave the data client fully
    // bootstrapped, the engine init-errored, and every acquired metadata
    // resource unwound by the bootstrap RAII before the data teardown.
    const entry stages[] = {
        {"meta create2", &fakeRadosApi::create2_fail_ordinal, {"create2"}},
        {"meta conf_read_file",
         &fakeRadosApi::conf_read_file_fail_ordinal,
         {"create2", "conf_read_file", "shutdown"}},
        {"meta connect",
         &fakeRadosApi::connect_fail_ordinal,
         {"create2", "conf_read_file", "connect", "shutdown"}},
        {"meta ioctx_create",
         &fakeRadosApi::ioctx_create_fail_ordinal,
         {"create2", "conf_read_file", "connect", "ioctx_create", "shutdown"}},
    };

    for (const auto &stage : stages) {
        custom_params_ = validBase();
        auto failing = std::make_unique<fakeRadosApi>();
        log = failing->log;
        auto binds = failing->view_bind_order;
        (failing.get()->*stage.fail_field) = 1;
        {
            nixlRadosEngine engine(&init_params_, std::move(failing));
            EXPECT_TRUE(engine.getInitErr()) << stage.name;

            nixlBackendMD *md = nullptr;
            EXPECT_EQ(engine.registerMem(nixlBlobDesc(0, 1024, 1, "obj"), OBJ_SEG, md),
                      NIXL_ERR_BACKEND)
                << stage.name;

            nixl_reg_dlist_t descs(OBJ_SEG);
            descs.addDesc(nixlBlobDesc(0, 0, 1, "obj"));
            std::vector<nixl_query_resp_t> resp(1, nixl_query_resp_t{nixl_b_params_t{}});
            EXPECT_EQ(engine.queryMem(descs, resp), NIXL_ERR_BACKEND) << stage.name;
            EXPECT_FALSE(resp[0].has_value()) << stage.name;
            EXPECT_EQ(std::count(log->begin(), log->end(), std::string("aio_stat")), 0)
                << stage.name;
            EXPECT_EQ(binds->size(), 1U) << stage.name;
        }
        std::vector<std::string> expected{
            "create2", "conf_read_file", "connect", "ioctx_create", "ioctx_from_raw"};
        expected.insert(
            expected.end(), stage.metadata_attempt.begin(), stage.metadata_attempt.end());
        expected.insert(expected.end(), {"ioctx_close", "ioctx_destroy", "shutdown"});
        EXPECT_EQ(*log, expected) << stage.name;
    }
}

TEST_F(radosEngineFixture, QueryMemUsesMetadataClientAndTransfersUseDataClient) {
    custom_params_ = validBase();
    fakeRadosApi *fake = nullptr;
    std::shared_ptr<std::vector<librados::IoCtx *>> binds;
    std::shared_ptr<std::vector<librados::IoCtx *>> closes;
    {
        auto engine = makeEngine(&fake);
        ASSERT_FALSE(engine->getInitErr());
        ASSERT_EQ(fake->view_bind_order->size(), 2U);
        binds = fake->view_bind_order;
        closes = fake->view_close_order;
        const librados::IoCtx *data_view = (*binds)[0];
        const librados::IoCtx *meta_view = (*binds)[1];
        ASSERT_NE(data_view, meta_view);
        EXPECT_TRUE(fake->write_ctxs.empty());
        EXPECT_TRUE(fake->stat_ctxs.empty());
        EXPECT_TRUE(fake->read_ctxs.empty());

        nixlBackendMD *md = registerObject(*engine, 1, "iso-obj");
        std::vector<unsigned char> src(64, 0x77);
        nixl_meta_dlist_t local(DRAM_SEG);
        nixl_meta_dlist_t remote(OBJ_SEG);
        local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(src.data()), src.size(), 0));
        remote.addDesc(nixlMetaDesc(0, src.size(), 1));

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_SUCCESS);
        ASSERT_EQ(engine->postXfer(NIXL_WRITE, local, remote, "test-agent", handle), NIXL_IN_PROG);
        ASSERT_EQ(fake->write_ctxs.size(), 1U);
        EXPECT_EQ(fake->write_ctxs[0], data_view) << "writes must submit on the data view";
        EXPECT_TRUE(fake->stat_ctxs.empty()) << "transfers must not touch the metadata client";
        fake->completeSubmission(0, 0);
        EXPECT_EQ(engine->checkXfer(handle), NIXL_SUCCESS);
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);

        fake->stat_sizes["iso-obj"] = 64;
        nixl_reg_dlist_t descs(OBJ_SEG);
        descs.addDesc(nixlBlobDesc(0, 0, 1, "iso-obj"));
        std::vector<nixl_query_resp_t> resp;
        ASSERT_EQ(engine->queryMem(descs, resp), NIXL_SUCCESS);
        ASSERT_TRUE(resp[0].has_value());
        EXPECT_EQ(resp[0]->at("size"), "64");
        ASSERT_EQ(fake->stat_ctxs.size(), 1U);
        EXPECT_EQ(fake->stat_ctxs[0], meta_view) << "queryMem must submit on the metadata view";
        EXPECT_EQ(fake->write_ctxs[0], data_view) << "the write view must stay the data view";
        EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
    }
    ASSERT_EQ(closes->size(), 2U);
    EXPECT_EQ((*closes)[0], (*binds)[1]) << "the metadata view must retire first";
    EXPECT_EQ((*closes)[1], (*binds)[0]) << "the data view must retire last";
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("create2")), 2);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("connect")), 2);
    EXPECT_EQ(std::count(log->begin(), log->end(), std::string("shutdown")), 2);
}

} // namespace gtest::rados
