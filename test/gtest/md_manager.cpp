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

#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common.h"
#include "nixl.h"
#include "nixl_md_manager.h"

namespace gtest::md_manager {

class MemBuffer {
public:
    explicit MemBuffer(size_t size) : vec_(size) {}

    operator uintptr_t() const {
        return reinterpret_cast<uintptr_t>(vec_.data());
    }

    nixlBasicDesc
    getBasicDesc() const {
        return nixlBasicDesc(static_cast<uintptr_t>(*this), vec_.size(), 0);
    }

    nixlBlobDesc
    getBlobDesc() const {
        return nixlBlobDesc(getBasicDesc(), "");
    }

private:
    std::vector<std::byte> vec_;
};

namespace {

    // RAII: snapshot a process env var, unset it, restore on scope exit.
    // Needed because ScopedEnv only supports the set-with-value flow.
    class ScopedUnsetEnv {
    public:
        explicit ScopedUnsetEnv(const char *name) : name_(name) {
            if (const char *p = ::getenv(name_)) {
                prev_ = std::string(p);
            }
            ::unsetenv(name_);
        }

        ~ScopedUnsetEnv() {
            if (prev_) {
                ::setenv(name_, prev_->c_str(), 1);
            } else {
                ::unsetenv(name_);
            }
        }

        ScopedUnsetEnv(const ScopedUnsetEnv &) = delete;
        ScopedUnsetEnv &
        operator=(const ScopedUnsetEnv &) = delete;

    private:
        const char *name_;
        std::optional<std::string> prev_;
    };

    // Bounded polling around checkRemoteMD: avoids fixed sleeps that make
    // async assertions slow and timing-sensitive. Returns the last observed
    // status (== `expected` on success, or the most recent value on timeout).
    nixl_status_t
    waitForRemoteMD(nixlMDManager *mdm,
                    const std::string &remote_name,
                    const nixl_xfer_dlist_t &descs,
                    nixl_status_t expected,
                    std::chrono::milliseconds timeout = std::chrono::seconds(3),
                    std::chrono::milliseconds interval = std::chrono::milliseconds(25)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        nixl_status_t last = mdm->checkRemoteMD(remote_name, descs);
        while (last != expected && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(interval);
            last = mdm->checkRemoteMD(remote_name, descs);
        }
        return last;
    }

} // namespace

// Env-var gate must default to off. Snapshot/restore the var so the test
// is independent of execution order (e.g. when gtest_shuffle is enabled).
TEST(MDManagerGate, DisabledWithoutEnvVar) {
    const ScopedUnsetEnv guard("NIXL_MD_MANAGER");

    nixlAgentConfig cfg;
    nixlAgent agent("gate_agent", cfg);

    nixlMDManager *mdm = reinterpret_cast<nixlMDManager *>(0xdeadbeef);
    ASSERT_EQ(agent.getMDManager(mdm), NIXL_ERR_NOT_SUPPORTED);
    ASSERT_EQ(mdm, nullptr);
}

class MDManagerFixture : public testing::Test {
protected:
    struct AgentContext {
        std::string name;
        std::string ip = "127.0.0.1";
        int port;
        nixlMDManager *mdm = nullptr;
        nixlBackendH *backend_handle = nullptr;
        std::vector<MemBuffer> buffers;
        std::unique_ptr<nixlAgent> agent;

        void
        createBackend() {
            ASSERT_EQ(agent->createBackend("UCX", {}, backend_handle), NIXL_SUCCESS);
            ASSERT_NE(backend_handle, nullptr);
        }

        void
        initAndRegisterBuffers(size_t count, size_t size) {
            for (size_t i = 0; i < count; i++) {
                buffers.emplace_back(size);
            }
            nixl_reg_dlist_t dlist(DRAM_SEG);
            for (const auto &buf : buffers) {
                dlist.addDesc(buf.getBlobDesc());
            }
            const LogIgnoreGuard lig_efa_warn(
                "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
            ASSERT_EQ(agent->registerMem(dlist), NIXL_SUCCESS);
        }
    };

    void
    SetUp() override {
        env_.addVar("NIXL_MD_MANAGER", "1");

        for (int i = 0; i < AGENT_COUNT_; i++) {
            AgentContext ctx;
            ctx.port = PortAllocator::next_tcp_port();
            ctx.name = "mdm_agent_" + std::to_string(i);

            nixlAgentConfig cfg;
            cfg.useListenThread = true;
            cfg.listenPort = ctx.port;
            cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
            ctx.agent = std::make_unique<nixlAgent>(ctx.name, cfg);

            ASSERT_EQ(ctx.agent->getMDManager(ctx.mdm), NIXL_SUCCESS);
            ASSERT_NE(ctx.mdm, nullptr);

            ctx.createBackend();
            ctx.initAndRegisterBuffers(BUFF_COUNT_, BUFF_SIZE_);

            agents_.push_back(std::move(ctx));
        }
    }

    void
    TearDown() override {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        agents_.clear();
        // env_ destructor restores prior NIXL_MD_MANAGER value.
    }

    static constexpr int AGENT_COUNT_ = 2;
    static constexpr size_t BUFF_COUNT_ = 4;
    static constexpr size_t BUFF_SIZE_ = 1024;

    // Force the P2P backend regardless of ambient env: if NIXL_ETCD_ENDPOINTS
    // is set, the manager would build the ETCD backend and break this
    // fixture's P2P assumptions (e.g. BackendIsP2P).
    ScopedUnsetEnv no_etcd_{"NIXL_ETCD_ENDPOINTS"};
    ScopedEnv env_;
    std::vector<AgentContext> agents_;
};

TEST_F(MDManagerFixture, GetMDManagerIsIdempotent) {
    auto &a = agents_[0];
    nixlMDManager *first = nullptr;
    nixlMDManager *second = nullptr;
    ASSERT_EQ(a.agent->getMDManager(first), NIXL_SUCCESS);
    ASSERT_EQ(a.agent->getMDManager(second), NIXL_SUCCESS);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first, second);
    EXPECT_EQ(first, a.mdm);
}

TEST_F(MDManagerFixture, RegisterRejectsEmptyInputs) {
    auto &a = agents_[0];
    EXPECT_EQ(a.mdm->registerMDPeer("", "127.0.0.1", 1234), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(a.mdm->registerMDPeer("peer", "", 1234), NIXL_ERR_INVALID_PARAM);
}

TEST_F(MDManagerFixture, RegisterRejectsMalformedIp) {
    auto &a = agents_[0];
    EXPECT_EQ(a.mdm->registerMDPeer("peer", "not-an-ip", 1234), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(a.mdm->registerMDPeer("peer", "256.0.0.1", 1234), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(a.mdm->registerMDPeer("peer", "127.0.0.1.5", 1234), NIXL_ERR_INVALID_PARAM);
}

TEST_F(MDManagerFixture, UnregisterUnknownPeerIsOk) {
    auto &a = agents_[0];
    EXPECT_EQ(a.mdm->unregisterMDPeer("never_registered"), NIXL_SUCCESS);
}

TEST_F(MDManagerFixture, ReRegisterSameAddressIsIdempotent) {
    auto &a = agents_[0];
    EXPECT_EQ(a.mdm->registerMDPeer("peer", "127.0.0.1", 1234), NIXL_SUCCESS);
    EXPECT_EQ(a.mdm->registerMDPeer("peer", "127.0.0.1", 1234), NIXL_SUCCESS);
}

TEST_F(MDManagerFixture, ReRegisterDifferentAddressIsRejected) {
    auto &a = agents_[0];
    EXPECT_EQ(a.mdm->registerMDPeer("peer", "127.0.0.1", 1234), NIXL_SUCCESS);
    EXPECT_EQ(a.mdm->registerMDPeer("peer", "127.0.0.1", 5678), NIXL_ERR_NOT_ALLOWED);
    EXPECT_EQ(a.mdm->registerMDPeer("peer", "10.0.0.1", 1234), NIXL_ERR_NOT_ALLOWED);
}

TEST_F(MDManagerFixture, NetworkOpsRequireRegistration) {
    auto &a = agents_[0];
    EXPECT_EQ(a.mdm->sendLocalMD("unregistered_peer"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(a.mdm->fetchRemoteMD("unregistered_peer"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(a.mdm->invalidateLocalMD("unregistered_peer"), NIXL_ERR_NOT_FOUND);

    nixl_reg_dlist_t descs(DRAM_SEG);
    EXPECT_EQ(a.mdm->sendLocalPartialMD("unregistered_peer", descs), NIXL_ERR_NOT_FOUND);
}

TEST_F(MDManagerFixture, SendLocalAndInvalidateLocal) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(src.mdm->registerMDPeer(dst.name, dst.ip, static_cast<uint16_t>(dst.port)),
              NIXL_SUCCESS);

    ASSERT_EQ(src.mdm->sendLocalMD(dst.name), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.mdm, src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);

    ASSERT_EQ(src.mdm->invalidateLocalMD(dst.name), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.mdm, src.name, {DRAM_SEG}, NIXL_ERR_NOT_FOUND),
              NIXL_ERR_NOT_FOUND);
}

TEST_F(MDManagerFixture, FetchRemote) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(dst.mdm->registerMDPeer(src.name, src.ip, static_cast<uint16_t>(src.port)),
              NIXL_SUCCESS);

    ASSERT_EQ(dst.mdm->fetchRemoteMD(src.name), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.mdm, src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);
}

TEST_F(MDManagerFixture, UnregisterTriggersInvalidate) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(src.mdm->registerMDPeer(dst.name, dst.ip, static_cast<uint16_t>(dst.port)),
              NIXL_SUCCESS);

    ASSERT_EQ(src.mdm->sendLocalMD(dst.name), NIXL_SUCCESS);
    ASSERT_EQ(waitForRemoteMD(dst.mdm, src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);

    ASSERT_EQ(src.mdm->unregisterMDPeer(dst.name), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.mdm, src.name, {DRAM_SEG}, NIXL_ERR_NOT_FOUND),
              NIXL_ERR_NOT_FOUND);

    // After unregister, the registry entry is gone; subsequent sends to
    // the same name should fail until the peer is re-registered.
    EXPECT_EQ(src.mdm->sendLocalMD(dst.name), NIXL_ERR_NOT_FOUND);
}

TEST_F(MDManagerFixture, BackendIsP2P) {
    auto &a = agents_[0];
    EXPECT_EQ(a.mdm->getBackend(), std::string_view("P2P"));
}

// ETCD-backed manager. Skipped unless NIXL_ETCD_ENDPOINTS is set (as in CI);
// no listen thread is needed since the store is the rendezvous point.
class MDManagerEtcdFixture : public testing::Test {
protected:
    struct AgentContext {
        std::string name;
        nixlMDManager *mdm = nullptr;
        nixlBackendH *backend_handle = nullptr;
        std::vector<MemBuffer> buffers;
        std::unique_ptr<nixlAgent> agent;

        void
        createBackend() {
            ASSERT_EQ(agent->createBackend("UCX", {}, backend_handle), NIXL_SUCCESS);
            ASSERT_NE(backend_handle, nullptr);
        }

        void
        initAndRegisterBuffers(size_t count, size_t size) {
            for (size_t i = 0; i < count; i++) {
                buffers.emplace_back(size);
            }
            nixl_reg_dlist_t dlist(DRAM_SEG);
            for (const auto &buf : buffers) {
                dlist.addDesc(buf.getBlobDesc());
            }
            const LogIgnoreGuard lig_efa_warn(
                "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
            ASSERT_EQ(agent->registerMem(dlist), NIXL_SUCCESS);
        }
    };

    void
    SetUp() override {
        if (::getenv("NIXL_ETCD_ENDPOINTS") == nullptr) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS not set; skipping ETCD backend tests";
        }
        env_.addVar("NIXL_MD_MANAGER", "1");

        // Unique per-fixture-instance names so leftover keys from one test do
        // not bleed into the next against a shared store.
        static int run = 0;
        const std::string suffix = "_" + std::to_string(run++);

        for (int i = 0; i < AGENT_COUNT_; i++) {
            AgentContext ctx;
            ctx.name = "mdm_etcd_agent_" + std::to_string(i) + suffix;

            nixlAgentConfig cfg;
            cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
            ctx.agent = std::make_unique<nixlAgent>(ctx.name, cfg);

            ASSERT_EQ(ctx.agent->getMDManager(ctx.mdm), NIXL_SUCCESS);
            ASSERT_NE(ctx.mdm, nullptr);

            ctx.createBackend();
            ctx.initAndRegisterBuffers(BUFF_COUNT_, BUFF_SIZE_);

            agents_.push_back(std::move(ctx));
        }
    }

    void
    TearDown() override {
        agents_.clear();
    }

    static constexpr int AGENT_COUNT_ = 2;
    static constexpr size_t BUFF_COUNT_ = 4;
    static constexpr size_t BUFF_SIZE_ = 1024;

    ScopedEnv env_;
    std::vector<AgentContext> agents_;
};

TEST_F(MDManagerEtcdFixture, BackendIsEtcd) {
    EXPECT_EQ(agents_[0].mdm->getBackend(), std::string_view("ETCD"));
}

TEST_F(MDManagerEtcdFixture, RegisterIgnoresAddress) {
    // Centralized backend accepts registration without an address.
    EXPECT_EQ(agents_[0].mdm->registerMDPeer("peer_no_addr", "", 0), NIXL_SUCCESS);
}

TEST_F(MDManagerEtcdFixture, SendAndFetchRoundTrip) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(src.mdm->registerMDPeer(dst.name, "", 0), NIXL_SUCCESS);
    ASSERT_EQ(dst.mdm->registerMDPeer(src.name, "", 0), NIXL_SUCCESS);

    ASSERT_EQ(src.mdm->sendLocalMD(dst.name), NIXL_SUCCESS);
    // ETCD fetch is synchronous (get, then watch-on-miss), so the metadata is
    // loaded by the time fetchRemoteMD returns.
    ASSERT_EQ(dst.mdm->fetchRemoteMD(src.name), NIXL_SUCCESS);
    EXPECT_EQ(dst.mdm->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);
}

TEST_F(MDManagerEtcdFixture, PartialRequiresLabel) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(src.mdm->registerMDPeer(dst.name, "", 0), NIXL_SUCCESS);

    nixl_reg_dlist_t descs(DRAM_SEG);
    for (const auto &buf : src.buffers) {
        descs.addDesc(buf.getBlobDesc());
    }

    // No label is rejected on the centralized backend.
    EXPECT_EQ(src.mdm->sendLocalPartialMD(dst.name, descs), NIXL_ERR_INVALID_PARAM);

    nixl_opt_args_t params;
    params.metadataLabel = "memView1";
    EXPECT_EQ(src.mdm->sendLocalPartialMD(dst.name, descs, &params), NIXL_SUCCESS);
}

TEST_F(MDManagerEtcdFixture, WatchDrivenInvalidation) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(src.mdm->registerMDPeer(dst.name, "", 0), NIXL_SUCCESS);
    ASSERT_EQ(dst.mdm->registerMDPeer(src.name, "", 0), NIXL_SUCCESS);

    ASSERT_EQ(src.mdm->sendLocalMD(dst.name), NIXL_SUCCESS);
    ASSERT_EQ(dst.mdm->fetchRemoteMD(src.name), NIXL_SUCCESS);
    ASSERT_EQ(dst.mdm->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);

    // Removing src's published subtree fires a DELETE watch on dst, which
    // drains into invalidateRemoteMD on the next check.
    ASSERT_EQ(src.mdm->invalidateLocalMD(dst.name), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.mdm, src.name, {DRAM_SEG}, NIXL_ERR_NOT_FOUND),
              NIXL_ERR_NOT_FOUND);
}

} // namespace gtest::md_manager
