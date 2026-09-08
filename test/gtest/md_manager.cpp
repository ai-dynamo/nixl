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
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "nixl.h"

// Exercises the agent-owned metadata manager: with NIXL_USE_MD_MANAGER set, the
// agent's public metadata methods route through nixlMDManager + a backend (P2P
// when an address is given, else the KV/ETCD backend) instead of the inline
// path. The observable behavior must match the inline path. The ETCD cases are
// gated on NIXL_ETCD_ENDPOINTS and skip when no live store is configured.
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

    nixl_opt_args_t
    peerArgs(const std::string &ip, int port) {
        nixl_opt_args_t args;
        args.ipAddr = ip;
        args.port = port;
        return args;
    }

    // An empty value means "not configured": getenv is non-null for VAR="", and
    // an empty endpoint would fail the fixture instead of skipping it.
    bool
    envConfigured(const char *name) {
        const char *value = std::getenv(name);
        return value != nullptr && *value != '\0';
    }

    // Bounded polling around checkRemoteMD: avoids fixed sleeps that make
    // async assertions slow and timing-sensitive.
    nixl_status_t
    waitForRemoteMD(nixlAgent *agent,
                    const std::string &remote_name,
                    const nixl_xfer_dlist_t &descs,
                    nixl_status_t expected,
                    std::chrono::milliseconds timeout = std::chrono::seconds(3),
                    std::chrono::milliseconds interval = std::chrono::milliseconds(25)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        nixl_status_t last = agent->checkRemoteMD(remote_name, descs);
        while (last != expected && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(interval);
            last = agent->checkRemoteMD(remote_name, descs);
        }
        return last;
    }

} // namespace

class MDManagerFixture : public testing::Test {
protected:
    struct AgentContext {
        std::string name;
        std::string ip = "127.0.0.1";
        int port;
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
        // Route the agent's P2P metadata methods through the manager. Scoped so
        // it is restored even when a fatal assertion below skips TearDown.
        env_.addVar("NIXL_USE_MD_MANAGER", "1");

        for (int i = 0; i < AGENT_COUNT_; i++) {
            AgentContext ctx;
            ctx.port = PortAllocator::next_tcp_port();
            ctx.name = "mdm_agent_" + std::to_string(i);

            nixlAgentConfig cfg;
            cfg.useListenThread = true;
            cfg.listenPort = ctx.port;
            cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
            ctx.agent = std::make_unique<nixlAgent>(ctx.name, cfg);

            ctx.createBackend();
            ctx.initAndRegisterBuffers(BUFF_COUNT_, BUFF_SIZE_);

            agents_.push_back(std::move(ctx));
        }
    }

    void
    TearDown() override {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        agents_.clear();
    }

    static constexpr int AGENT_COUNT_ = 2;
    static constexpr size_t BUFF_COUNT_ = 4;
    static constexpr size_t BUFF_SIZE_ = 1024;

    ScopedEnv env_;
    std::vector<AgentContext> agents_;
};

TEST_F(MDManagerFixture, SendLocalAndInvalidateLocal) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    const nixl_opt_args_t args = peerArgs(dst.ip, dst.port);

    ASSERT_EQ(src.agent->sendLocalMD(&args), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.agent.get(), src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);

    ASSERT_EQ(src.agent->invalidateLocalMD(&args), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.agent.get(), src.name, {DRAM_SEG}, NIXL_ERR_NOT_FOUND),
              NIXL_ERR_NOT_FOUND);
}

// An agent that does not listen runs no metadata thread, so the P2P backend has
// nothing to run the send on and does it inline. Previously the task was queued
// against a worker that never started and the peer was never contacted.
TEST_F(MDManagerFixture, SendLocalWithoutWorker) {
    auto &dst = agents_[0];

    AgentContext src;
    src.name = "mdm_agent_no_worker";
    nixlAgentConfig cfg;
    // Explicit, since it is the precondition under test: no listener, no thread.
    cfg.useListenThread = false;
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
    src.agent = std::make_unique<nixlAgent>(src.name, cfg);
    src.createBackend();
    src.initAndRegisterBuffers(BUFF_COUNT_, BUFF_SIZE_);

    const nixl_opt_args_t args = peerArgs(dst.ip, dst.port);
    ASSERT_EQ(src.agent->sendLocalMD(&args), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.agent.get(), src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);
}

TEST_F(MDManagerFixture, FetchRemote) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    const nixl_opt_args_t args = peerArgs(src.ip, src.port);

    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, &args), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.agent.get(), src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);
}

// ETCD (KV) backend: a no-address metadata call routes through the manager's KV
// backend, which reuses the same ETCD comm-thread work items as the inline path.
// Gated on a live store via NIXL_ETCD_ENDPOINTS.
class MDManagerEtcdFixture : public testing::Test {
protected:
    struct AgentContext {
        std::string name;
        nixlBackendH *backend_handle = nullptr;
        std::vector<MemBuffer> buffers;
        std::unique_ptr<nixlAgent> agent;
    };

    void
    SetUp() override {
        if (!envConfigured("NIXL_ETCD_ENDPOINTS")) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS not set; skipping ETCD backend tests";
        }
        // No-address metadata routes through the manager (to the KV backend).
        env_.addVar("NIXL_USE_MD_MANAGER", "1");

        // Unique per-run names so stale ETCD keys from earlier runs cannot leak in.
        const std::string suffix =
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        for (int i = 0; i < AGENT_COUNT_; i++) {
            AgentContext ctx;
            ctx.name = "mdm_etcd_agent_" + std::to_string(i) + "_" + suffix;

            // No listen thread: the ETCD path drives the comm thread on its own.
            nixlAgentConfig cfg;
            cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
            ctx.agent = std::make_unique<nixlAgent>(ctx.name, cfg);

            ASSERT_EQ(ctx.agent->createBackend("UCX", {}, ctx.backend_handle), NIXL_SUCCESS);
            ASSERT_NE(ctx.backend_handle, nullptr);

            for (size_t b = 0; b < BUFF_COUNT_; b++) {
                ctx.buffers.emplace_back(BUFF_SIZE_);
            }
            nixl_reg_dlist_t dlist(DRAM_SEG);
            for (const auto &buf : ctx.buffers) {
                dlist.addDesc(buf.getBlobDesc());
            }
            const LogIgnoreGuard lig_efa_warn(
                "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
            ASSERT_EQ(ctx.agent->registerMem(dlist), NIXL_SUCCESS);

            agents_.push_back(std::move(ctx));
        }
    }

    void
    TearDown() override {
        // Drop each agent's published metadata so the shared store does not
        // accumulate orphaned keys across CI runs (names are unique per run).
        for (auto &ctx : agents_) {
            ctx.agent->invalidateLocalMD(nullptr);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        agents_.clear();
    }

    static constexpr int AGENT_COUNT_ = 2;
    static constexpr size_t BUFF_COUNT_ = 4;
    static constexpr size_t BUFF_SIZE_ = 1024;

    ScopedEnv env_;
    std::vector<AgentContext> agents_;
};

TEST_F(MDManagerEtcdFixture, SendAndFetchByName) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(src.agent->sendLocalMD(nullptr), NIXL_SUCCESS);
    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, nullptr), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.agent.get(), src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);
}

TEST_F(MDManagerEtcdFixture, InvalidateLocalRemovesRemote) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(src.agent->sendLocalMD(nullptr), NIXL_SUCCESS);
    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, nullptr), NIXL_SUCCESS);
    ASSERT_EQ(waitForRemoteMD(dst.agent.get(), src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);

    ASSERT_EQ(src.agent->invalidateLocalMD(nullptr), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.agent.get(), src.name, {DRAM_SEG}, NIXL_ERR_NOT_FOUND),
              NIXL_ERR_NOT_FOUND);
}

// TCPStore (KV) backend: a no-address metadata call routes through the manager's
// TCPStore backend, which does its store I/O over the c10d wire protocol
// (no libtorch). Gated on a live store via NIXL_TCPSTORE_ENDPOINT (host:port),
// e.g. a torch.distributed.TCPStore master.
class MDManagerTcpStoreFixture : public testing::Test {
protected:
    struct AgentContext {
        std::string name;
        nixlBackendH *backend_handle = nullptr;
        std::vector<MemBuffer> buffers;
        std::unique_ptr<nixlAgent> agent;
    };

    void
    SetUp() override {
        if (!envConfigured("NIXL_TCPSTORE_ENDPOINT")) {
            GTEST_SKIP() << "NIXL_TCPSTORE_ENDPOINT not set; skipping TCPStore backend tests";
        }
        // ETCD and TCPStore are mutually exclusive; with both set the manager
        // selects ETCD and the agent ctor throws, so skip rather than exercise
        // the wrong backend under a "TCPStore" name.
        if (envConfigured("NIXL_ETCD_ENDPOINTS")) {
            GTEST_SKIP()
                << "NIXL_ETCD_ENDPOINTS also set; skipping mutually-exclusive TCPStore tests";
        }
        // No-address metadata routes through the manager (to the KV backend).
        env_.addVar("NIXL_USE_MD_MANAGER", "1");

        // Unique per-run names so stale keys from earlier runs cannot leak in.
        const std::string suffix =
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        for (int i = 0; i < AGENT_COUNT_; i++) {
            AgentContext ctx;
            ctx.name = "mdm_tcpstore_agent_" + std::to_string(i) + "_" + suffix;

            // TCPStore is manager-only and synchronous; no comm/listen thread.
            nixlAgentConfig cfg;
            cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
            ctx.agent = std::make_unique<nixlAgent>(ctx.name, cfg);

            ASSERT_EQ(ctx.agent->createBackend("UCX", {}, ctx.backend_handle), NIXL_SUCCESS);
            ASSERT_NE(ctx.backend_handle, nullptr);

            for (size_t b = 0; b < BUFF_COUNT_; b++) {
                ctx.buffers.emplace_back(BUFF_SIZE_);
            }
            nixl_reg_dlist_t dlist(DRAM_SEG);
            for (const auto &buf : ctx.buffers) {
                dlist.addDesc(buf.getBlobDesc());
            }
            const LogIgnoreGuard lig_efa_warn(
                "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
            ASSERT_EQ(ctx.agent->registerMem(dlist), NIXL_SUCCESS);

            agents_.push_back(std::move(ctx));
        }
    }

    void
    TearDown() override {
        // Drop each agent's published metadata so the shared store does not
        // accumulate orphaned keys across CI runs (names are unique per run).
        for (auto &ctx : agents_) {
            ctx.agent->invalidateLocalMD(nullptr);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        agents_.clear();
    }

    static constexpr int AGENT_COUNT_ = 2;
    static constexpr size_t BUFF_COUNT_ = 4;
    static constexpr size_t BUFF_SIZE_ = 1024;

    ScopedEnv env_;
    std::vector<AgentContext> agents_;
};

TEST_F(MDManagerTcpStoreFixture, SendAndFetchByName) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(src.agent->sendLocalMD(nullptr), NIXL_SUCCESS);
    // The fetch is asynchronous and stays pending until the key appears, so no
    // retry loop is needed around it; readiness is observed via checkRemoteMD.
    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, nullptr), NIXL_SUCCESS);
    EXPECT_EQ(waitForRemoteMD(dst.agent.get(), src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);
}

TEST_F(MDManagerTcpStoreFixture, InvalidateLocalRemovesRemote) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    ASSERT_EQ(src.agent->sendLocalMD(nullptr), NIXL_SUCCESS);
    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, nullptr), NIXL_SUCCESS);
    ASSERT_EQ(waitForRemoteMD(dst.agent.get(), src.name, {DRAM_SEG}, NIXL_SUCCESS), NIXL_SUCCESS);

    // invalidateLocal removes the key from the store. TCPStore has no watch, so
    // the peer drops its cached copy itself; the re-fetch then finds nothing and
    // the entry stays absent.
    ASSERT_EQ(src.agent->invalidateLocalMD(nullptr), NIXL_SUCCESS);
    ASSERT_EQ(dst.agent->invalidateRemoteMD(src.name), NIXL_SUCCESS);
    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, nullptr), NIXL_SUCCESS);
    EXPECT_EQ(
        waitForRemoteMD(
            dst.agent.get(), src.name, {DRAM_SEG}, NIXL_SUCCESS, std::chrono::milliseconds(500)),
        NIXL_ERR_NOT_FOUND);
}

} // namespace gtest::md_manager
