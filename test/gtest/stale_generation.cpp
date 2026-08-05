/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
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

// Stale-generation handle validation: transfer-request handles and prepped
// descriptor lists bind to a specific remote registration generation. When a
// remote agent is invalidated and re-registers (e.g. after a disconnect), the
// old generation's handles must be rejected with NIXL_ERR_NOT_FOUND instead of
// feeding stale remote key metadata to the backend, and re-prepped/re-created
// handles against the new registration must work again.

#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "nixl.h"
#include "common.h"

namespace gtest {
namespace stale_generation {

    namespace {

        constexpr size_t BUFF_COUNT = 3;
        constexpr size_t BUFF_SIZE = 4096;

        struct MemBuffer {
            std::vector<std::byte> vec{BUFF_SIZE};

            nixlBasicDesc
            getBasicDesc() const {
                return nixlBasicDesc(reinterpret_cast<uintptr_t>(vec.data()), vec.size(), 0);
            }
        };

        struct AgentContext {
            const std::string name;
            const int port;
            std::unique_ptr<nixlAgent> agent;
            nixlBackendH *backend = nullptr;
            std::vector<MemBuffer> buffers;

            explicit AgentContext(std::string agent_name)
                : name(std::move(agent_name)),
                  port(PortAllocator::next_tcp_port()) {
                nixlAgentConfig cfg;
                cfg.useProgThread = true;
                cfg.useListenThread = true;
                cfg.listenPort = port;
                cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
                agent = std::make_unique<nixlAgent>(name, cfg);
            }

            void
            init() {
                ASSERT_EQ(agent->createBackend("UCX", {}, backend), NIXL_SUCCESS);
                ASSERT_NE(backend, nullptr);

                nixl_reg_dlist_t dlist(DRAM_SEG);
                buffers.resize(BUFF_COUNT);
                for (const auto &buffer : buffers) {
                    dlist.addDesc(nixlBlobDesc(buffer.getBasicDesc(), ""));
                }
                // Ignore EFA hardware mismatch warning on machines without GPUs
                const LogIgnoreGuard lig(
                    "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
                ASSERT_EQ(agent->registerMem(dlist), NIXL_SUCCESS);
            }

            nixl_xfer_dlist_t
            xferDlist() const {
                nixl_xfer_dlist_t dlist(DRAM_SEG);
                for (const auto &buffer : buffers) {
                    dlist.addDesc(buffer.getBasicDesc());
                }
                return dlist;
            }
        };

        class StaleGenerationTest : public testing::Test {
        protected:
            void
            SetUp() override {
                client_.init();
                server_.init();
                loadRemote();
            }

            void
            TearDown() override {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }

            void
            loadRemote() {
                nixl_blob_t md;
                ASSERT_EQ(server_.agent->getLocalMD(md), NIXL_SUCCESS);
                std::string loaded_name;
                ASSERT_EQ(client_.agent->loadRemoteMD(md, loaded_name), NIXL_SUCCESS);
                ASSERT_EQ(loaded_name, server_.name);
            }

            void
            invalidateAndReload() {
                ASSERT_EQ(client_.agent->invalidateRemoteMD(server_.name), NIXL_SUCCESS);
                loadRemote();
            }

            AgentContext client_{"stalegen_client"}, server_{"stalegen_server"};
        };

    } // namespace

    // A prepped remote dlist must be rejected by makeXferReq once the remote
    // registration it was prepared from is gone, and re-prepping must recover.
    TEST_F(StaleGenerationTest, PreppedDlistInvalidatedByReregistration) {
        nixlDlistH *local_side = nullptr, *remote_side = nullptr;
        ASSERT_EQ(client_.agent->prepXferDlist(client_.xferDlist(), local_side), NIXL_SUCCESS);
        ASSERT_EQ(client_.agent->prepXferDlist(server_.name, server_.xferDlist(), remote_side),
                  NIXL_SUCCESS);

        const std::vector<int> indices{0, 1, 2};
        const auto make_xfer = [&]() {
            nixlXferReqH *req = nullptr;
            const nixl_status_t status = client_.agent->makeXferReq(
                NIXL_WRITE, local_side, indices, remote_side, indices, req);
            if (req != nullptr) {
                EXPECT_EQ(client_.agent->releaseXferReq(req), NIXL_SUCCESS);
            }
            return status;
        };

        // Baseline: request creation succeeds against the live registration
        ASSERT_EQ(make_xfer(), NIXL_SUCCESS);

        invalidateAndReload();

        {
            const LogIgnoreGuard lig("was invalidated or re-registered after prepped xfer request "
                                     "creation");
            EXPECT_EQ(make_xfer(), NIXL_ERR_NOT_FOUND);
        }

        // Recovery: re-prep against the new registration works again
        ASSERT_EQ(client_.agent->prepXferDlist(server_.name, server_.xferDlist(), remote_side),
                  NIXL_SUCCESS);
        EXPECT_EQ(make_xfer(), NIXL_SUCCESS);

        EXPECT_EQ(client_.agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(client_.agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    // A transfer request must be rejected by postXferReq/estimateXferCost once the
    // remote registration generation it was created against is gone, while
    // releasing the stale handle and cached completion stay clean.
    TEST_F(StaleGenerationTest, XferHandleInvalidatedByReregistration) {
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(client_.agent->createXferReq(
                      NIXL_WRITE, client_.xferDlist(), server_.xferDlist(), server_.name, req),
                  NIXL_SUCCESS);

        invalidateAndReload();

        {
            const LogIgnoreGuard lig(
                "invalid request handle, remote agent was invalidated or re-registered");
            std::chrono::microseconds duration, err_margin;
            nixl_cost_t method;
            EXPECT_EQ(client_.agent->estimateXferCost(req, duration, err_margin, method),
                      NIXL_ERR_NOT_FOUND);
        }
        {
            const LogIgnoreGuard lig("was invalidated or re-registered after transfer request "
                                     "creation");
            EXPECT_EQ(client_.agent->postXferReq(req), NIXL_ERR_NOT_FOUND);
        }
        // Releasing a stale handle must keep working so callers can reclaim it
        EXPECT_EQ(client_.agent->releaseXferReq(req), NIXL_SUCCESS);

        // A newly created request against the new registration transfers fine
        ASSERT_EQ(client_.agent->createXferReq(
                      NIXL_WRITE, client_.xferDlist(), server_.xferDlist(), server_.name, req),
                  NIXL_SUCCESS);
        nixl_status_t status = client_.agent->postXferReq(req);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
            status = client_.agent->getXferStatus(req);
        }
        ASSERT_EQ(status, NIXL_SUCCESS);

        invalidateAndReload();
        // Cached completion is the request's own state and survives re-registration
        EXPECT_EQ(client_.agent->getXferStatus(req), NIXL_SUCCESS);
        EXPECT_EQ(client_.agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    // Reloading unchanged metadata keeps the same registration generation alive, so
    //  handles created against it remain valid (no invalidate in between).
    TEST_F(StaleGenerationTest, MetadataReloadKeepsHandlesValid) {
        nixlDlistH *local_side = nullptr, *remote_side = nullptr;
        ASSERT_EQ(client_.agent->prepXferDlist(client_.xferDlist(), local_side), NIXL_SUCCESS);
        ASSERT_EQ(client_.agent->prepXferDlist(server_.name, server_.xferDlist(), remote_side),
                  NIXL_SUCCESS);

        // Re-load the same metadata without invalidating: refresh path, same generation
        loadRemote();

        const std::vector<int> indices{0, 1, 2};
        nixlXferReqH *req = nullptr;
        EXPECT_EQ(
            client_.agent->makeXferReq(NIXL_WRITE, local_side, indices, remote_side, indices, req),
            NIXL_SUCCESS);
        if (req != nullptr) {
            nixl_status_t status = client_.agent->postXferReq(req);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
                status = client_.agent->getXferStatus(req);
            }
            EXPECT_EQ(status, NIXL_SUCCESS);
            EXPECT_EQ(client_.agent->releaseXferReq(req), NIXL_SUCCESS);
        }

        EXPECT_EQ(client_.agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(client_.agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

} // namespace stale_generation
} // namespace gtest
