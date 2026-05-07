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
#include <chrono>
#include <memory>
#include <regex>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "common.h"
#include "common/uuid_v7.h"
#include "nixl.h"
#include "serdes/serdes.h"

#if HAVE_ETCD
#include "backends/etcd_metadata_backend.h"
#endif

namespace gtest {
namespace metadata_format {

    namespace {

        // Builds a metadata blob with no UUID extension fields.
        std::string
        makeLegacyBlob(const std::string &agent_name) {
            nixlSerDes sd;
            sd.addStr("Agent", agent_name);
            size_t conn_cnt = 0;
            sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));
            sd.addStr("", "MemSection");
            size_t seg_count = 0;
            sd.addBuf("nixlSecElms", &seg_count, sizeof(seg_count));
            return sd.exportStr();
        }

        class AgentForBlobFormat : public ::testing::Test {
        protected:
            void
            SetUp() override {
                const auto port = PortAllocator::next_tcp_port();
                nixlAgentConfig cfg;
                cfg.useListenThread = true;
                cfg.listenPort = port;
                cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
                agent_ = std::make_unique<nixlAgent>(name_, cfg);

                nixlBackendH *backend_handle = nullptr;
                const LogIgnoreGuard lig_efa(
                    "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
                ASSERT_EQ(agent_->createBackend("UCX", {}, backend_handle), NIXL_SUCCESS);
                ASSERT_NE(backend_handle, nullptr);
            }

            void
            TearDown() override {
                // No `invalidateLocalMD(nullptr)` here: this fixture never publishes
                // metadata over a socket or ETCD, so there is nothing to invalidate
                // and the call would emit a spurious "ETCD is not supported and
                // socket information was not provided either" diagnostic in
                // environments without `NIXL_ETCD_ENDPOINTS`.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                agent_.reset();
            }

            std::unique_ptr<nixlAgent> agent_;
            const std::string name_ = "blob_format_agent";
        };

    } // namespace

    TEST_F(AgentForBlobFormat, GetLocalMdEmitsAgentNameAndUuidTags) {
        nixl_blob_t md;
        ASSERT_EQ(agent_->getLocalMD(md), NIXL_SUCCESS);

        EXPECT_NE(md.find("AgentName"), std::string::npos);
        EXPECT_NE(md.find("AgentUUID"), std::string::npos);

        nixlSerDes sd;
        ASSERT_EQ(sd.importStr(md), NIXL_SUCCESS);
        EXPECT_EQ(sd.getStr("Agent"), name_);
        size_t conn_cnt = 0;
        EXPECT_EQ(sd.getBuf("Conns", &conn_cnt, sizeof(conn_cnt)), NIXL_SUCCESS);
        for (size_t i = 0; i < conn_cnt; ++i) {
            EXPECT_FALSE(sd.getStr("t").empty());
            EXPECT_FALSE(sd.getStr("c").empty());
        }
        EXPECT_EQ(sd.getStr(""), "MemSection");
        size_t seg_count = 0;
        EXPECT_EQ(sd.getBuf("nixlSecElms", &seg_count, sizeof(seg_count)), NIXL_SUCCESS);
        EXPECT_EQ(sd.getStr("AgentName"), name_);
        const std::string uuid = sd.getStr("AgentUUID");
        EXPECT_FALSE(uuid.empty());
        EXPECT_EQ(uuid.size(), 36u);
    }

    TEST_F(AgentForBlobFormat, LoadRemoteMdHonoursLegacyAgentTag) {
        nixlAgentConfig cfg;
        cfg.useListenThread = true;
        cfg.listenPort = PortAllocator::next_tcp_port();
        cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
        auto receiver = std::make_unique<nixlAgent>("legacy_blob_receiver", cfg);

        nixlBackendH *backend_handle = nullptr;
        const LogIgnoreGuard lig_efa(
            "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
        ASSERT_EQ(receiver->createBackend("UCX", {}, backend_handle), NIXL_SUCCESS);

        const LogIgnoreGuard lig_mismatch(
            "Deserialization of tag AgentName failed for incomplete or missing header");

        const std::string legacy_blob = makeLegacyBlob("legacy_peer");
        std::string parsed_name;
        const nixl_status_t ret = receiver->loadRemoteMD(legacy_blob, parsed_name);
        EXPECT_EQ(ret, NIXL_SUCCESS);
        EXPECT_EQ(parsed_name, "legacy_peer");
        EXPECT_GE(lig_mismatch.getIgnoredCount(), 1u);
    }

    TEST(MetadataFormatUuid, UuidV7HasCorrectVersionAndVariant) {
        const std::string s = nixl::UUIDv7().toString();
        ASSERT_EQ(s.size(), 36u);

        // 8-4-4-4-12 with version `7` at position 14 and variant in {8,9,a,b}
        // at position 19.
        const std::regex shape{
            R"(^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$)"};
        EXPECT_TRUE(std::regex_match(s, shape)) << "got: " << s;
    }

    TEST(MetadataFormatUuid, UuidV7IsTimeOrderedAcrossDelay) {
        const std::string a = nixl::UUIDv7().toString();
        std::string b;
        for (int i = 0; i < 20; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            b = nixl::UUIDv7().toString();
            if (a.substr(0, 13) != b.substr(0, 13)) {
                break;
            }
        }

        // First 12 hex chars encode the 48-bit Unix-ms timestamp.
        EXPECT_LT(a.substr(0, 13), b.substr(0, 13));
    }

#if HAVE_ETCD

    TEST(MetadataFormatEtcdKey, LegacyKeyMatchesLegacyShape) {
        // Constructing the backend without `NIXL_ETCD_ENDPOINTS` set will fail
        // to connect, but `legacyKey` / `uuidBackedKey` are pure functions of
        // the configured namespace and don't need a live client. We construct
        // the backend in a try/catch and skip if the env-driven connect fails.
        const char *endpoints = std::getenv("NIXL_ETCD_ENDPOINTS");
        if (!endpoints || !*endpoints) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS not set; skipping ETCD key-shape test";
        }

        std::unique_ptr<nixlEtcdMetadataBackend> backend;
        try {
            backend = std::make_unique<nixlEtcdMetadataBackend>("agent_x",
                                                                std::chrono::microseconds(100000));
        }
        catch (const std::exception &e) {
            GTEST_SKIP() << "ETCD backend construction failed: " << e.what();
        }

        // Agent-name-key compatibility preserves a double slash if the configured
        // namespace ends with `/`.
        const std::string legacy = backend->legacyKey("agent_x", "metadata");
        EXPECT_NE(legacy.find("/agent_x/metadata"), std::string::npos);

        // UUID-backed: `{ns_no_trailing_slash}/agents/[{label}/]{uuid}/{dst}`.
        const std::string with_label =
            backend->uuidBackedKey("018f-uuid", "agent_dst", "first_partial");
        EXPECT_NE(with_label.find("/agents/first_partial/018f-uuid/agent_dst"), std::string::npos);

        const std::string no_label = backend->uuidBackedKey("018f-uuid", "NULL_AGENT", "");
        EXPECT_NE(no_label.find("/agents/018f-uuid/NULL_AGENT"), std::string::npos);
        EXPECT_EQ(no_label.find("//018f-uuid"), std::string::npos)
            << "uuid-backed shape must not have an accidental double-slash before the uuid";
    }

#endif // HAVE_ETCD

} // namespace metadata_format
} // namespace gtest
