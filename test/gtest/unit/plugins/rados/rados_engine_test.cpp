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

#include <cerrno>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rados_backend.h"
#include "rados_capi.h"

namespace gtest::rados {

namespace {

    class fakeCApi final : public nixl::rados::cApi {
    public:
        fakeCApi() : log(std::make_shared<std::vector<std::string>>()) {}

        ~fakeCApi() override {
            EXPECT_TRUE(violations.empty()) << violations;
            for (const auto &rec : clusters) {
                EXPECT_FALSE(rec.live) << "cluster handle left open at fake teardown";
            }
            for (const auto &rec : ioctxs) {
                EXPECT_FALSE(rec.live) << "raw ioctx left open at fake teardown";
            }
        }

        std::shared_ptr<std::vector<std::string>> log;
        std::vector<std::pair<std::string, std::string>> options;
        std::string violations;
        int connect_rc = 0;
        std::vector<std::string> create2_names;
        std::vector<std::string> conf_paths;
        std::vector<std::string> pools;

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
            return requireLiveCluster(cluster) ? 0 : -EINVAL;
        }

        int
        conf_set(rados_t cluster, const char *option, const char *value) override {
            log->push_back(std::string("conf_set:") + option);
            options.emplace_back(option, value);
            return requireLiveCluster(cluster) ? 0 : -EINVAL;
        }

        int
        connect(rados_t cluster) override {
            log->push_back("connect");
            if (!requireLiveCluster(cluster)) {
                return -EINVAL;
            }
            return connect_rc;
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

    private:
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
    };

    nixl_b_params_t
    validBase() {
        return {{"conf", "fixture.conf"}, {"pool", "qa-pool"}, {"name", "client.qa"}};
    }

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

    // Representative successful construction: validBase parameters plus the client bootstrap.
    std::unique_ptr<nixlRadosEngine>
    makeReady(fakeCApi **retained_fake = nullptr) {
        custom_params_ = validBase();
        auto engine = makeEngine(retained_fake);
        EXPECT_FALSE(engine->getInitErr());
        return engine;
    }

    nixlBackendInitParams init_params_;
    nixl_b_params_t custom_params_;
};

// radosConfigTest owns the invalid-parameter matrix; one representative proves
// the engine rejects it before touching the librados seam.
TEST_F(radosEngineFixture, RejectsInvalidParamsBeforeAnyAPICall) {
    auto params = validBase();
    params["rbd_cache"] = "true";
    custom_params_ = params;
    auto engine = makeEngine();
    EXPECT_TRUE(engine->getInitErr());
    EXPECT_TRUE(log->empty()) << "rejection must not touch the librados seam";

    custom_params_ = validBase();
    nixlRadosEngine null_seam(&init_params_, nullptr);
    EXPECT_TRUE(null_seam.getInitErr());
}

TEST_F(radosEngineFixture, BootstrapsClientAndRetiresResources) {
    auto params = validBase();
    params["ceph.zeta"] = "3";
    params["ceph.alpha"] = "";
    custom_params_ = params;

    fakeCApi *fake = nullptr;
    {
        auto engine = makeEngine(&fake);
        ASSERT_FALSE(engine->getInitErr());
        EXPECT_EQ(*log,
                  (std::vector<std::string>{"create2",
                                           "conf_read_file",
                                           "conf_set:alpha",
                                           "conf_set:zeta",
                                           "connect",
                                           "ioctx_create"}));
        EXPECT_EQ(fake->create2_names, (std::vector<std::string>{"client.qa"}));
        EXPECT_EQ(fake->conf_paths, (std::vector<std::string>{"fixture.conf"}));
        EXPECT_EQ(fake->pools, (std::vector<std::string>{"qa-pool"}));
        EXPECT_EQ(fake->options,
                  (std::vector<std::pair<std::string, std::string>>{{"alpha", ""}, {"zeta", "3"}}));
    }
    EXPECT_EQ(*log,
              (std::vector<std::string>{"create2",
                                       "conf_read_file",
                                       "conf_set:alpha",
                                       "conf_set:zeta",
                                       "connect",
                                       "ioctx_create",
                                       "ioctx_destroy",
                                       "shutdown"}));
}

// A bootstrap failure unwinds exactly what was acquired, leaves the engine
// init-errored, and gates registration on both segments.
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

TEST_F(radosEngineFixture, TransferAndQuerySurfaceUnsupportedUntilFollowups) {
    fakeCApi *fake = nullptr;
    const auto engine = makeReady(&fake);
    nixlBackendMD *md = nullptr;
    ASSERT_EQ(engine->registerMem(nixlBlobDesc(0, 1024, 1, "obj"), OBJ_SEG, md), NIXL_SUCCESS);

    const nixl_meta_dlist_t local(DRAM_SEG);
    const nixl_meta_dlist_t remote(OBJ_SEG);
    nixlBackendReqH *handle = reinterpret_cast<nixlBackendReqH *>(static_cast<uintptr_t>(0x1));
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, "test-agent", handle),
              NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(handle, nullptr);
    nixlBackendReqH *null_handle = nullptr;
    EXPECT_EQ(engine->postXfer(NIXL_WRITE, local, remote, "test-agent", null_handle),
              NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(engine->checkXfer(nullptr), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(engine->releaseReqH(nullptr), NIXL_ERR_NOT_SUPPORTED);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 0, 1, "obj"));
    std::vector<nixl_query_resp_t> resp(1, nixl_query_resp_t{nixl_b_params_t{}});
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_TRUE(resp[0].has_value());

    EXPECT_EQ(engine->deregisterMem(md), NIXL_SUCCESS);
}

} // namespace gtest::rados
