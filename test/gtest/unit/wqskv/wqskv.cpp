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

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "backend/backend_aux.h"
#include "kv_interface.h"
#include "nixl_descriptors.h"
#include "nixl_types.h"
#include "wqskv_helpers.h"

namespace gtest::wqskv {

// ----- parseCustomParamKeys -----

TEST(WqskvParseCustomParamKeys, EmptyBlobReturnsFalse) {
    std::vector<std::string> out;
    EXPECT_FALSE(::wqskv::parseCustomParamKeys("", 1, out));
    EXPECT_TRUE(out.empty());
}

TEST(WqskvParseCustomParamKeys, NonPositiveExpectedCountReturnsFalse) {
    std::vector<std::string> out;
    EXPECT_FALSE(::wqskv::parseCustomParamKeys("key1", 0, out));
    EXPECT_FALSE(::wqskv::parseCustomParamKeys("key1", -1, out));
}

TEST(WqskvParseCustomParamKeys, SingleKey) {
    std::vector<std::string> out;
    ASSERT_TRUE(::wqskv::parseCustomParamKeys("key1", 1, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], "key1");
}

TEST(WqskvParseCustomParamKeys, MultipleKeys) {
    std::vector<std::string> out;
    ASSERT_TRUE(::wqskv::parseCustomParamKeys("a\nb\nc", 3, out));
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], "a");
    EXPECT_EQ(out[1], "b");
    EXPECT_EQ(out[2], "c");
}

TEST(WqskvParseCustomParamKeys, CountMismatchClearsOutput) {
    std::vector<std::string> out;
    EXPECT_FALSE(::wqskv::parseCustomParamKeys("a\nb", 3, out));
    EXPECT_TRUE(out.empty());

    EXPECT_FALSE(::wqskv::parseCustomParamKeys("a\nb\nc", 2, out));
    EXPECT_TRUE(out.empty());
}

TEST(WqskvParseCustomParamKeys, TrailingNewlineAddsEmptyKey) {
    // "a\n" -> ["a", ""] -> count 2 expected.
    std::vector<std::string> out;
    ASSERT_TRUE(::wqskv::parseCustomParamKeys("a\n", 2, out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], "a");
    EXPECT_EQ(out[1], "");
}

// ----- resolveConfigPath -----

class WqskvResolveConfigPath : public ::testing::Test {
protected:
    void
    SetUp() override {
        unsetenv("WDS_BACKEND_CONFIG_PATH");
    }

    void
    TearDown() override {
        unsetenv("WDS_BACKEND_CONFIG_PATH");
    }
};

TEST_F(WqskvResolveConfigPath, ReturnsCustomParamWhenSet) {
    nixl_b_params_t params{{"config_path", "/cfg/from-custom.json"}};
    nixlBackendInitParams init{};
    init.customParams = &params;

    EXPECT_EQ(::wqskv::resolveConfigPath(&init), "/cfg/from-custom.json");
}

TEST_F(WqskvResolveConfigPath, FallsBackToEnvVar) {
    setenv("WDS_BACKEND_CONFIG_PATH", "/cfg/from-env.json", 1);
    nixlBackendInitParams init{};
    init.customParams = nullptr;

    EXPECT_EQ(::wqskv::resolveConfigPath(&init), "/cfg/from-env.json");
}

TEST_F(WqskvResolveConfigPath, ReturnsEmptyWhenNeitherSet) {
    nixlBackendInitParams init{};
    init.customParams = nullptr;

    EXPECT_EQ(::wqskv::resolveConfigPath(&init), "");
}

TEST_F(WqskvResolveConfigPath, CustomParamTakesPriorityOverEnv) {
    setenv("WDS_BACKEND_CONFIG_PATH", "/cfg/from-env.json", 1);
    nixl_b_params_t params{{"config_path", "/cfg/from-custom.json"}};
    nixlBackendInitParams init{};
    init.customParams = &params;

    EXPECT_EQ(::wqskv::resolveConfigPath(&init), "/cfg/from-custom.json");
}

TEST_F(WqskvResolveConfigPath, EmptyCustomParamFallsThroughToEnv) {
    setenv("WDS_BACKEND_CONFIG_PATH", "/cfg/from-env.json", 1);
    nixl_b_params_t params{{"config_path", ""}};
    nixlBackendInitParams init{};
    init.customParams = &params;

    EXPECT_EQ(::wqskv::resolveConfigPath(&init), "/cfg/from-env.json");
}

// ----- isValidPrepXferParams -----

TEST(WqskvIsValidPrepXferParams, AcceptsWriteWithDramDram) {
    nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
    EXPECT_TRUE(::wqskv::isValidPrepXferParams(NIXL_WRITE, local, remote));
}

TEST(WqskvIsValidPrepXferParams, AcceptsReadWithDramDram) {
    nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
    EXPECT_TRUE(::wqskv::isValidPrepXferParams(NIXL_READ, local, remote));
}

TEST(WqskvIsValidPrepXferParams, RejectsUnknownOp) {
    nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
    const auto bad_op = static_cast<nixl_xfer_op_t>(42);
    EXPECT_FALSE(::wqskv::isValidPrepXferParams(bad_op, local, remote));
}

TEST(WqskvIsValidPrepXferParams, RejectsLocalNonDram) {
    nixl_meta_dlist_t local(VRAM_SEG), remote(DRAM_SEG);
    EXPECT_FALSE(::wqskv::isValidPrepXferParams(NIXL_WRITE, local, remote));
}

TEST(WqskvIsValidPrepXferParams, RejectsRemoteNonDram) {
    nixl_meta_dlist_t local(DRAM_SEG), remote(FILE_SEG);
    EXPECT_FALSE(::wqskv::isValidPrepXferParams(NIXL_READ, local, remote));
}

// ----- loadKVCacheOptionsFromJson -----

class WqskvLoadJson : public ::testing::Test {
protected:
    std::string
    writeTempJson(const std::string &content) {
        char tmpl[] = "/tmp/wqskv_test_json_XXXXXX";
        int fd = ::mkstemp(tmpl);
        if (fd < 0) {
            ADD_FAILURE() << "mkstemp failed";
            return {};
        }
        std::string path(tmpl);
        ::close(fd);
        std::ofstream out(path, std::ofstream::binary);
        out << content;
        created_.push_back(path);
        return path;
    }

    void
    TearDown() override {
        for (const auto &p : created_) {
            std::remove(p.c_str());
        }
    }

    std::vector<std::string> created_;
};

TEST_F(WqskvLoadJson, ParsesValidConfig) {
    const std::string content = R"({
        "poolid": 1,
        "thread_num": 2,
        "thread_mode": "poll",
        "wengine_conf": "/etc/wengine.conf",
        "node_id": 11,
        "bvar_port": 7654,
        "bind_cpus": "20,21",
        "mem_size": 16384
    })";
    const auto path = writeTempJson(content);
    ASSERT_FALSE(path.empty());

    KVCacheOptions opts;
    ASSERT_TRUE(::wqskv::loadKVCacheOptionsFromJson(path, opts));
    EXPECT_EQ(opts.poolid, 1u);
    EXPECT_EQ(opts.thread_num, 2);
    EXPECT_EQ(opts.thread_mode, KVCACHE_THREAD_POLL);
    EXPECT_EQ(opts.conf, "/etc/wengine.conf");
    EXPECT_EQ(opts.nid, 11u);
    EXPECT_EQ(opts.bvar_dummy_port, 7654u);
    EXPECT_EQ(opts.mem_size, 16384u);
    ASSERT_EQ(opts.bind_cpus.size(), 2u);
    EXPECT_EQ(opts.bind_cpus[0], 20);
    EXPECT_EQ(opts.bind_cpus[1], 21);
}

TEST_F(WqskvLoadJson, AcceptsEventThreadMode) {
    const std::string content = R"({
        "poolid": 0, "thread_num": 1, "thread_mode": "event",
        "wengine_conf": "/x", "node_id": 0, "bvar_port": 0,
        "bind_cpus": "0", "mem_size": 1
    })";
    const auto path = writeTempJson(content);
    ASSERT_FALSE(path.empty());

    KVCacheOptions opts;
    ASSERT_TRUE(::wqskv::loadKVCacheOptionsFromJson(path, opts));
    EXPECT_EQ(opts.thread_mode, KVCACHE_THREAD_EVENT);
}

TEST_F(WqskvLoadJson, RejectsInvalidThreadMode) {
    const std::string content = R"({
        "poolid": 0, "thread_num": 1, "thread_mode": "spin",
        "wengine_conf": "/x", "node_id": 0, "bvar_port": 0,
        "bind_cpus": "0", "mem_size": 1
    })";
    const auto path = writeTempJson(content);
    ASSERT_FALSE(path.empty());

    KVCacheOptions opts;
    EXPECT_FALSE(::wqskv::loadKVCacheOptionsFromJson(path, opts));
}

TEST_F(WqskvLoadJson, RejectsNonExistentFile) {
    KVCacheOptions opts;
    EXPECT_FALSE(::wqskv::loadKVCacheOptionsFromJson("/no/such/wqskv_config.json", opts));
}

TEST_F(WqskvLoadJson, RejectsMalformedJson) {
    const auto path = writeTempJson("{ not json at all");
    ASSERT_FALSE(path.empty());

    KVCacheOptions opts;
    EXPECT_FALSE(::wqskv::loadKVCacheOptionsFromJson(path, opts));
}

TEST_F(WqskvLoadJson, ParsesOptionalKeys) {
    const std::string content = R"({
        "poolid": 0, "thread_num": 1, "thread_mode": "poll",
        "wengine_conf": "/x", "node_id": 0, "bvar_port": 0,
        "bind_cpus": "0", "mem_size": 1,
        "mempool_objsz": 4096,
        "mempool_objnum": 100,
        "use_round_robin": true,
        "wds_debug_log": true
    })";
    const auto path = writeTempJson(content);
    ASSERT_FALSE(path.empty());

    KVCacheOptions opts;
    ASSERT_TRUE(::wqskv::loadKVCacheOptionsFromJson(path, opts));
    EXPECT_EQ(opts.mempool_obj_sz, 4096u);
    EXPECT_EQ(opts.mempool_obj_num, 100u);
    EXPECT_TRUE(opts.use_round_robin);
    EXPECT_TRUE(opts.debug_log);
}

TEST_F(WqskvLoadJson, RejectsNonNumericBindCpusToken) {
    const std::string content = R"({
        "poolid": 0, "thread_num": 1, "thread_mode": "poll",
        "wengine_conf": "/x", "node_id": 0, "bvar_port": 0,
        "bind_cpus": "abc", "mem_size": 1
    })";
    const auto path = writeTempJson(content);
    ASSERT_FALSE(path.empty());

    KVCacheOptions opts;
    EXPECT_FALSE(::wqskv::loadKVCacheOptionsFromJson(path, opts));
}

} // namespace gtest::wqskv
