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

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"

namespace gtest::rados {

namespace {

    const char *
    environmentValue(const char *name) {
        const char *value = std::getenv(name);
        return value == nullptr ? "" : value;
    }

    constexpr const char *localAgent = "rados_discovery_test";

} // namespace

class radosDiscoveryFixture : public testing::Test {
protected:
    void
    SetUp() override {
        if (std::getenv("NIXL_PLUGIN_DIR") == nullptr) {
            ASSERT_EQ(setenv("NIXL_PLUGIN_DIR", RADOS_TEST_PLUGIN_DIR, 1), 0)
                << "cannot export NIXL_PLUGIN_DIR";
        }
    }
};

// One representative through the loaded plugin's factory: agent-level error
// propagation for a rejected parameter, failing before any Ceph access. The
// rejection matrix itself is owned by radosConfigTest.
TEST_F(radosDiscoveryFixture, CreateBackendRejectsUnprefixedParameter) {
    nixl_b_params_t parameters{{"conf", "missing-ceph.conf"},
                               {"pool", "qa-pool"},
                               {"name", "qa-client"},
                               {"qa_unprefixed_option", "1"}};
    const nixlAgentConfig agent_config;
    nixlAgent agent(localAgent, agent_config);

    nixlBackendH *backend = nullptr;
    EXPECT_EQ(agent.createBackend("RADOS", parameters, backend), NIXL_ERR_BACKEND);
    EXPECT_EQ(backend, nullptr);
}

TEST_F(radosDiscoveryFixture, LoadedPluginDeclaresExactlyDramAndObjSegments) {
    const nixlAgentConfig agent_config;
    nixlAgent agent(localAgent, agent_config);

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    ASSERT_EQ(agent.getPluginParams("RADOS", mems, params), NIXL_SUCCESS);
    EXPECT_EQ(mems, (nixl_mem_list_t{DRAM_SEG, OBJ_SEG}));
    EXPECT_TRUE(params.empty());
}

TEST_F(radosDiscoveryFixture, DirectRoundTripAndCleanShutdown) {
    const std::string conf = environmentValue("NIXL_RADOS_CONF");
    const std::string pool = environmentValue("NIXL_RADOS_POOL");
    const std::string name = environmentValue("NIXL_RADOS_NAME");
    if (conf.empty() || pool.empty() || name.empty()) {
        GTEST_SKIP() << "set NIXL_RADOS_CONF, NIXL_RADOS_POOL, and "
                        "NIXL_RADOS_NAME to run this live RADOS case";
    }
    ASSERT_EQ(access(conf.c_str(), R_OK), 0) << "Ceph conf is not readable: " << conf;

    nixl_b_params_t parameters{{"conf", conf}, {"pool", pool}, {"name", name}};
    const std::string object_namespace = environmentValue("NIXL_RADOS_NS");
    if (!object_namespace.empty()) {
        parameters["namespace"] = object_namespace;
    }
    const nixlAgentConfig agent_config;
    nixlAgent agent(localAgent, agent_config);

    nixlBackendH *backend = nullptr;
    ASSERT_EQ(agent.createBackend("RADOS", parameters, backend), NIXL_SUCCESS);
    ASSERT_NE(backend, nullptr);

    std::vector<unsigned char> bytes(1024, 0x5a);
    std::vector<unsigned char> readback(bytes.size(), 0x00);
    nixl_opt_args_t reg_args;
    reg_args.backends = {backend};
    const nixlBlobDesc dram_desc(reinterpret_cast<uintptr_t>(bytes.data()), bytes.size(), 0, {});
    const nixlBlobDesc readback_desc(
        reinterpret_cast<uintptr_t>(readback.data()), readback.size(), 0, {});
    nixl_reg_dlist_t dram_registrations(DRAM_SEG);
    dram_registrations.addDesc(dram_desc);
    dram_registrations.addDesc(readback_desc);
    ASSERT_EQ(agent.registerMem(dram_registrations, &reg_args), NIXL_SUCCESS);

    // Unique key per run; the plugin has no delete operation, so the object is
    // deliberately retained in the pool.
    const std::string object_key = "nixl-rados-gtest-" + std::to_string(::getpid()) + "-" +
        std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count());
    constexpr uint64_t object_offset = 4096;
    const nixlBlobDesc object_desc(object_offset, bytes.size(), 1, object_key);
    nixl_reg_dlist_t object_registrations(OBJ_SEG);
    object_registrations.addDesc(object_desc);
    ASSERT_EQ(agent.registerMem(object_registrations, &reg_args), NIXL_SUCCESS);

    const auto run_to_completion = [&agent](nixlXferReqH *request) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        nixl_status_t status = agent.postXferReq(request);
        while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            status = agent.getXferStatus(request);
        }
        EXPECT_EQ(status, NIXL_SUCCESS) << "live RADOS transfer did not complete";
        EXPECT_EQ(agent.releaseXferReq(request), NIXL_SUCCESS);
    };

    nixl_xfer_dlist_t write_local(DRAM_SEG);
    write_local.addDesc(dram_desc);
    nixl_xfer_dlist_t object_remote(OBJ_SEG);
    object_remote.addDesc(nixlBasicDesc(object_offset, bytes.size(), 1));
    nixlXferReqH *write_request = nullptr;
    ASSERT_EQ(agent.createXferReq(
                  NIXL_WRITE, write_local, object_remote, localAgent, write_request, &reg_args),
              NIXL_SUCCESS);
    run_to_completion(write_request);

    nixl_xfer_dlist_t read_local(DRAM_SEG);
    read_local.addDesc(readback_desc);
    nixlXferReqH *read_request = nullptr;
    ASSERT_EQ(agent.createXferReq(
                  NIXL_READ, read_local, object_remote, localAgent, read_request, &reg_args),
              NIXL_SUCCESS);
    run_to_completion(read_request);
    EXPECT_EQ(bytes, readback);

    if (!object_namespace.empty()) {
        // A default-namespace backend on the same pool must not see the
        // namespaced object: the read must terminate NIXL_ERR_NOT_FOUND.
        nixl_b_params_t plain{{"conf", conf}, {"pool", pool}, {"name", name}};
        const nixlAgentConfig default_ns_agent_config;
        nixlAgent default_ns_agent(localAgent, default_ns_agent_config);
        nixlBackendH *default_ns_backend = nullptr;
        ASSERT_EQ(default_ns_agent.createBackend("RADOS", plain, default_ns_backend), NIXL_SUCCESS);
        ASSERT_NE(default_ns_backend, nullptr);
        nixl_opt_args_t default_ns_args;
        default_ns_args.backends = {default_ns_backend};

        std::vector<unsigned char> invisible(bytes.size(), 0x00);
        const nixlBlobDesc invisible_desc(
            reinterpret_cast<uintptr_t>(invisible.data()), invisible.size(), 0, {});
        nixl_reg_dlist_t invisible_reg(DRAM_SEG);
        invisible_reg.addDesc(invisible_desc);
        ASSERT_EQ(default_ns_agent.registerMem(invisible_reg, &default_ns_args), NIXL_SUCCESS);
        nixl_reg_dlist_t object_reg(OBJ_SEG);
        object_reg.addDesc(object_desc);
        ASSERT_EQ(default_ns_agent.registerMem(object_reg, &default_ns_args), NIXL_SUCCESS);

        nixl_xfer_dlist_t invisible_local(DRAM_SEG);
        invisible_local.addDesc(invisible_desc);
        nixlXferReqH *invisible_request = nullptr;
        ASSERT_EQ(default_ns_agent.createXferReq(NIXL_READ,
                                                 invisible_local,
                                                 object_remote,
                                                 localAgent,
                                                 invisible_request,
                                                 &default_ns_args),
                  NIXL_SUCCESS);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        nixl_status_t status = default_ns_agent.postXferReq(invisible_request);
        while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            status = default_ns_agent.getXferStatus(invisible_request);
        }
        EXPECT_EQ(status, NIXL_ERR_NOT_FOUND)
            << "namespaced object leaked into the pool's default namespace";
        EXPECT_EQ(default_ns_agent.releaseXferReq(invisible_request), NIXL_SUCCESS);
        EXPECT_EQ(default_ns_agent.deregisterMem(object_reg, &default_ns_args), NIXL_SUCCESS);
        EXPECT_EQ(default_ns_agent.deregisterMem(invisible_reg, &default_ns_args), NIXL_SUCCESS);
    }

    ASSERT_EQ(agent.deregisterMem(object_registrations, &reg_args), NIXL_SUCCESS);
    ASSERT_EQ(agent.deregisterMem(dram_registrations, &reg_args), NIXL_SUCCESS);
}

} // namespace gtest::rados
