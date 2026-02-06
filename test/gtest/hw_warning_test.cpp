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
#include <algorithm>
#include <string>
#include <vector>

#include "common.h"
#include "nixl.h"
#include "ucx_utils.h"
#include "common/hw_info.h"

class HardwareWarningTest : public ::testing::Test {
protected:
    gtest::ScopedEnv envHelper_;
};

/**
 * Test that a warning is logged when NVIDIA GPUs are present but UCX
 * CUDA support is not available.
 */
TEST_F(HardwareWarningTest, WarnWhenGpuPresentButCudaNotSupported) {
    const nixl::hwInfo hw_info;
    if (hw_info.numNvidiaGpus == 0) {
        GTEST_SKIP() << "No NVIDIA GPUs detected, skipping test";
    }

    // Disable CUDA transport in UCX
    envHelper_.addVar("UCX_TLS", "^cuda");

    std::vector<std::string> devs;
    nixlUcxContext ctx(devs, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE, 0);

    gtest::scopedTestLogSink log_sink;
    ctx.warnAboutHardwareSupportMismatch();

    EXPECT_EQ(log_sink.warningCount(), 1);
    EXPECT_EQ(log_sink.countWarningsMatching(
                  "NVIDIA GPU(s) were detected, but UCX CUDA support was not found"),
              1);

    envHelper_.popVar();
}

/**
 * Test that a warning is logged when IB devices are present but UCX
 * RDMA support is not available.
 *
 * Note: This warning only triggers for UCX >= 1.21.
 */
TEST_F(HardwareWarningTest, WarnWhenIbPresentButRdmaNotSupported) {
    unsigned major, minor, release;
    ucp_get_version(&major, &minor, &release);
    if (UCP_VERSION(major, minor) < UCP_VERSION(1, 21)) {
        GTEST_SKIP() << "UCX version " << major << "." << minor
                     << " is less than 1.21, skipping test";
    }

    const nixl::hwInfo hw_info;
    if (hw_info.numIbDevices == 0) {
        GTEST_SKIP() << "No IB devices detected, skipping test";
    }

    // Disable IB transport in UCX
    envHelper_.addVar("UCX_TLS", "^ib");

    std::vector<std::string> devs;
    nixlUcxContext ctx(devs, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE, 0);

    gtest::scopedTestLogSink log_sink;
    ctx.warnAboutHardwareSupportMismatch();

    EXPECT_EQ(log_sink.warningCount(), 1);
    EXPECT_EQ(log_sink.countWarningsMatching(
                  "IB device(s) were detected, but accelerated IB support was not found"),
              1);

    envHelper_.popVar();
}

/**
 * Test that no warnings are logged when UCX_TLS includes both ib and cuda.
 */
TEST_F(HardwareWarningTest, NoWarningWhenIbAndCudaSupported) {
    const nixl::hwInfo hw_info;
    if (hw_info.numNvidiaGpus == 0 || hw_info.numIbDevices == 0) {
        GTEST_SKIP() << "No NVIDIA GPUs or IB devices detected, skipping test";
    }

    // Enable IB and CUDA transports in UCX
    envHelper_.addVar("UCX_TLS", "ib,cuda");

    std::vector<std::string> devs;
    nixlUcxContext ctx(devs, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE, 0);

    gtest::scopedTestLogSink log_sink;
    ctx.warnAboutHardwareSupportMismatch();

    EXPECT_EQ(log_sink.warningCount(), 0);

    envHelper_.popVar();
}

/**
 * Test that a warning is logged when EFA devices are present but a
 * non-LIBFABRIC backend is created.
 */
TEST_F(HardwareWarningTest, WarnWhenEfaPresentAndNonLibfabricBackend) {
    const nixl::hwInfo hw_info;
    if (hw_info.numEfaDevices == 0) {
        GTEST_SKIP() << "No EFA devices detected, skipping test";
    }

    envHelper_.addVar("NIXL_PLUGIN_DIR", std::string(BUILD_DIR) + "/src/plugins/ucx");
    nixlAgent agent("EfaTestAgent", nixlAgentConfig(true));

    gtest::scopedTestLogSink log_sink;

    nixlBackendH *backend;
    EXPECT_EQ(agent.createBackend("UCX", {}, backend), NIXL_SUCCESS);

    EXPECT_EQ(log_sink.warningCount(), 1);
    EXPECT_EQ(
        log_sink.countWarningsMatching("Amazon EFA(s) were detected, it is recommended to use "
                                       "the LIBFABRIC backend for best performance"),
        1);
}

/**
 * Test that no warning is logged when EFA devices are present and the
 * LIBFABRIC backend is created.
 */
TEST_F(HardwareWarningTest, NoWarningWhenEfaPresentAndLibfabricBackend) {
#ifndef HAVE_LIBFABRIC
    GTEST_SKIP() << "LIBFABRIC plugin not built";
#endif

    const nixl::hwInfo hw_info;
    if (hw_info.numEfaDevices == 0) {
        GTEST_SKIP() << "No EFA devices detected, skipping test";
    }

    envHelper_.addVar("NIXL_PLUGIN_DIR", std::string(BUILD_DIR) + "/src/plugins/libfabric");
    nixlAgent agent("EfaTestAgent", nixlAgentConfig(true));

    gtest::scopedTestLogSink log_sink;

    nixlBackendH *backend;
    EXPECT_EQ(agent.createBackend("LIBFABRIC", {}, backend), NIXL_SUCCESS);

    EXPECT_EQ(log_sink.warningCount(), 0);
}
