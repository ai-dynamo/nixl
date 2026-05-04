/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "worker/nixl/nixl_backend_params.h"

#include "utils/utils.h"

#include <gtest/gtest.h>

namespace nixlbench {
namespace {

void
setOption(benchmarkConfig &config, const std::string &name, const char *value) {
    config.backend.options[name] = {value, false, true};
}

void
setOption(benchmarkConfig &config, const std::string &name, bool value) {
    config.backend.options[name] = {value ? "true" : "false", value, true};
}

TEST(NixlBackendParamsTest, UcxAddsBenchmarkComputedOptions) {
    benchmarkConfig config;
    config.backend.name = XFERBENCH_BACKEND_UCX;
    config.worker.progress_threads = 4;
    config.worker.num_initiator_dev = 2;
    config.worker.num_target_dev = 2;
    config.transfer.num_threads = 7;
    std::vector<std::string> devices{"mlx5_0", "mlx5_1"};

    const nixl_b_params_t params =
        buildNixlBackendParams(config, {}, devices, false, 3);

    EXPECT_EQ(params.at("num_threads"), "4");
    EXPECT_EQ(params.at("num_workers"), "8");
    EXPECT_EQ(params.at("device_list"), "mlx5_1");
}

TEST(NixlBackendParamsTest, PosixConvertsCompatibilityApiOption) {
    benchmarkConfig config;
    config.backend.name = XFERBENCH_BACKEND_POSIX;
    setOption(config, "use_aio", XFERBENCH_POSIX_API_URING);
    setOption(config, "ios_pool_size", "13");
    setOption(config, "kernel_queue_size", "21");

    const nixl_b_params_t params = buildNixlBackendParams(config, {}, {"all"}, true, 0);

    EXPECT_EQ(params.at("use_aio"), "false");
    EXPECT_EQ(params.at("use_uring"), "true");
    EXPECT_EQ(params.at("use_posix_aio"), "false");
    EXPECT_EQ(params.at("ios_pool_size"), "13");
    EXPECT_EQ(params.at("kernel_queue_size"), "21");
}

TEST(NixlBackendParamsTest, ObjPassesProvidedPluginOptions) {
    benchmarkConfig config;
    config.backend.name = XFERBENCH_BACKEND_OBJ;
    setOption(config, "access_key", "access");
    setOption(config, "secret_key", "secret");
    setOption(config, "bucket", "bucket");
    setOption(config, "scheme", XFERBENCH_OBJ_SCHEME_HTTPS);
    setOption(config, "region", "us-west-2");
    setOption(config, "use_virtual_addressing", true);
    setOption(config, "endpoint_override", "endpoint");
    setOption(config, "req_checksum", XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED);
    setOption(config, "ca_bundle", "ca.pem");
    setOption(config, "accelerated", true);
    setOption(config, "type", "vendor");

    const nixl_b_params_t params = buildNixlBackendParams(config, {}, {"all"}, true, 0);

    EXPECT_EQ(params.at("access_key"), "access");
    EXPECT_EQ(params.at("secret_key"), "secret");
    EXPECT_EQ(params.at("bucket"), "bucket");
    EXPECT_EQ(params.at("scheme"), XFERBENCH_OBJ_SCHEME_HTTPS);
    EXPECT_EQ(params.at("region"), "us-west-2");
    EXPECT_EQ(params.at("use_virtual_addressing"), "true");
    EXPECT_EQ(params.at("endpoint_override"), "endpoint");
    EXPECT_EQ(params.at("req_checksum"), XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED);
    EXPECT_EQ(params.at("ca_bundle"), "ca.pem");
    EXPECT_EQ(params.at("accelerated"), "true");
    EXPECT_EQ(params.at("type"), "vendor");
}

TEST(NixlBackendParamsTest, StorageBackendParamsPassThroughByDefault) {
    benchmarkConfig config;
    config.backend.name = XFERBENCH_BACKEND_HF3FS;
    setOption(config, "iopool_size", "96");

    const nixl_b_params_t params = buildNixlBackendParams(config, {}, {"all"}, true, 0);

    EXPECT_EQ(params.at("iopool_size"), "96");
}

TEST(NixlBackendParamsTest, DefaultsFromPluginArePreservedWhenOptionNotProvided) {
    benchmarkConfig config;
    config.backend.name = "NEW_PLUGIN";
    config.backend.capabilities.canUseAsStorage = true;
    config.backend.options["new_option"] = {"from-cli-default", false, false};
    nixl_b_params_t defaults{{"new_option", "from-plugin-default"}};

    const nixl_b_params_t params =
        buildNixlBackendParams(config, defaults, {"all"}, true, 0);

    EXPECT_EQ(params.at("new_option"), "from-plugin-default");
}

} // namespace
} // namespace nixlbench
