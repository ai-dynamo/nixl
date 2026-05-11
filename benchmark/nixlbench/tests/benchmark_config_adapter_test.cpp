/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_config.h"
#include "nixl_types.h"
#include "utils/cli/metadata_plugin_command.h"
#include "utils/cli/raw_execution.h"
#include "utils/utils.h"

#include <gtest/gtest.h>

#include <type_traits>
#include <variant>

namespace nixlbench {
namespace {

const metadataPluginOptionValue &
option(const benchmarkConfig &config, const std::string &name) {
    return config.backend.options.at(name);
}

void
provideStringOption(southboundPluginBenchmarkCommand &command,
                    const std::string &name,
                    const std::string &value) {
    bool found = false;
    for (const auto &option : command.getOptions()) {
        if (option.name != name) {
            continue;
        }

        std::visit(
            [&](auto *target) {
                if constexpr (std::is_same_v<std::decay_t<decltype(*target)>,
                                             metadataPluginOptionValue>) {
                    target->value = value;
                } else if constexpr (std::is_same_v<std::decay_t<decltype(*target)>,
                                                    std::string>) {
                    *target = value;
                } else {
                    ADD_FAILURE() << "Unexpected non-string option target for " << name;
                }
            },
            option.target);
        if (option.provided != nullptr) {
            *option.provided = true;
        }
        found = true;
        break;
    }
    ASSERT_TRUE(found) << "Missing option " << name;
}

void
provideFlagOption(southboundPluginBenchmarkCommand &command, const std::string &name) {
    bool found = false;
    for (const auto &option : command.getOptions()) {
        if (option.name != name) {
            continue;
        }

        std::visit(
            [&](auto *target) {
                if constexpr (std::is_same_v<std::decay_t<decltype(*target)>,
                                             metadataPluginOptionValue>) {
                    target->boolValue = true;
                } else if constexpr (std::is_same_v<std::decay_t<decltype(*target)>, bool>) {
                    *target = true;
                } else {
                    ADD_FAILURE() << "Unexpected non-flag option target for " << name;
                }
            },
            option.target);
        if (option.provided != nullptr) {
            *option.provided = true;
        }
        found = true;
        break;
    }
    ASSERT_TRUE(found) << "Missing option " << name;
}

TEST(BenchmarkConfigAdapterTest, LegacyConversionMapsCommonTransferAndWorkerFields) {
    xferBenchConfig legacy;
    legacy.benchmark_group = "group-a";
    legacy.check_consistency = true;
    legacy.recreate_xfer = true;
    legacy.num_iter = 320;
    legacy.large_blk_iter_ftr = 8;
    legacy.warmup_iter = 32;
    legacy.runtime_type = "ETCD";
    legacy.etcd_endpoints = "http://etcd:2379";
    legacy.initiator_seg_type = XFERBENCH_SEG_TYPE_VRAM;
    legacy.target_seg_type = XFERBENCH_SEG_TYPE_DRAM;
    legacy.scheme = XFERBENCH_SCHEME_MANY_TO_ONE;
    legacy.mode = XFERBENCH_MODE_MG;
    legacy.op_type = XFERBENCH_OP_READ;
    legacy.total_buffer_size = 4096;
    legacy.start_block_size = 64;
    legacy.max_block_size = 512;
    legacy.start_batch_size = 2;
    legacy.max_batch_size = 4;
    legacy.num_threads = 2;
    legacy.worker_type = XFERBENCH_WORKER_NIXL;
    legacy.num_initiator_dev = 3;
    legacy.num_target_dev = 4;
    legacy.enable_pt = true;
    legacy.progress_threads = 5;
    legacy.device_list = "mlx5_0,mlx5_1";
    legacy.enable_vmm = true;

    const benchmarkConfig config = makeBenchmarkConfigFromLegacy(legacy);

    EXPECT_EQ(config.common.benchmark_group, "group-a");
    EXPECT_TRUE(config.common.check_consistency);
    EXPECT_TRUE(config.common.recreate_xfer);
    EXPECT_EQ(config.common.num_iter, 320);
    EXPECT_EQ(config.common.large_blk_iter_ftr, 8);
    EXPECT_EQ(config.common.warmup_iter, 32);
    EXPECT_EQ(config.runtime.etcd_endpoints, "http://etcd:2379");
    EXPECT_EQ(config.transfer.initiator_seg_type, XFERBENCH_SEG_TYPE_VRAM);
    EXPECT_EQ(config.transfer.target_seg_type, XFERBENCH_SEG_TYPE_DRAM);
    EXPECT_EQ(config.transfer.scheme, XFERBENCH_SCHEME_MANY_TO_ONE);
    EXPECT_EQ(config.transfer.mode, XFERBENCH_MODE_MG);
    EXPECT_EQ(config.transfer.op_type, XFERBENCH_OP_READ);
    EXPECT_EQ(config.transfer.total_buffer_size, 4096U);
    EXPECT_EQ(config.transfer.start_block_size, 64U);
    EXPECT_EQ(config.transfer.max_block_size, 512U);
    EXPECT_EQ(config.transfer.start_batch_size, 2U);
    EXPECT_EQ(config.transfer.max_batch_size, 4U);
    EXPECT_EQ(config.transfer.num_threads, 2);
    EXPECT_EQ(config.worker.type, XFERBENCH_WORKER_NIXL);
    EXPECT_EQ(config.worker.num_initiator_dev, 3);
    EXPECT_EQ(config.worker.num_target_dev, 4);
    EXPECT_TRUE(config.worker.enable_progress_thread);
    EXPECT_EQ(config.worker.progress_threads, 5U);
    EXPECT_EQ(config.worker.device_list, "mlx5_0,mlx5_1");
    EXPECT_TRUE(config.worker.enable_vmm);
}

TEST(BenchmarkConfigAdapterTest, RawRequestConversionMapsStorageAndGdsFields) {
    rawRequest request;
    request.backend.setProvided(XFERBENCH_BACKEND_GDS);
    request.filepath.setProvided("/data");
    request.filenames.setProvided("a.bin,b.bin");
    request.num_files.setProvided(2);
    request.storage_enable_direct.setProvided(true);
    request.backend_memory_types = {FILE_SEG};
    request.gds_batch_pool_size.setProvided(7);
    request.gds_batch_limit.setProvided(11);

    const benchmarkConfig config = makeBenchmarkConfigFromRawRequest(request);

    EXPECT_EQ(config.backend.name, XFERBENCH_BACKEND_GDS);
    EXPECT_TRUE(isStorageBackend(config.backend));
    EXPECT_EQ(config.storage.filepath, "/data");
    EXPECT_EQ(config.storage.filenames, "a.bin,b.bin");
    EXPECT_EQ(config.storage.num_files, 2);
    EXPECT_TRUE(config.storage.enable_direct);
    EXPECT_EQ(option(config, "batch_pool_size").value, "7");
    EXPECT_EQ(option(config, "batch_limit").value, "11");
}

TEST(BenchmarkConfigAdapterTest, LegacyConversionMapsObjBackendOptions) {
    xferBenchConfig legacy;
    legacy.backend = XFERBENCH_BACKEND_OBJ;
    legacy.obj_access_key = "access";
    legacy.obj_secret_key = "secret";
    legacy.obj_session_token = "token";
    legacy.obj_bucket_name = "bucket";
    legacy.obj_scheme = XFERBENCH_OBJ_SCHEME_HTTPS;
    legacy.obj_region = "us-west-2";
    legacy.obj_use_virtual_addressing = true;
    legacy.obj_endpoint_override = "https://endpoint";
    legacy.obj_req_checksum = XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED;
    legacy.obj_ca_bundle = "/tmp/ca.pem";
    legacy.obj_crt_min_limit = 1234;
    legacy.obj_accelerated_enable = true;
    legacy.obj_accelerated_type = "vendor";

    const benchmarkConfig config = makeBenchmarkConfigFromLegacy(legacy);

    EXPECT_TRUE(isObjStorageBackend(config.backend));
    EXPECT_EQ(option(config, "access_key").value, "access");
    EXPECT_EQ(option(config, "secret_key").value, "secret");
    EXPECT_EQ(option(config, "session_token").value, "token");
    EXPECT_EQ(option(config, "bucket").value, "bucket");
    EXPECT_EQ(option(config, "scheme").value, XFERBENCH_OBJ_SCHEME_HTTPS);
    EXPECT_EQ(option(config, "region").value, "us-west-2");
    EXPECT_TRUE(option(config, "use_virtual_addressing").boolValue);
    EXPECT_EQ(option(config, "endpoint_override").value, "https://endpoint");
    EXPECT_EQ(option(config, "req_checksum").value, XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED);
    EXPECT_EQ(option(config, "ca_bundle").value, "/tmp/ca.pem");
    EXPECT_EQ(option(config, "crtMinLimit").value, "1234");
    EXPECT_TRUE(option(config, "accelerated").boolValue);
    EXPECT_EQ(option(config, "type").value, "vendor");
}

TEST(BenchmarkConfigAdapterTest, LegacyConversionMapsGusliFieldsWithoutValidationSideEffects) {
    xferBenchConfig legacy;
    legacy.backend = XFERBENCH_BACKEND_GUSLI;
    legacy.filepath = "/gusli";
    legacy.filenames = "dev0";
    legacy.num_files = 1;
    legacy.storage_enable_direct = false;
    legacy.gusli_client_name = "client-a";
    legacy.gusli_max_simultaneous_requests = 64;
    legacy.gusli_config_file = "config-body";
    legacy.gusli_device_byte_offsets = "1024,2048";
    legacy.gusli_device_security = "sec=0x3,sec=0x71";

    const benchmarkConfig config = makeBenchmarkConfigFromLegacy(legacy);

    EXPECT_EQ(config.backend.name, XFERBENCH_BACKEND_GUSLI);
    EXPECT_EQ(config.storage.filepath, "/gusli");
    EXPECT_EQ(config.storage.filenames, "dev0");
    EXPECT_EQ(config.storage.num_files, 1);
    EXPECT_FALSE(config.storage.enable_direct);
    EXPECT_EQ(option(config, "client_name").value, "client-a");
    EXPECT_EQ(option(config, "max_num_simultaneous_requests").value, "64");
    EXPECT_EQ(option(config, "config_file").value, "config-body");
    EXPECT_EQ(option(config, "device_byte_offsets").value, "1024,2048");
    EXPECT_EQ(option(config, "device_security").value, "sec=0x3,sec=0x71");
}

TEST(BenchmarkConfigAdapterTest, RawRequestPreservesDynamicPluginOptions) {
    rawRequest request;
    request.backend.setProvided("NEW_PLUGIN");
    request.backend_capabilities.requiresDirectStorage = true;
    request.backend_memory_types = {FILE_SEG, DRAM_SEG};
    request.backend_options["custom_param"] = {"custom-value", false, true};

    const benchmarkConfig config = makeBenchmarkConfigFromRawRequest(request);

    EXPECT_EQ(config.backend.name, "NEW_PLUGIN");
    EXPECT_TRUE(config.backend.capabilities.requiresDirectStorage);
    EXPECT_EQ(config.backend.memory_types, (nixl_mem_list_t{FILE_SEG, DRAM_SEG}));
    EXPECT_EQ(option(config, "custom_param").value, "custom-value");
}

TEST(BenchmarkConfigAdapterTest, MetadataPluginStoresProvidedFileWorkloadOptions) {
    nixlBackendPluginCapabilities capabilities;
    nixl_b_params_t option_specs{
        {"use_posix_aio",
         "true"},
    };
    metadataPluginCommand plugin(
        XFERBENCH_BACKEND_POSIX, capabilities, option_specs, {FILE_SEG, DRAM_SEG});
    provideStringOption(plugin, "filepath", "/images/containerSpace/");
    provideFlagOption(plugin, "enable_direct");
    provideFlagOption(plugin, "use_posix_aio");

    const auto &options = plugin.metadataOptions();

    EXPECT_EQ(options.at("filepath").value, "/images/containerSpace/");
    EXPECT_TRUE(options.at("filepath").isProvided);
    EXPECT_TRUE(options.at("enable_direct").boolValue);
    EXPECT_TRUE(options.at("enable_direct").isProvided);
    EXPECT_TRUE(options.at("use_posix_aio").boolValue);
}

TEST(BenchmarkConfigAdapterTest, StructuredCopyBackMapsCommonTransferWorkerAndStorageFields) {
    benchmarkConfig config;
    config.common.benchmark_group = "group-b";
    config.common.check_consistency = true;
    config.common.recreate_xfer = true;
    config.common.num_iter = 128;
    config.common.large_blk_iter_ftr = 4;
    config.common.warmup_iter = 16;
    config.runtime.type = XFERBENCH_RT_ETCD;
    config.runtime.etcd_endpoints = "http://etcd:1234";
    config.transfer.initiator_seg_type = XFERBENCH_SEG_TYPE_VRAM;
    config.transfer.target_seg_type = XFERBENCH_SEG_TYPE_DRAM;
    config.transfer.scheme = XFERBENCH_SCHEME_ONE_TO_MANY;
    config.transfer.mode = XFERBENCH_MODE_MG;
    config.transfer.op_type = XFERBENCH_OP_READ;
    config.transfer.total_buffer_size = 8192;
    config.transfer.start_block_size = 128;
    config.transfer.max_block_size = 1024;
    config.transfer.start_batch_size = 2;
    config.transfer.max_batch_size = 8;
    config.transfer.num_threads = 3;
    config.worker.type = XFERBENCH_WORKER_NIXL;
    config.worker.num_initiator_dev = 2;
    config.worker.num_target_dev = 5;
    config.worker.enable_progress_thread = true;
    config.worker.progress_threads = 6;
    config.worker.device_list = "mlx5_2";
    config.worker.enable_vmm = true;
    config.backend.name = XFERBENCH_BACKEND_GDS;
    config.backend.options["batch_pool_size"] = {"9", false, true};
    config.backend.options["batch_limit"] = {"17", false, true};
    config.storage.filepath = "/storage";
    config.storage.filenames = "a,b";
    config.storage.num_files = 2;
    config.storage.enable_direct = true;

    const xferBenchConfig legacy = makeLegacyConfigFromBenchmarkConfig(config);

    EXPECT_EQ(legacy.benchmark_group, "group-b");
    EXPECT_TRUE(legacy.check_consistency);
    EXPECT_TRUE(legacy.recreate_xfer);
    EXPECT_EQ(legacy.num_iter, 128);
    EXPECT_EQ(legacy.large_blk_iter_ftr, 4);
    EXPECT_EQ(legacy.warmup_iter, 16);
    EXPECT_EQ(legacy.runtime_type, XFERBENCH_RT_ETCD);
    EXPECT_EQ(legacy.etcd_endpoints, "http://etcd:1234");
    EXPECT_EQ(legacy.initiator_seg_type, XFERBENCH_SEG_TYPE_VRAM);
    EXPECT_EQ(legacy.target_seg_type, XFERBENCH_SEG_TYPE_DRAM);
    EXPECT_EQ(legacy.scheme, XFERBENCH_SCHEME_ONE_TO_MANY);
    EXPECT_EQ(legacy.mode, XFERBENCH_MODE_MG);
    EXPECT_EQ(legacy.op_type, XFERBENCH_OP_READ);
    EXPECT_EQ(legacy.total_buffer_size, 8192U);
    EXPECT_EQ(legacy.start_block_size, 128U);
    EXPECT_EQ(legacy.max_block_size, 1024U);
    EXPECT_EQ(legacy.start_batch_size, 2U);
    EXPECT_EQ(legacy.max_batch_size, 8U);
    EXPECT_EQ(legacy.num_threads, 3);
    EXPECT_EQ(legacy.worker_type, XFERBENCH_WORKER_NIXL);
    EXPECT_EQ(legacy.num_initiator_dev, 2);
    EXPECT_EQ(legacy.num_target_dev, 5);
    EXPECT_TRUE(legacy.enable_pt);
    EXPECT_EQ(legacy.progress_threads, 6U);
    EXPECT_EQ(legacy.device_list, "mlx5_2");
    EXPECT_TRUE(legacy.enable_vmm);
    EXPECT_EQ(legacy.backend, XFERBENCH_BACKEND_GDS);
    EXPECT_EQ(legacy.gds_batch_pool_size, 9);
    EXPECT_EQ(legacy.gds_batch_limit, 17);
    EXPECT_EQ(legacy.filepath, "/storage");
    EXPECT_EQ(legacy.filenames, "a,b");
    EXPECT_EQ(legacy.num_files, 2);
    EXPECT_TRUE(legacy.storage_enable_direct);
}

TEST(BenchmarkConfigAdapterTest, StructuredCopyBackMapsPosixAndObjBackendOptions) {
    benchmarkConfig posix_config;
    posix_config.backend.name = XFERBENCH_BACKEND_POSIX;
    posix_config.backend.options["use_uring"] = {"false", true, true};
    posix_config.backend.options["ios_pool_size"] = {"13", false, true};
    posix_config.backend.options["kernel_queue_size"] = {"21", false, true};

    const xferBenchConfig posix_legacy = makeLegacyConfigFromBenchmarkConfig(posix_config);

    EXPECT_EQ(posix_legacy.posix_api_type, XFERBENCH_POSIX_API_URING);
    EXPECT_EQ(posix_legacy.posix_ios_pool_size, 13);
    EXPECT_EQ(posix_legacy.posix_kernel_queue_size, 21);

    benchmarkConfig obj_config;
    obj_config.backend.name = XFERBENCH_BACKEND_OBJ;
    obj_config.backend.options["access_key"] = {"access", false, true};
    obj_config.backend.options["secret_key"] = {"secret", false, true};
    obj_config.backend.options["session_token"] = {"token", false, true};
    obj_config.backend.options["bucket"] = {"bucket", false, true};
    obj_config.backend.options["scheme"] = {XFERBENCH_OBJ_SCHEME_HTTPS, false, true};
    obj_config.backend.options["region"] = {"region", false, true};
    obj_config.backend.options["use_virtual_addressing"] = {"true", true, true};
    obj_config.backend.options["endpoint_override"] = {"endpoint", false, true};
    obj_config.backend.options["req_checksum"] = {XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED,
                                                   false,
                                                   true};
    obj_config.backend.options["ca_bundle"] = {"ca.pem", false, true};
    obj_config.backend.options["crtMinLimit"] = {"55", false, true};
    obj_config.backend.options["accelerated"] = {"true", true, true};
    obj_config.backend.options["type"] = {"vendor", false, true};

    const xferBenchConfig obj_legacy = makeLegacyConfigFromBenchmarkConfig(obj_config);

    EXPECT_EQ(obj_legacy.obj_access_key, "access");
    EXPECT_EQ(obj_legacy.obj_secret_key, "secret");
    EXPECT_EQ(obj_legacy.obj_session_token, "token");
    EXPECT_EQ(obj_legacy.obj_bucket_name, "bucket");
    EXPECT_EQ(obj_legacy.obj_scheme, XFERBENCH_OBJ_SCHEME_HTTPS);
    EXPECT_EQ(obj_legacy.obj_region, "region");
    EXPECT_TRUE(obj_legacy.obj_use_virtual_addressing);
    EXPECT_EQ(obj_legacy.obj_endpoint_override, "endpoint");
    EXPECT_EQ(obj_legacy.obj_req_checksum, XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED);
    EXPECT_EQ(obj_legacy.obj_ca_bundle, "ca.pem");
    EXPECT_EQ(obj_legacy.obj_crt_min_limit, 55U);
    EXPECT_TRUE(obj_legacy.obj_accelerated_enable);
    EXPECT_EQ(obj_legacy.obj_accelerated_type, "vendor");
}

TEST(BenchmarkConfigAdapterTest, RawValidationAppliesPreRunAdjustments) {
    xferBenchConfig config;
    config.backend = XFERBENCH_BACKEND_UCX;
    config.etcd_endpoints = "";
    config.total_buffer_size = 12288;
    config.max_block_size = 1024;
    config.max_batch_size = 1;
    config.num_threads = 3;
    config.num_initiator_dev = 1;
    config.num_target_dev = 1;
    config.large_blk_iter_ftr = 2;
    config.num_iter = 10;
    config.warmup_iter = 7;

    ASSERT_TRUE(validateRawConfigForRun(config));

    EXPECT_EQ(config.etcd_endpoints, "http://localhost:2379");
    EXPECT_EQ(config.num_iter, 12);
    EXPECT_EQ(config.warmup_iter, 12);
}

TEST(BenchmarkConfigAdapterTest, RawValidationRejectsInvalidPosixApi) {
    xferBenchConfig config;
    config.backend = XFERBENCH_BACKEND_POSIX;
    config.posix_api_type = "bad";

    EXPECT_FALSE(validateRawConfigForRun(config));
}

} // namespace
} // namespace nixlbench
