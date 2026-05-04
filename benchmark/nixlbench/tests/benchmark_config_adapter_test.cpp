/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_config.h"
#include "utils/utils.h"

#include <gtest/gtest.h>

namespace nixlbench {
namespace {

const metadataPluginOptionValue &
option(const benchmarkConfig &config, const std::string &name) {
    return config.backend.options.at(name);
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
    EXPECT_TRUE(config.worker.enable_pt);
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
    request.gds_batch_pool_size.setProvided(7);
    request.gds_batch_limit.setProvided(11);

    const benchmarkConfig config = makeBenchmarkConfigFromRawRequest(request);

    EXPECT_EQ(config.backend.name, XFERBENCH_BACKEND_GDS);
    EXPECT_TRUE(isStorageBackend(config.backend));
    EXPECT_TRUE(config.backend.capabilities.canReadWriteFiles);
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

    EXPECT_TRUE(isStorageBackend(config.backend));
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
    EXPECT_TRUE(isStorageBackend(config.backend));
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

} // namespace
} // namespace nixlbench
