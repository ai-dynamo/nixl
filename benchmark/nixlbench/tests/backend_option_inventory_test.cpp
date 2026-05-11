/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/metadata_plugin_command.h"

#include <gtest/gtest.h>

#include <initializer_list>
#include <string>

namespace nixlbench {
namespace {

nixl_b_params_t
availableOptions(std::initializer_list<const char *> names) {
    nixl_b_params_t options;
    for (const auto *name : names) {
        options.emplace(name, "");
    }
    return options;
}

void
expectOption(const metadataPluginCommand &command, const std::string &name) {
    const auto &options = command.metadataOptions();
    const auto option = options.find(name);
    ASSERT_NE(option, options.end()) << name;
    EXPECT_FALSE(option->second.isProvided) << name;
}

TEST(BackendOptionInventoryTest, UcxOptionsExposeBenchAndEngineKeys) {
    metadataPluginCommand command("UCX",
                                  {},
                                  availableOptions({"ucx_devices",
                                                    "device_list",
                                                    "num_workers",
                                                    "num_threads",
                                                    "split_batch_size",
                                                    "ucx_num_device_channels",
                                                    "engine_config",
                                                    "ucx_error_handling_mode"}),
                                  {DRAM_SEG, VRAM_SEG});

    expectOption(command, "ucx_devices");
    expectOption(command, "device_list");
    expectOption(command, "num_workers");
    expectOption(command, "num_threads");
    expectOption(command, "split_batch_size");
    expectOption(command, "ucx_num_device_channels");
    expectOption(command, "engine_config");
    expectOption(command, "ucx_error_handling_mode");
}

TEST(BackendOptionInventoryTest, StorageBackendsExposeOptions) {
    metadataPluginCommand posix("POSIX",
                                {},
                                availableOptions({"use_aio",
                                                  "use_uring",
                                                  "use_posix_aio",
                                                  "ios_pool_size",
                                                  "kernel_queue_size"}),
                                {DRAM_SEG, FILE_SEG});
    expectOption(posix, "use_aio");
    expectOption(posix, "use_uring");
    expectOption(posix, "use_posix_aio");
    expectOption(posix, "ios_pool_size");
    expectOption(posix, "kernel_queue_size");

    metadataPluginCommand gds("GDS",
                              {},
                              availableOptions({"batch_pool_size",
                                                "batch_limit",
                                                "max_request_size"}),
                              {DRAM_SEG, VRAM_SEG, FILE_SEG});
    expectOption(gds, "batch_pool_size");
    expectOption(gds, "batch_limit");
    expectOption(gds, "max_request_size");

    metadataPluginCommand hf3fs("HF3FS",
                                {},
                                availableOptions({"mount_point", "mem_config", "iopool_size"}),
                                {FILE_SEG, DRAM_SEG});
    expectOption(hf3fs, "mount_point");
    expectOption(hf3fs, "mem_config");
    expectOption(hf3fs, "iopool_size");
}

TEST(BackendOptionInventoryTest, ObjectBackendsExposeOptions) {
    metadataPluginCommand obj("OBJ",
                              {},
                              availableOptions({"access_key",
                                                "secret_key",
                                                "session_token",
                                                "bucket",
                                                "scheme",
                                                "region",
                                                "use_virtual_addressing",
                                                "endpoint_override",
                                                "req_checksum",
                                                "ca_bundle",
                                                "crtMinLimit",
                                                "accelerated",
                                                "type",
                                                "num_threads"}),
                              {DRAM_SEG, OBJ_SEG});
    expectOption(obj, "crtMinLimit");
    expectOption(obj, "accelerated");
    expectOption(obj, "type");
    expectOption(obj, "num_threads");

    metadataPluginCommand azure("AZURE_BLOB",
                                {},
                                availableOptions({"account_url",
                                                  "container_name",
                                                  "connection_string",
                                                  "ca_bundle",
                                                  "num_threads"}),
                                {DRAM_SEG, OBJ_SEG});
    expectOption(azure, "account_url");
    expectOption(azure, "container_name");
    expectOption(azure, "connection_string");
    expectOption(azure, "ca_bundle");
    expectOption(azure, "num_threads");
}

TEST(BackendOptionInventoryTest, NetworkBackendsExposeOptions) {
    metadataPluginCommand gpunetio("GPUNETIO",
                                   {},
                                   availableOptions({"network_devices",
                                                     "oob_interface",
                                                     "gpu_devices",
                                                     "cuda_streams"}),
                                   {DRAM_SEG, VRAM_SEG});
    expectOption(gpunetio, "network_devices");
    expectOption(gpunetio, "oob_interface");
    expectOption(gpunetio, "gpu_devices");
    expectOption(gpunetio, "cuda_streams");

    metadataPluginCommand uccl("UCCL",
                               {},
                               availableOptions({"in_python", "num_cpus"}),
                               {DRAM_SEG, VRAM_SEG});
    expectOption(uccl, "in_python");
    expectOption(uccl, "num_cpus");

    metadataPluginCommand libfabric("LIBFABRIC",
                                    {},
                                    availableOptions({"striping_threshold",
                                                      "max_bw_per_dram_seg"}),
                                    {DRAM_SEG, VRAM_SEG});
    expectOption(libfabric, "striping_threshold");
    expectOption(libfabric, "max_bw_per_dram_seg");
}

TEST(BackendOptionInventoryTest, GusliExposesEngineAndBenchConfigInputs) {
    metadataPluginCommand command("GUSLI",
                                  {},
                                  availableOptions({"client_name",
                                                    "max_num_simultaneous_requests",
                                                    "config_file",
                                                    "device_byte_offsets",
                                                    "device_security"}),
                                  {BLK_SEG, DRAM_SEG});

    expectOption(command, "client_name");
    expectOption(command, "max_num_simultaneous_requests");
    expectOption(command, "config_file");
    expectOption(command, "device_byte_offsets");
    expectOption(command, "device_security");
}

} // namespace
} // namespace nixlbench
