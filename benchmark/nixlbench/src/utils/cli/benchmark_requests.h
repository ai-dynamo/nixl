/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_REQUESTS_H
#define NIXLBENCH_BENCHMARK_REQUESTS_H

#include "nixl_types.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nixlbench {

using request_key_value_pairs_t = std::vector<std::pair<std::string, std::string>>;

template <typename T>
struct providedValue {
    T value{};
    bool isProvided = false;

    providedValue() = default;
    explicit providedValue(const T &default_value) : value(default_value) {}

    T *valuePtr() { return &value; }
    bool *providedPtr() { return &isProvided; }
    bool wasProvided() const { return isProvided; }
    void setProvided(const T &new_value) {
        value = new_value;
        isProvided = true;
    }
};

enum class scenario_type_t {
    NONE,
    RAW,
    G3,
    G4,
};

// enum class plugin_type_t {
//     NONE,
//     POSIX,
//     OBJ,
//     GDS,
//     GDS_MT,
//     GPUNETIO,
//     AZURE_BLOB,
//     HF3FS,
//     GUSLI,
// };

struct metadataPluginOptionValue {
    std::string value;
    bool boolValue = false;
    bool isProvided = false;
    bool isFlag = false;
};

using metadata_plugin_option_map_t = std::unordered_map<std::string, metadataPluginOptionValue>;

struct fileWorkloadRequest {
    providedValue<std::string> filepath;
    providedValue<std::string> filenames;
    providedValue<int> num_files{1};
    providedValue<bool> enable_direct{false};
};

struct g3ScenarioRequest {
    std::string file_size = "1GB";
    int parallel_threads = 1;
    uint64_t block_size_bytes = 4096;
    uint64_t batch_size = 1;
    std::string action_mode = "read";
    bool randomized_read_location = true;

    request_key_value_pairs_t
    toKeyValuePairs() const {
        return {{"file_size", file_size},
                {"parallel_threads", std::to_string(parallel_threads)},
                {"block_size_bytes", std::to_string(block_size_bytes)},
                {"batch_size", std::to_string(batch_size)},
                {"action_mode", action_mode},
                {"randomized_read_location", randomized_read_location ? "true" : "false"}};
    }
};

struct g4ScenarioRequest {
    std::string file_size;
    int num_kvs = 0;
    int parallel_threads = 1;
    std::string batch_size;

    request_key_value_pairs_t
    toKeyValuePairs() const {
        return {{"file_size", file_size},
                {"num_kvs", std::to_string(num_kvs)},
                {"parallel_threads", std::to_string(parallel_threads)},
                {"batch_size", batch_size}};
    }
};

struct rawRequest {
    providedValue<std::string> config_file;
    providedValue<std::string> benchmark_group{"default"};
    providedValue<std::string> runtime_type{"ETCD"};
    providedValue<std::string> worker_type{"nixl"};
    providedValue<std::string> backend{"UCX"};
    providedValue<std::string> initiator_seg_type{"DRAM"};
    providedValue<std::string> target_seg_type{"DRAM"};
    providedValue<std::string> scheme{"pairwise"};
    providedValue<std::string> mode{"SG"};
    providedValue<std::string> op_type{"WRITE"};
    providedValue<bool> check_consistency{false};
    providedValue<uint64_t> total_buffer_size{8589934592ULL};
    providedValue<uint64_t> start_block_size{4096};
    providedValue<uint64_t> max_block_size{67108864};
    providedValue<uint64_t> start_batch_size{1};
    providedValue<uint64_t> max_batch_size{1};
    providedValue<int> num_iter{1000};
    providedValue<bool> recreate_xfer{false};
    providedValue<int> large_blk_iter_ftr{16};
    providedValue<int> warmup_iter{100};
    providedValue<int> num_threads{1};
    providedValue<int> num_initiator_dev{1};
    providedValue<int> num_target_dev{1};
    providedValue<bool> enable_pt{false};
    providedValue<uint64_t> progress_threads{0};
    providedValue<bool> enable_vmm{false};
    providedValue<std::string> filepath;
    providedValue<std::string> filenames;
    providedValue<int> num_files{1};
    providedValue<bool> storage_enable_direct{false};
    providedValue<int> gds_batch_pool_size{32};
    providedValue<int> gds_batch_limit{128};
    providedValue<int> gds_mt_num_threads{1};
    providedValue<std::string> device_list{"all"};
    providedValue<std::string> etcd_endpoints;
    providedValue<std::string> posix_api_type{"AIO"};
    providedValue<int> posix_ios_pool_size{65536};
    providedValue<int> posix_kernel_queue_size{256};
    providedValue<std::string> gpunetio_device_list{"0"};
    providedValue<std::string> gpunetio_oob_list;
    providedValue<std::string> obj_access_key;
    providedValue<std::string> obj_secret_key;
    providedValue<std::string> obj_session_token;
    providedValue<std::string> obj_bucket_name;
    providedValue<std::string> obj_scheme{"http"};
    providedValue<std::string> obj_region{"eu-central-1"};
    providedValue<bool> obj_use_virtual_addressing{false};
    providedValue<std::string> obj_endpoint_override;
    providedValue<std::string> obj_req_checksum{"supported"};
    providedValue<std::string> obj_ca_bundle;
    providedValue<uint64_t> obj_crt_min_limit{0};
    providedValue<bool> obj_accelerated_enable{false};
    providedValue<std::string> obj_accelerated_type;
    providedValue<std::string> azure_blob_account_url;
    providedValue<std::string> azure_blob_container_name;
    providedValue<std::string> azure_blob_connection_string;
    providedValue<int> hf3fs_iopool_size{64};
    providedValue<std::string> gusli_client_name{"NIXLBench"};
    providedValue<int> gusli_max_simultaneous_requests{32};
    providedValue<std::string> gusli_config_file;
    providedValue<std::string> gusli_device_byte_offsets;
    providedValue<std::string> gusli_device_security;
    nixlBackendPluginCapabilities backend_capabilities{};
    nixl_mem_list_t backend_memory_types;
    metadata_plugin_option_map_t backend_options;

    request_key_value_pairs_t
    toKeyValuePairs() const {
        return {{"config_file", config_file.value},
                {"benchmark_group", benchmark_group.value},
                {"runtime_type", runtime_type.value},
                {"worker_type", worker_type.value},
                {"backend", backend.value},
                {"initiator_seg_type", initiator_seg_type.value},
                {"target_seg_type", target_seg_type.value},
                {"scheme", scheme.value},
                {"mode", mode.value},
                {"op_type", op_type.value},
                {"check_consistency", check_consistency.value ? "true" : "false"},
                {"total_buffer_size", std::to_string(total_buffer_size.value)},
                {"start_block_size", std::to_string(start_block_size.value)},
                {"max_block_size", std::to_string(max_block_size.value)},
                {"start_batch_size", std::to_string(start_batch_size.value)},
                {"max_batch_size", std::to_string(max_batch_size.value)},
                {"num_iter", std::to_string(num_iter.value)},
                {"recreate_xfer", recreate_xfer.value ? "true" : "false"},
                {"large_blk_iter_ftr", std::to_string(large_blk_iter_ftr.value)},
                {"warmup_iter", std::to_string(warmup_iter.value)},
                {"num_threads", std::to_string(num_threads.value)},
                {"num_initiator_dev", std::to_string(num_initiator_dev.value)},
                {"num_target_dev", std::to_string(num_target_dev.value)},
                {"enable_pt", enable_pt.value ? "true" : "false"},
                {"progress_threads", std::to_string(progress_threads.value)},
                {"enable_vmm", enable_vmm.value ? "true" : "false"},
                {"filepath", filepath.value},
                {"filenames", filenames.value},
                {"num_files", std::to_string(num_files.value)},
                {"storage_enable_direct", storage_enable_direct.value ? "true" : "false"},
                {"gds_batch_pool_size", std::to_string(gds_batch_pool_size.value)},
                {"gds_batch_limit", std::to_string(gds_batch_limit.value)},
                {"gds_mt_num_threads", std::to_string(gds_mt_num_threads.value)},
                {"device_list", device_list.value},
                {"etcd_endpoints", etcd_endpoints.value},
                {"posix_api_type", posix_api_type.value},
                {"posix_ios_pool_size", std::to_string(posix_ios_pool_size.value)},
                {"posix_kernel_queue_size", std::to_string(posix_kernel_queue_size.value)},
                {"gpunetio_device_list", gpunetio_device_list.value},
                {"gpunetio_oob_list", gpunetio_oob_list.value},
                {"obj_access_key", obj_access_key.value},
                {"obj_secret_key", obj_secret_key.value},
                {"obj_session_token", obj_session_token.value},
                {"obj_bucket_name", obj_bucket_name.value},
                {"obj_scheme", obj_scheme.value},
                {"obj_region", obj_region.value},
                {"obj_use_virtual_addressing",
                 obj_use_virtual_addressing.value ? "true" : "false"},
                {"obj_endpoint_override", obj_endpoint_override.value},
                {"obj_req_checksum", obj_req_checksum.value},
                {"obj_ca_bundle", obj_ca_bundle.value},
                {"obj_crt_min_limit", std::to_string(obj_crt_min_limit.value)},
                {"obj_accelerated_enable", obj_accelerated_enable.value ? "true" : "false"},
                {"obj_accelerated_type", obj_accelerated_type.value},
                {"azure_blob_account_url", azure_blob_account_url.value},
                {"azure_blob_container_name", azure_blob_container_name.value},
                {"azure_blob_connection_string", azure_blob_connection_string.value},
                {"hf3fs_iopool_size", std::to_string(hf3fs_iopool_size.value)},
                {"gusli_client_name", gusli_client_name.value},
                {"gusli_max_simultaneous_requests",
                 std::to_string(gusli_max_simultaneous_requests.value)},
                {"gusli_config_file", gusli_config_file.value},
                {"gusli_device_byte_offsets", gusli_device_byte_offsets.value},
                {"gusli_device_security", gusli_device_security.value}};
    }
};

struct posixPluginRequest {
    bool should_split_dir_per_thread = false;
    providedValue<std::string> filepath;
    providedValue<std::string> filenames;
    providedValue<int> num_files{1};
    providedValue<bool> storage_enable_direct{false};
    providedValue<std::string> api_type{"AIO"};
    providedValue<int> ios_pool_size{65536};
    providedValue<int> kernel_queue_size{256};
    providedValue<bool> enable_direct{false};
};

struct objPluginRequest {
    providedValue<std::string> access_key;
    providedValue<std::string> secret_key;
    providedValue<std::string> session_token;
    providedValue<std::string> bucket_name;
    providedValue<std::string> scheme{"http"};
    providedValue<std::string> region{"eu-central-1"};
    providedValue<bool> use_virtual_addressing{false};
    providedValue<std::string> endpoint_override;
    providedValue<std::string> req_checksum{"supported"};
    providedValue<std::string> ca_bundle;
    providedValue<uint64_t> crt_min_limit{0};
    providedValue<bool> accelerated_enable{false};
    providedValue<std::string> accelerated_type;
};

struct gdsPluginRequest {
    providedValue<std::string> filepath;
    providedValue<std::string> filenames;
    providedValue<int> num_files{1};
    providedValue<bool> storage_enable_direct{false};
    providedValue<int> batch_pool_size{32};
    providedValue<int> batch_limit{128};
};

struct gdsMtPluginRequest {
    providedValue<std::string> filepath;
    providedValue<std::string> filenames;
    providedValue<int> num_files{1};
    providedValue<bool> storage_enable_direct{false};
    providedValue<int> num_threads{1};
};

struct gpunetioPluginRequest {
    providedValue<std::string> device_list{"0"};
    providedValue<std::string> oob_list;
};

struct azureBlobPluginRequest {
    providedValue<std::string> blob_account_url;
    providedValue<std::string> blob_container_name;
    providedValue<std::string> blob_connection_string;
};

struct hf3fsPluginRequest {
    providedValue<std::string> filepath;
    providedValue<std::string> filenames;
    providedValue<int> num_files{1};
    providedValue<bool> storage_enable_direct{false};
    providedValue<int> iopool_size{64};
};

struct gusliPluginRequest {
    providedValue<std::string> filepath;
    providedValue<std::string> filenames;
    providedValue<int> num_files{1};
    providedValue<bool> storage_enable_direct{false};
    providedValue<std::string> client_name{"NIXLBench"};
    providedValue<int> max_simultaneous_requests{32};
    providedValue<std::string> config_file;
    providedValue<std::string> device_byte_offsets;
    providedValue<std::string> device_security;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_REQUESTS_H
