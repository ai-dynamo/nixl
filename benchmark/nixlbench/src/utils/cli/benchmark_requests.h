/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_REQUESTS_H
#define NIXLBENCH_BENCHMARK_REQUESTS_H

#include <cstdint>
#include <string>

namespace nixlbench {

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

enum class plugin_type_t {
    NONE,
    POSIX,
    OBJ,
    GDS,
    GDS_MT,
    GPUNETIO,
    AZURE_BLOB,
    HF3FS,
    GUSLI,
};

struct g3ScenarioRequest {
    std::string file_size;
    int parallel_threads = 1;
    std::string batch_size;
};

struct g4ScenarioRequest {
    std::string file_size;
    int num_kvs = 0;
    int parallel_threads = 1;
    std::string batch_size;
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
