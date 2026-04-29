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
struct Provided {
    T value{};
    bool provided = false;

    Provided() = default;
    explicit Provided(const T &default_value) : value(default_value) {}

    T *valuePtr() { return &value; }
    bool *providedPtr() { return &provided; }
    bool wasProvided() const { return provided; }
    void setProvided(const T &new_value) {
        value = new_value;
        provided = true;
    }
};

enum class ScenarioType {
    None,
    Raw,
    G3,
    G4,
};

enum class PluginType {
    None,
    Posix,
    Obj,
    Gds,
    GdsMt,
    GpuNetIo,
    AzureBlob,
    Hf3fs,
    Gusli,
};

struct G3ScenarioRequest {
    std::string file_size;
    int parallel_threads = 1;
    std::string batch_size;
};

struct G4ScenarioRequest {
    std::string file_size;
    int num_kvs = 0;
    int parallel_threads = 1;
    std::string batch_size;
};

struct RawRequest {
    Provided<std::string> config_file;
    Provided<std::string> benchmark_group{"default"};
    Provided<std::string> runtime_type{"ETCD"};
    Provided<std::string> worker_type{"nixl"};
    Provided<std::string> backend{"UCX"};
    Provided<std::string> initiator_seg_type{"DRAM"};
    Provided<std::string> target_seg_type{"DRAM"};
    Provided<std::string> scheme{"pairwise"};
    Provided<std::string> mode{"SG"};
    Provided<std::string> op_type{"WRITE"};
    Provided<bool> check_consistency{false};
    Provided<uint64_t> total_buffer_size{8589934592ULL};
    Provided<uint64_t> start_block_size{4096};
    Provided<uint64_t> max_block_size{67108864};
    Provided<uint64_t> start_batch_size{1};
    Provided<uint64_t> max_batch_size{1};
    Provided<int> num_iter{1000};
    Provided<bool> recreate_xfer{false};
    Provided<int> large_blk_iter_ftr{16};
    Provided<int> warmup_iter{100};
    Provided<int> num_threads{1};
    Provided<int> num_initiator_dev{1};
    Provided<int> num_target_dev{1};
    Provided<bool> enable_pt{false};
    Provided<uint64_t> progress_threads{0};
    Provided<bool> enable_vmm{false};
    Provided<std::string> filepath;
    Provided<std::string> filenames;
    Provided<int> num_files{1};
    Provided<bool> storage_enable_direct{false};
    Provided<int> gds_batch_pool_size{32};
    Provided<int> gds_batch_limit{128};
    Provided<int> gds_mt_num_threads{1};
    Provided<std::string> device_list{"all"};
    Provided<std::string> etcd_endpoints;
    Provided<std::string> posix_api_type{"AIO"};
    Provided<int> posix_ios_pool_size{65536};
    Provided<int> posix_kernel_queue_size{256};
    Provided<std::string> gpunetio_device_list{"0"};
    Provided<std::string> gpunetio_oob_list;
    Provided<std::string> obj_access_key;
    Provided<std::string> obj_secret_key;
    Provided<std::string> obj_session_token;
    Provided<std::string> obj_bucket_name;
    Provided<std::string> obj_scheme{"http"};
    Provided<std::string> obj_region{"eu-central-1"};
    Provided<bool> obj_use_virtual_addressing{false};
    Provided<std::string> obj_endpoint_override;
    Provided<std::string> obj_req_checksum{"supported"};
    Provided<std::string> obj_ca_bundle;
    Provided<uint64_t> obj_crt_min_limit{0};
    Provided<bool> obj_accelerated_enable{false};
    Provided<std::string> obj_accelerated_type;
    Provided<std::string> azure_blob_account_url;
    Provided<std::string> azure_blob_container_name;
    Provided<std::string> azure_blob_connection_string;
    Provided<int> hf3fs_iopool_size{64};
    Provided<std::string> gusli_client_name{"NIXLBench"};
    Provided<int> gusli_max_simultaneous_requests{32};
    Provided<std::string> gusli_config_file;
    Provided<std::string> gusli_device_byte_offsets;
    Provided<std::string> gusli_device_security;
};

struct PosixPluginRequest {
    std::string storage_path;
    bool should_split_dir_per_thread = false;
    std::string mode = "aio";
    Provided<std::string> filepath;
    Provided<std::string> filenames;
    Provided<int> num_files{1};
    Provided<bool> storage_enable_direct{false};
    Provided<std::string> posix_api_type{"AIO"};
    Provided<int> posix_ios_pool_size{65536};
    Provided<int> posix_kernel_queue_size{256};
    Provided<std::string> api_type{"AIO"};
    Provided<bool> enable_direct{false};
};

struct ObjPluginRequest {
    std::string endpoint_url;
    std::string bucket_name;
    Provided<std::string> obj_access_key;
    Provided<std::string> obj_secret_key;
    Provided<std::string> obj_session_token;
    Provided<std::string> obj_bucket_name;
    Provided<std::string> obj_scheme{"http"};
    Provided<std::string> obj_region{"eu-central-1"};
    Provided<bool> obj_use_virtual_addressing{false};
    Provided<std::string> obj_endpoint_override;
    Provided<std::string> obj_req_checksum{"supported"};
    Provided<std::string> obj_ca_bundle;
    Provided<uint64_t> obj_crt_min_limit{0};
    Provided<bool> obj_accelerated_enable{false};
    Provided<std::string> obj_accelerated_type;
};

struct GdsPluginRequest {
    Provided<std::string> filepath;
    Provided<std::string> filenames;
    Provided<int> num_files{1};
    Provided<bool> storage_enable_direct{false};
    Provided<int> gds_batch_pool_size{32};
    Provided<int> gds_batch_limit{128};
};

struct GdsMtPluginRequest {
    Provided<std::string> filepath;
    Provided<std::string> filenames;
    Provided<int> num_files{1};
    Provided<bool> storage_enable_direct{false};
    Provided<int> gds_mt_num_threads{1};
};

struct GpuNetIoPluginRequest {
    Provided<std::string> gpunetio_device_list{"0"};
    Provided<std::string> gpunetio_oob_list;
};

struct AzureBlobPluginRequest {
    Provided<std::string> azure_blob_account_url;
    Provided<std::string> azure_blob_container_name;
    Provided<std::string> azure_blob_connection_string;
};

struct Hf3fsPluginRequest {
    Provided<std::string> filepath;
    Provided<std::string> filenames;
    Provided<int> num_files{1};
    Provided<bool> storage_enable_direct{false};
    Provided<int> hf3fs_iopool_size{64};
};

struct GusliPluginRequest {
    Provided<std::string> filepath;
    Provided<std::string> filenames;
    Provided<int> num_files{1};
    Provided<bool> storage_enable_direct{false};
    Provided<std::string> gusli_client_name{"NIXLBench"};
    Provided<int> gusli_max_simultaneous_requests{32};
    Provided<std::string> gusli_config_file;
    Provided<std::string> gusli_device_byte_offsets;
    Provided<std::string> gusli_device_security;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_REQUESTS_H
