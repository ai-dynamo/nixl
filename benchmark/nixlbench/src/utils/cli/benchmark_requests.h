/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_REQUESTS_H
#define NIXLBENCH_BENCHMARK_REQUESTS_H

#include <cstdint>
#include <string>

namespace nixlbench {

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
    std::string config_file;
    bool config_file_provided = false;
    std::string benchmark_group = "default";
    bool benchmark_group_provided = false;
    std::string runtime_type = "ETCD";
    bool runtime_type_provided = false;
    std::string worker_type = "nixl";
    bool worker_type_provided = false;
    std::string backend = "UCX";
    bool backend_provided = false;
    std::string initiator_seg_type = "DRAM";
    bool initiator_seg_type_provided = false;
    std::string target_seg_type = "DRAM";
    bool target_seg_type_provided = false;
    std::string scheme = "pairwise";
    bool scheme_provided = false;
    std::string mode = "SG";
    bool mode_provided = false;
    std::string op_type = "WRITE";
    bool op_type_provided = false;
    bool check_consistency = false;
    bool check_consistency_provided = false;
    uint64_t total_buffer_size = 8589934592ULL;
    bool total_buffer_size_provided = false;
    uint64_t start_block_size = 4096;
    bool start_block_size_provided = false;
    uint64_t max_block_size = 67108864;
    bool max_block_size_provided = false;
    uint64_t start_batch_size = 1;
    bool start_batch_size_provided = false;
    uint64_t max_batch_size = 1;
    bool max_batch_size_provided = false;
    int num_iter = 1000;
    bool num_iter_provided = false;
    bool recreate_xfer = false;
    bool recreate_xfer_provided = false;
    int large_blk_iter_ftr = 16;
    bool large_blk_iter_ftr_provided = false;
    int warmup_iter = 100;
    bool warmup_iter_provided = false;
    int num_threads = 1;
    bool num_threads_provided = false;
    int num_initiator_dev = 1;
    bool num_initiator_dev_provided = false;
    int num_target_dev = 1;
    bool num_target_dev_provided = false;
    bool enable_pt = false;
    bool enable_pt_provided = false;
    uint64_t progress_threads = 0;
    bool progress_threads_provided = false;
    bool enable_vmm = false;
    bool enable_vmm_provided = false;
    std::string filepath;
    bool filepath_provided = false;
    std::string filenames;
    bool filenames_provided = false;
    int num_files = 1;
    bool num_files_provided = false;
    bool storage_enable_direct = false;
    bool storage_enable_direct_provided = false;
    int gds_batch_pool_size = 32;
    bool gds_batch_pool_size_provided = false;
    int gds_batch_limit = 128;
    bool gds_batch_limit_provided = false;
    int gds_mt_num_threads = 1;
    bool gds_mt_num_threads_provided = false;
    std::string device_list = "all";
    bool device_list_provided = false;
    std::string etcd_endpoints;
    bool etcd_endpoints_provided = false;
    std::string posix_api_type = "AIO";
    bool posix_api_type_provided = false;
    int posix_ios_pool_size = 65536;
    bool posix_ios_pool_size_provided = false;
    int posix_kernel_queue_size = 256;
    bool posix_kernel_queue_size_provided = false;
    std::string gpunetio_device_list = "0";
    bool gpunetio_device_list_provided = false;
    std::string gpunetio_oob_list;
    bool gpunetio_oob_list_provided = false;
    std::string obj_access_key;
    bool obj_access_key_provided = false;
    std::string obj_secret_key;
    bool obj_secret_key_provided = false;
    std::string obj_session_token;
    bool obj_session_token_provided = false;
    std::string obj_bucket_name;
    bool obj_bucket_name_provided = false;
    std::string obj_scheme = "http";
    bool obj_scheme_provided = false;
    std::string obj_region = "eu-central-1";
    bool obj_region_provided = false;
    bool obj_use_virtual_addressing = false;
    bool obj_use_virtual_addressing_provided = false;
    std::string obj_endpoint_override;
    bool obj_endpoint_override_provided = false;
    std::string obj_req_checksum = "supported";
    bool obj_req_checksum_provided = false;
    std::string obj_ca_bundle;
    bool obj_ca_bundle_provided = false;
    uint64_t obj_crt_min_limit = 0;
    bool obj_crt_min_limit_provided = false;
    bool obj_accelerated_enable = false;
    bool obj_accelerated_enable_provided = false;
    std::string obj_accelerated_type;
    bool obj_accelerated_type_provided = false;
    std::string azure_blob_account_url;
    bool azure_blob_account_url_provided = false;
    std::string azure_blob_container_name;
    bool azure_blob_container_name_provided = false;
    std::string azure_blob_connection_string;
    bool azure_blob_connection_string_provided = false;
    int hf3fs_iopool_size = 64;
    bool hf3fs_iopool_size_provided = false;
    std::string gusli_client_name = "NIXLBench";
    bool gusli_client_name_provided = false;
    int gusli_max_simultaneous_requests = 32;
    bool gusli_max_simultaneous_requests_provided = false;
    std::string gusli_config_file;
    bool gusli_config_file_provided = false;
    std::string gusli_device_byte_offsets;
    bool gusli_device_byte_offsets_provided = false;
    std::string gusli_device_security;
    bool gusli_device_security_provided = false;
};

struct PosixPluginRequest {
    std::string storage_path;
    bool should_split_dir_per_thread = false;
    std::string mode = "aio";
    std::string filepath;
    bool filepath_provided = false;
    std::string filenames;
    bool filenames_provided = false;
    int num_files = 1;
    bool num_files_provided = false;
    bool storage_enable_direct = false;
    bool storage_enable_direct_provided = false;
    std::string posix_api_type = "AIO";
    bool posix_api_type_provided = false;
    int posix_ios_pool_size = 65536;
    bool posix_ios_pool_size_provided = false;
    int posix_kernel_queue_size = 256;
    bool posix_kernel_queue_size_provided = false;
    std::string api_type = "AIO";
    bool api_type_provided = false;
    bool enable_direct = false;
    bool enable_direct_provided = false;
};

struct ObjPluginRequest {
    std::string endpoint_url;
    std::string bucket_name;
    std::string obj_access_key;
    bool obj_access_key_provided = false;
    std::string obj_secret_key;
    bool obj_secret_key_provided = false;
    std::string obj_session_token;
    bool obj_session_token_provided = false;
    std::string obj_bucket_name;
    bool obj_bucket_name_provided = false;
    std::string obj_scheme = "http";
    bool obj_scheme_provided = false;
    std::string obj_region = "eu-central-1";
    bool obj_region_provided = false;
    bool obj_use_virtual_addressing = false;
    bool obj_use_virtual_addressing_provided = false;
    std::string obj_endpoint_override;
    bool obj_endpoint_override_provided = false;
    std::string obj_req_checksum = "supported";
    bool obj_req_checksum_provided = false;
    std::string obj_ca_bundle;
    bool obj_ca_bundle_provided = false;
    uint64_t obj_crt_min_limit = 0;
    bool obj_crt_min_limit_provided = false;
    bool obj_accelerated_enable = false;
    bool obj_accelerated_enable_provided = false;
    std::string obj_accelerated_type;
    bool obj_accelerated_type_provided = false;
};

struct GdsPluginRequest {
    std::string filepath;
    bool filepath_provided = false;
    std::string filenames;
    bool filenames_provided = false;
    int num_files = 1;
    bool num_files_provided = false;
    bool storage_enable_direct = false;
    bool storage_enable_direct_provided = false;
    int gds_batch_pool_size = 32;
    bool gds_batch_pool_size_provided = false;
    int gds_batch_limit = 128;
    bool gds_batch_limit_provided = false;
};

struct GdsMtPluginRequest {
    std::string filepath;
    bool filepath_provided = false;
    std::string filenames;
    bool filenames_provided = false;
    int num_files = 1;
    bool num_files_provided = false;
    bool storage_enable_direct = false;
    bool storage_enable_direct_provided = false;
    int gds_mt_num_threads = 1;
    bool gds_mt_num_threads_provided = false;
};

struct GpuNetIoPluginRequest {
    std::string gpunetio_device_list = "0";
    bool gpunetio_device_list_provided = false;
    std::string gpunetio_oob_list;
    bool gpunetio_oob_list_provided = false;
};

struct AzureBlobPluginRequest {
    std::string azure_blob_account_url;
    bool azure_blob_account_url_provided = false;
    std::string azure_blob_container_name;
    bool azure_blob_container_name_provided = false;
    std::string azure_blob_connection_string;
    bool azure_blob_connection_string_provided = false;
};

struct Hf3fsPluginRequest {
    std::string filepath;
    bool filepath_provided = false;
    std::string filenames;
    bool filenames_provided = false;
    int num_files = 1;
    bool num_files_provided = false;
    bool storage_enable_direct = false;
    bool storage_enable_direct_provided = false;
    int hf3fs_iopool_size = 64;
    bool hf3fs_iopool_size_provided = false;
};

struct GusliPluginRequest {
    std::string filepath;
    bool filepath_provided = false;
    std::string filenames;
    bool filenames_provided = false;
    int num_files = 1;
    bool num_files_provided = false;
    bool storage_enable_direct = false;
    bool storage_enable_direct_provided = false;
    std::string gusli_client_name = "NIXLBench";
    bool gusli_client_name_provided = false;
    int gusli_max_simultaneous_requests = 32;
    bool gusli_max_simultaneous_requests_provided = false;
    std::string gusli_config_file;
    bool gusli_config_file_provided = false;
    std::string gusli_device_byte_offsets;
    bool gusli_device_byte_offsets_provided = false;
    std::string gusli_device_security;
    bool gusli_device_security_provided = false;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_REQUESTS_H
