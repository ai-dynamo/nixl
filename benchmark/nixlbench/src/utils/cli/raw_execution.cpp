/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/raw_execution.h"

#include "benchmark_runner.h"
#include "utils/utils.h"

namespace nixlbench {
namespace {

void
applyRawRequestToConfig(const rawRequest &request, xferBenchConfig &config) {
    config.benchmark_group = request.benchmark_group.value;
    config.runtime_type = request.runtime_type.value;
    config.worker_type = request.worker_type.value;
    config.backend = request.backend.value;
    config.initiator_seg_type = request.initiator_seg_type.value;
    config.target_seg_type = request.target_seg_type.value;
    config.scheme = request.scheme.value;
    config.mode = request.mode.value;
    config.op_type = request.op_type.value;
    config.check_consistency = request.check_consistency.value;
    config.total_buffer_size = request.total_buffer_size.value;
    config.start_block_size = request.start_block_size.value;
    config.max_block_size = request.max_block_size.value;
    config.start_batch_size = request.start_batch_size.value;
    config.max_batch_size = request.max_batch_size.value;
    config.num_iter = request.num_iter.value;
    config.recreate_xfer = request.recreate_xfer.value;
    config.large_blk_iter_ftr = request.large_blk_iter_ftr.value;
    config.warmup_iter = request.warmup_iter.value;
    config.num_threads = request.num_threads.value;
    config.num_initiator_dev = request.num_initiator_dev.value;
    config.num_target_dev = request.num_target_dev.value;
    config.enable_pt = request.enable_pt.value;
    config.progress_threads = request.progress_threads.value;
    config.enable_vmm = request.enable_vmm.value;
    config.filepath = request.filepath.value;
    config.filenames = request.filenames.value;
    config.num_files = request.num_files.value;
    config.storage_enable_direct = request.storage_enable_direct.value;
    config.gds_batch_pool_size = request.gds_batch_pool_size.value;
    config.gds_batch_limit = request.gds_batch_limit.value;
    config.gds_mt_num_threads = request.gds_mt_num_threads.value;
    config.device_list = request.device_list.value;
    config.etcd_endpoints = request.etcd_endpoints.value;
    config.posix_api_type = request.posix_api_type.value;
    config.posix_ios_pool_size = request.posix_ios_pool_size.value;
    config.posix_kernel_queue_size = request.posix_kernel_queue_size.value;
    config.gpunetio_device_list = request.gpunetio_device_list.value;
    config.gpunetio_oob_list = request.gpunetio_oob_list.value;
    config.obj_access_key = request.obj_access_key.value;
    config.obj_secret_key = request.obj_secret_key.value;
    config.obj_session_token = request.obj_session_token.value;
    config.obj_bucket_name = request.obj_bucket_name.value;
    config.obj_scheme = request.obj_scheme.value;
    config.obj_region = request.obj_region.value;
    config.obj_use_virtual_addressing = request.obj_use_virtual_addressing.value;
    config.obj_endpoint_override = request.obj_endpoint_override.value;
    config.obj_req_checksum = request.obj_req_checksum.value;
    config.obj_ca_bundle = request.obj_ca_bundle.value;
    config.obj_crt_min_limit = request.obj_crt_min_limit.value;
    config.obj_accelerated_enable = request.obj_accelerated_enable.value;
    config.obj_accelerated_type = request.obj_accelerated_type.value;
    config.azure_blob_account_url = request.azure_blob_account_url.value;
    config.azure_blob_container_name = request.azure_blob_container_name.value;
    config.azure_blob_connection_string = request.azure_blob_connection_string.value;
    config.hf3fs_iopool_size = request.hf3fs_iopool_size.value;
    config.gusli_client_name = request.gusli_client_name.value;
    config.gusli_max_simultaneous_requests = request.gusli_max_simultaneous_requests.value;
    config.gusli_config_file = request.gusli_config_file.value;
    config.gusli_device_byte_offsets = request.gusli_device_byte_offsets.value;
    config.gusli_device_security = request.gusli_device_security.value;
}

} // namespace

int
runRawRequest(const rawRequest &request) {
    xferBenchConfig config;
    applyRawRequestToConfig(request, config);
    if (!validateRawConfigForRun(config)) {
        return 1;
    }
    return runBenchmark(config);
}

} // namespace nixlbench
