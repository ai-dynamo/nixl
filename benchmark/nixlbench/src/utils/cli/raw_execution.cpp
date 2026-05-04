/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/raw_execution.h"

#include "benchmark_runner.h"
#include "utils/utils.h"

#include <iostream>

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

bool
validateRawConfig(xferBenchConfig &config) {
#if defined(HAVE_CUDA) && !defined(HAVE_CUDA_FABRIC)
    if (config.enable_vmm) {
        std::cerr << "VMM is not supported in CUDA version " << CUDA_VERSION << std::endl;
        return false;
    }
#endif

    if (config.backend == XFERBENCH_BACKEND_POSIX &&
        config.posix_api_type != XFERBENCH_POSIX_API_AIO &&
        config.posix_api_type != XFERBENCH_POSIX_API_URING &&
        config.posix_api_type != XFERBENCH_POSIX_API_POSIXAIO) {
        std::cerr << "Invalid POSIX API type: " << config.posix_api_type
                  << ". Must be one of [AIO, URING, POSIXAIO]" << std::endl;
        return false;
    }

    if (config.backend == XFERBENCH_BACKEND_OBJ) {
        if (config.obj_scheme != XFERBENCH_OBJ_SCHEME_HTTP &&
            config.obj_scheme != XFERBENCH_OBJ_SCHEME_HTTPS) {
            std::cerr << "Invalid OBJ S3 scheme: " << config.obj_scheme
                      << ". Must be one of [http, https]" << std::endl;
            return false;
        }
        if (config.obj_req_checksum != XFERBENCH_OBJ_REQ_CHECKSUM_SUPPORTED &&
            config.obj_req_checksum != XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED) {
            std::cerr << "Invalid OBJ S3 required checksum: " << config.obj_req_checksum
                      << ". Must be one of [supported, required]" << std::endl;
            return false;
        }
    }

    if (!config.recreate_xfer && XFERBENCH_BACKEND_GUSLI == config.backend) {
        std::cout << "GUSLI backend requires per-iteration request creation due to library bug."
                  << " Setting recreate_xfer to true." << std::endl;
        config.recreate_xfer = true;
    }

    if (!config.isStorageBackend() && config.etcd_endpoints.empty()) {
        config.etcd_endpoints = "http://localhost:2379";
        std::cout << "Using default ETCD endpoint for non-storage backend: "
                  << config.etcd_endpoints << std::endl;
    }

    if (config.worker_type == XFERBENCH_WORKER_NVSHMEM) {
        if (!((XFERBENCH_SEG_TYPE_VRAM == config.initiator_seg_type) &&
              (XFERBENCH_SEG_TYPE_VRAM == config.target_seg_type) &&
              (1 == config.num_threads) &&
              (1 == config.num_initiator_dev) &&
              (1 == config.num_target_dev) &&
              (XFERBENCH_SCHEME_PAIRWISE == config.scheme))) {
            std::cerr << "Unsupported configuration for NVSHMEM worker" << std::endl;
            return false;
        }
    }

    if ((config.max_block_size * config.max_batch_size) >
        (config.total_buffer_size / config.num_initiator_dev)) {
        std::cerr << "Incorrect buffer size configuration for Initiator"
                  << "(max_block_size * max_batch_size) is > "
                  << "(total_buffer_size / num_initiator_dev)" << std::endl;
        return false;
    }
    if ((config.max_block_size * config.max_batch_size) >
        (config.total_buffer_size / config.num_target_dev)) {
        std::cerr << "Incorrect buffer size configuration for Target"
                  << "(max_block_size * max_batch_size) is > "
                  << "(total_buffer_size / num_initiator_dev)" << std::endl;
        return false;
    }
    if ((config.max_block_size * config.max_batch_size) >
        (config.total_buffer_size / config.num_threads)) {
        std::cerr << "Incorrect buffer size configuration "
                  << "(max_block_size * max_batch_size) "
                  << "(" << (config.max_block_size * config.max_batch_size)
                  << ")"
                  << " is > (total_buffer_size / num_threads) ("
                  << (config.total_buffer_size / config.num_threads) << ")"
                  << std::endl;
        return false;
    }

    if (config.large_blk_iter_ftr <= 0) {
        std::cerr << "iter_factor must be greater than 0" << std::endl;
        return false;
    }

    int partition = (config.num_threads * config.large_blk_iter_ftr);
    if (config.num_iter % partition) {
        config.num_iter += partition - (config.num_iter % partition);
        std::cout << "WARNING: Adjusting num_iter to " << config.num_iter
                  << " to allow equal distribution to " << config.num_threads
                  << " threads" << std::endl;
    }
    if (config.warmup_iter % partition) {
        config.warmup_iter += partition - (config.warmup_iter % partition);
        std::cout << "WARNING: Adjusting warmup_iter to " << config.warmup_iter
                  << " to allow equal distribution to " << config.num_threads
                  << " threads" << std::endl;
    }
    partition = (config.num_initiator_dev * config.num_threads);
    if (config.total_buffer_size % partition) {
        std::cerr << "Total_buffer_size must be divisible by the product of num_threads and "
                     "num_initiator_dev"
                  << ", next such value is "
                  << config.total_buffer_size + partition - (config.total_buffer_size % partition)
                  << std::endl;
        return false;
    }
    partition = (config.num_target_dev * config.num_threads);
    if (config.total_buffer_size % partition) {
        std::cerr << "Total_buffer_size must be divisible by the product of num_threads and "
                     "num_target_dev"
                  << ", next such value is "
                  << config.total_buffer_size + partition - (config.total_buffer_size % partition)
                  << std::endl;
        return false;
    }

    return true;
}

} // namespace

int
runRawRequest(const rawRequest &request) {
    xferBenchConfig config;
    applyRawRequestToConfig(request, config);
    if (!validateRawConfig(config)) {
        return 1;
    }
    return runBenchmark(config);
}

} // namespace nixlbench
