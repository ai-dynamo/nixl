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
applyRawRequestToConfig(const RawRequest &request) {
    xferBenchConfig::benchmark_group = request.benchmark_group.value;
    xferBenchConfig::runtime_type = request.runtime_type.value;
    xferBenchConfig::worker_type = request.worker_type.value;
    xferBenchConfig::backend = request.backend.value;
    xferBenchConfig::initiator_seg_type = request.initiator_seg_type.value;
    xferBenchConfig::target_seg_type = request.target_seg_type.value;
    xferBenchConfig::scheme = request.scheme.value;
    xferBenchConfig::mode = request.mode.value;
    xferBenchConfig::op_type = request.op_type.value;
    xferBenchConfig::check_consistency = request.check_consistency.value;
    xferBenchConfig::total_buffer_size = request.total_buffer_size.value;
    xferBenchConfig::start_block_size = request.start_block_size.value;
    xferBenchConfig::max_block_size = request.max_block_size.value;
    xferBenchConfig::start_batch_size = request.start_batch_size.value;
    xferBenchConfig::max_batch_size = request.max_batch_size.value;
    xferBenchConfig::num_iter = request.num_iter.value;
    xferBenchConfig::recreate_xfer = request.recreate_xfer.value;
    xferBenchConfig::large_blk_iter_ftr = request.large_blk_iter_ftr.value;
    xferBenchConfig::warmup_iter = request.warmup_iter.value;
    xferBenchConfig::num_threads = request.num_threads.value;
    xferBenchConfig::num_initiator_dev = request.num_initiator_dev.value;
    xferBenchConfig::num_target_dev = request.num_target_dev.value;
    xferBenchConfig::enable_pt = request.enable_pt.value;
    xferBenchConfig::progress_threads = request.progress_threads.value;
    xferBenchConfig::enable_vmm = request.enable_vmm.value;
    xferBenchConfig::filepath = request.filepath.value;
    xferBenchConfig::filenames = request.filenames.value;
    xferBenchConfig::num_files = request.num_files.value;
    xferBenchConfig::storage_enable_direct = request.storage_enable_direct.value;
    xferBenchConfig::gds_batch_pool_size = request.gds_batch_pool_size.value;
    xferBenchConfig::gds_batch_limit = request.gds_batch_limit.value;
    xferBenchConfig::gds_mt_num_threads = request.gds_mt_num_threads.value;
    xferBenchConfig::device_list = request.device_list.value;
    xferBenchConfig::etcd_endpoints = request.etcd_endpoints.value;
    xferBenchConfig::posix_api_type = request.posix_api_type.value;
    xferBenchConfig::posix_ios_pool_size = request.posix_ios_pool_size.value;
    xferBenchConfig::posix_kernel_queue_size = request.posix_kernel_queue_size.value;
    xferBenchConfig::gpunetio_device_list = request.gpunetio_device_list.value;
    xferBenchConfig::gpunetio_oob_list = request.gpunetio_oob_list.value;
    xferBenchConfig::obj_access_key = request.obj_access_key.value;
    xferBenchConfig::obj_secret_key = request.obj_secret_key.value;
    xferBenchConfig::obj_session_token = request.obj_session_token.value;
    xferBenchConfig::obj_bucket_name = request.obj_bucket_name.value;
    xferBenchConfig::obj_scheme = request.obj_scheme.value;
    xferBenchConfig::obj_region = request.obj_region.value;
    xferBenchConfig::obj_use_virtual_addressing = request.obj_use_virtual_addressing.value;
    xferBenchConfig::obj_endpoint_override = request.obj_endpoint_override.value;
    xferBenchConfig::obj_req_checksum = request.obj_req_checksum.value;
    xferBenchConfig::obj_ca_bundle = request.obj_ca_bundle.value;
    xferBenchConfig::obj_crt_min_limit = request.obj_crt_min_limit.value;
    xferBenchConfig::obj_accelerated_enable = request.obj_accelerated_enable.value;
    xferBenchConfig::obj_accelerated_type = request.obj_accelerated_type.value;
    xferBenchConfig::azure_blob_account_url = request.azure_blob_account_url.value;
    xferBenchConfig::azure_blob_container_name = request.azure_blob_container_name.value;
    xferBenchConfig::azure_blob_connection_string = request.azure_blob_connection_string.value;
    xferBenchConfig::hf3fs_iopool_size = request.hf3fs_iopool_size.value;
    xferBenchConfig::gusli_client_name = request.gusli_client_name.value;
    xferBenchConfig::gusli_max_simultaneous_requests = request.gusli_max_simultaneous_requests.value;
    xferBenchConfig::gusli_config_file = request.gusli_config_file.value;
    xferBenchConfig::gusli_device_byte_offsets = request.gusli_device_byte_offsets.value;
    xferBenchConfig::gusli_device_security = request.gusli_device_security.value;
}

bool
validateRawConfig() {
#if defined(HAVE_CUDA) && !defined(HAVE_CUDA_FABRIC)
    if (xferBenchConfig::enable_vmm) {
        std::cerr << "VMM is not supported in CUDA version " << CUDA_VERSION << std::endl;
        return false;
    }
#endif

    if (xferBenchConfig::backend == XFERBENCH_BACKEND_POSIX &&
        xferBenchConfig::posix_api_type != XFERBENCH_POSIX_API_AIO &&
        xferBenchConfig::posix_api_type != XFERBENCH_POSIX_API_URING &&
        xferBenchConfig::posix_api_type != XFERBENCH_POSIX_API_POSIXAIO) {
        std::cerr << "Invalid POSIX API type: " << xferBenchConfig::posix_api_type
                  << ". Must be one of [AIO, URING, POSIXAIO]" << std::endl;
        return false;
    }

    if (xferBenchConfig::backend == XFERBENCH_BACKEND_OBJ) {
        if (xferBenchConfig::obj_scheme != XFERBENCH_OBJ_SCHEME_HTTP &&
            xferBenchConfig::obj_scheme != XFERBENCH_OBJ_SCHEME_HTTPS) {
            std::cerr << "Invalid OBJ S3 scheme: " << xferBenchConfig::obj_scheme
                      << ". Must be one of [http, https]" << std::endl;
            return false;
        }
        if (xferBenchConfig::obj_req_checksum != XFERBENCH_OBJ_REQ_CHECKSUM_SUPPORTED &&
            xferBenchConfig::obj_req_checksum != XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED) {
            std::cerr << "Invalid OBJ S3 required checksum: " << xferBenchConfig::obj_req_checksum
                      << ". Must be one of [supported, required]" << std::endl;
            return false;
        }
    }

    if (!xferBenchConfig::recreate_xfer && XFERBENCH_BACKEND_GUSLI == xferBenchConfig::backend) {
        std::cout << "GUSLI backend requires per-iteration request creation due to library bug."
                  << " Setting recreate_xfer to true." << std::endl;
        xferBenchConfig::recreate_xfer = true;
    }

    if (!xferBenchConfig::isStorageBackend() && xferBenchConfig::etcd_endpoints.empty()) {
        xferBenchConfig::etcd_endpoints = "http://localhost:2379";
        std::cout << "Using default ETCD endpoint for non-storage backend: "
                  << xferBenchConfig::etcd_endpoints << std::endl;
    }

    if (xferBenchConfig::worker_type == XFERBENCH_WORKER_NVSHMEM) {
        if (!((XFERBENCH_SEG_TYPE_VRAM == xferBenchConfig::initiator_seg_type) &&
              (XFERBENCH_SEG_TYPE_VRAM == xferBenchConfig::target_seg_type) &&
              (1 == xferBenchConfig::num_threads) &&
              (1 == xferBenchConfig::num_initiator_dev) &&
              (1 == xferBenchConfig::num_target_dev) &&
              (XFERBENCH_SCHEME_PAIRWISE == xferBenchConfig::scheme))) {
            std::cerr << "Unsupported configuration for NVSHMEM worker" << std::endl;
            return false;
        }
    }

    if ((xferBenchConfig::max_block_size * xferBenchConfig::max_batch_size) >
        (xferBenchConfig::total_buffer_size / xferBenchConfig::num_initiator_dev)) {
        std::cerr << "Incorrect buffer size configuration for Initiator"
                  << "(max_block_size * max_batch_size) is > "
                  << "(total_buffer_size / num_initiator_dev)" << std::endl;
        return false;
    }
    if ((xferBenchConfig::max_block_size * xferBenchConfig::max_batch_size) >
        (xferBenchConfig::total_buffer_size / xferBenchConfig::num_target_dev)) {
        std::cerr << "Incorrect buffer size configuration for Target"
                  << "(max_block_size * max_batch_size) is > "
                  << "(total_buffer_size / num_initiator_dev)" << std::endl;
        return false;
    }
    if ((xferBenchConfig::max_block_size * xferBenchConfig::max_batch_size) >
        (xferBenchConfig::total_buffer_size / xferBenchConfig::num_threads)) {
        std::cerr << "Incorrect buffer size configuration "
                  << "(max_block_size * max_batch_size) "
                  << "(" << (xferBenchConfig::max_block_size * xferBenchConfig::max_batch_size)
                  << ")"
                  << " is > (total_buffer_size / num_threads) ("
                  << (xferBenchConfig::total_buffer_size / xferBenchConfig::num_threads) << ")"
                  << std::endl;
        return false;
    }

    if (xferBenchConfig::large_blk_iter_ftr <= 0) {
        std::cerr << "iter_factor must be greater than 0" << std::endl;
        return false;
    }

    int partition = (xferBenchConfig::num_threads * xferBenchConfig::large_blk_iter_ftr);
    if (xferBenchConfig::num_iter % partition) {
        xferBenchConfig::num_iter += partition - (xferBenchConfig::num_iter % partition);
        std::cout << "WARNING: Adjusting num_iter to " << xferBenchConfig::num_iter
                  << " to allow equal distribution to " << xferBenchConfig::num_threads
                  << " threads" << std::endl;
    }
    if (xferBenchConfig::warmup_iter % partition) {
        xferBenchConfig::warmup_iter += partition - (xferBenchConfig::warmup_iter % partition);
        std::cout << "WARNING: Adjusting warmup_iter to " << xferBenchConfig::warmup_iter
                  << " to allow equal distribution to " << xferBenchConfig::num_threads
                  << " threads" << std::endl;
    }
    partition = (xferBenchConfig::num_initiator_dev * xferBenchConfig::num_threads);
    if (xferBenchConfig::total_buffer_size % partition) {
        std::cerr << "Total_buffer_size must be divisible by the product of num_threads and "
                     "num_initiator_dev"
                  << ", next such value is "
                  << xferBenchConfig::total_buffer_size + partition -
                         (xferBenchConfig::total_buffer_size % partition)
                  << std::endl;
        return false;
    }
    partition = (xferBenchConfig::num_target_dev * xferBenchConfig::num_threads);
    if (xferBenchConfig::total_buffer_size % partition) {
        std::cerr << "Total_buffer_size must be divisible by the product of num_threads and "
                     "num_target_dev"
                  << ", next such value is "
                  << xferBenchConfig::total_buffer_size + partition -
                         (xferBenchConfig::total_buffer_size % partition)
                  << std::endl;
        return false;
    }

    return true;
}

} // namespace

int
runRawRequest(const RawRequest &request) {
    applyRawRequestToConfig(request);
    if (!validateRawConfig()) {
        return 1;
    }
    return runBenchmarkWithCurrentConfig();
}

} // namespace nixlbench
