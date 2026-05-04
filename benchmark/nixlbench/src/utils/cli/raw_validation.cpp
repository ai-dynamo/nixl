/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/raw_execution.h"

#include "utils/utils.h"

#include <iostream>

namespace nixlbench {

bool
validateRawConfigForRun(xferBenchConfig &config) {
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

} // namespace nixlbench
