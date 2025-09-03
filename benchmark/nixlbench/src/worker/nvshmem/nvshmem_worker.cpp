/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "worker/nvshmem/nvshmem_worker.h"
#include "runtime/runtime.h"
#include "utils/utils.h"
#include <iostream>
#include <cstring>

#if HAVE_NVSHMEM && HAVE_CUDA
#define CHECK_NVSHMEM_ERROR(result, message)                                    \
    do {                                                                        \
        if (0 != result) {                                                      \
            std::cerr << "NVSHMEM: " << message << " (Error code: " << result   \
                      << ")" << std::endl;                                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

xferBenchNvshmemWorker::xferBenchNvshmemWorker(int *argc, char ***argv): xferBenchWorker(argc, argv) {
    // Initialize NVSHMEM
    if (XFERBENCH_RT_ETCD == xferBenchConfig::runtime_type) {
	    rank = rt->getRank();
	    size = rt->getSize();

        return;        //NVSHMEM not yet initialized
    }

    std::cout << "Runtime " << xferBenchConfig::runtime_type
		      << " not supported for NVSHMEM worker" << std::endl;
    exit(EXIT_FAILURE);
}

xferBenchNvshmemWorker::~xferBenchNvshmemWorker() {
    // Finalize NVSHMEM
    nvshmem_finalize();
}

std::optional<xferBenchIOV> xferBenchNvshmemWorker::initBasicDescNvshmem(size_t buffer_size, int mem_dev_id) {
    void *addr;

    addr = nvshmem_malloc(buffer_size);
    if (!addr) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of NVSHMEM memory" << std::endl;
        return std::nullopt;
    }

    if (isInitiator()) {
        cudaMemset(addr, XFERBENCH_INITIATOR_BUFFER_ELEMENT, buffer_size);
    } else if (isTarget()) {
        cudaMemset(addr, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);
    }

    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, mem_dev_id);
}

void xferBenchNvshmemWorker::cleanupBasicDescNvshmem(xferBenchIOV &iov) {
    nvshmem_free((void *)iov.addr);
}

std::vector<std::vector<xferBenchIOV>> xferBenchNvshmemWorker::allocateMemory(int num_threads) {
    std::vector<std::vector<xferBenchIOV>> iov_lists;
    size_t i, buffer_size, num_devices = 0;

    if (1 != num_threads) {
        std::cerr << "NVSHMEM: Only 1 thread is supported for now" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (isInitiator()) {
        num_devices = xferBenchConfig::num_initiator_dev;
    } else if (isTarget()) {
        num_devices = xferBenchConfig::num_target_dev;
    }
    buffer_size = xferBenchConfig::total_buffer_size / (num_devices * num_threads);

    for (int list_idx = 0; list_idx < num_threads; list_idx++) {
        std::vector<xferBenchIOV> iov_list;
        for (i = 0; i < num_devices; i++) {
            std::optional<xferBenchIOV> basic_desc;
            basic_desc = initBasicDescNvshmem(buffer_size, i);
            if (basic_desc) {
                iov_list.push_back(basic_desc.value());
            }
        }
        iov_lists.push_back(iov_list);
    }
    return iov_lists;
}

void xferBenchNvshmemWorker::deallocateMemory(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    nvshmem_barrier_all();
    for (auto &iov_list: iov_lists) {
        for (auto &iov: iov_list) {
            cleanupBasicDescNvshmem(iov);
        }
    }
}

int xferBenchNvshmemWorker::exchangeMetadata() {
    // No metadata exchange needed for NVSHMEM
    return 0;
}

std::vector<std::vector<xferBenchIOV>>
xferBenchNvshmemWorker::exchangeIOV(const std::vector<std::vector<xferBenchIOV>> &iov_lists,
                                    size_t block_size) {
    // For NVSHMEM, we don't need to exchange IOV lists
    // This will just return local IOV list
    return iov_lists;
}

// No thread support for NVSHMEM yet
static int
execTransfer(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
             const std::vector<std::vector<xferBenchIOV>> &remote_iovs,
             const int num_iter,
             cudaStream_t stream,
             xferBenchStats &stats) {
    int ret = 0, tid = 0, target_rank;

    target_rank = 1;

    const auto &local_iov = local_iovs[tid];
    const auto &remote_iov = remote_iovs[tid];

    xferBenchTimer total_timer;
    xferBenchTimer timer;

    for (int i = 0; i < num_iter; i++) {
        for (size_t i = 0; i < local_iov.size(); i++) {
            auto &local = local_iov[i];
            auto &remote = remote_iov[i];
            if (XFERBENCH_OP_WRITE == xferBenchConfig::op_type) {
                nvshmemx_putmem_on_stream(
                    (void *)remote.addr, (void *)local.addr, local.len, target_rank, stream);
            } else if (XFERBENCH_OP_READ == xferBenchConfig::op_type) {
                nvshmemx_getmem_on_stream(
                    (void *)remote.addr, (void *)local.addr, local.len, target_rank, stream);
            }
        }
        nvshmemx_quiet_on_stream(stream);
        nixlTime::us_t transfer_duration = timer.lap();
        stats.transfer_duration.add(transfer_duration);
    }

    nixlTime::us_t total_duration = total_timer.lap();
    stats.total_duration.add(total_duration);

    return ret;
}

std::variant<xferBenchStats, int>
xferBenchNvshmemWorker::transfer(size_t block_size,
                                 const std::vector<std::vector<xferBenchIOV>> &local_trans_lists,
                                 const std::vector<std::vector<xferBenchIOV>> &remote_trans_lists) {
    cudaEvent_t start_event, stop_event;
    int num_iter = xferBenchConfig::num_iter / xferBenchConfig::num_threads;
    int skip = xferBenchConfig::warmup_iter / xferBenchConfig::num_threads;
    int ret = 0;
    xferBenchStats stats;

    // Create events to time the transfer
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event), "Failed to create CUDA event");
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event), "Failed to create CUDA event");

    // Here the local_trans_lists is the same as remote_trans_lists
    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= xferBenchConfig::large_blk_iter_ftr;
        num_iter /= xferBenchConfig::large_blk_iter_ftr;
    }

    if (skip > 0) {
        ret = execTransfer(local_trans_lists, remote_trans_lists, skip, stream, stats);
        if (ret < 0) {
            return std::variant<xferBenchStats, int>(ret);
        }
        stats.clear();
    }
    nvshmemx_barrier_all_on_stream(stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream), "Failed to synchronize CUDA stream");

    CHECK_CUDA_ERROR(cudaEventRecord(start_event, stream), "Failed to record CUDA event");

    ret = execTransfer(local_trans_lists, remote_trans_lists, num_iter, stream, stats);

    CHECK_CUDA_ERROR(cudaEventRecord(stop_event, stream), "Failed to record CUDA event");

    nvshmemx_barrier_all_on_stream(stream);
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event), "Failed to synchronize CUDA event");
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream), "Failed to synchronize CUDA stream");

    return ret < 0 ? std::variant<xferBenchStats, int>(ret) :
                     std::variant<xferBenchStats, int>(stats);
}

void
xferBenchNvshmemWorker::poll(size_t block_size) {
    // For NVSHMEM, we don't need to poll
    // The transfer is already complete when we reach this point
    nvshmemx_barrier_all_on_stream(stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream), "Failed to synchronize CUDA stream");

    nvshmemx_barrier_all_on_stream(stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream), "Failed to synchronize CUDA stream");
}

int xferBenchNvshmemWorker::synchronizeStart() {
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    group_id = NVSHMEMX_UNIQUEID_INITIALIZER;

    if (xferBenchConfig::runtime_type == XFERBENCH_RT_ETCD) {
        if (rank == 0 && group_id_initialized == 0) {
            nvshmemx_get_uniqueid(&group_id);
        }

        rt->broadcastInt((int *)&group_id, sizeof(nvshmemx_uniqueid_t), 0);
        group_id_initialized = 1;

        nvshmemx_set_attr_uniqueid_args(rank, size, &group_id, &attr);
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

        // Create a stream
        CHECK_CUDA_ERROR(cudaSetDevice(rank), "Failed to set CUDA device");
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream), "Failed to create CUDA stream");
    }

    // Barrier to ensure all workers have initialized NVSHMEM
    nvshmemx_barrier_all_on_stream(stream);

    return 0;
}

#endif
