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

#ifndef __NVSHMEM_WORKER_H
#define __NVSHMEM_WORKER_H

#include "config.h"
#include "worker/worker.h"

#if HAVE_NVSHMEM && HAVE_CUDA
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>

class xferBenchNvshmemWorker: public xferBenchWorker {
    private:
        // Additional members used in implementation
        int rank;
        int size;

        // CUDA stream
        cudaStream_t stream;
        nvshmemx_uniqueid_t group_id;
        int group_id_initialized = 0;

    public:
        xferBenchNvshmemWorker(int *argc, char ***argv);
        ~xferBenchNvshmemWorker() override;

        // Memory management
        std::vector<std::vector<xferBenchIOV>> allocateMemory(int num_threads) override;
        void deallocateMemory(std::vector<std::vector<xferBenchIOV>> &iov_lists) override;

        // Communication and synchronization
        int exchangeMetadata() override;
        std::vector<std::vector<xferBenchIOV>>
        exchangeIOV(const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                    size_t block_size) override;
        void
        poll(size_t block_size) override;
        int
        synchronizeStart();

        // Data operations
        std::variant<xferBenchStats, int>
        transfer(size_t block_size,
                 const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                 const std::vector<std::vector<xferBenchIOV>> &remote_iov_lists) override;

    private:
        std::optional<xferBenchIOV>
        initBasicDescNvshmem(size_t buffer_size, int mem_dev_id);
        void
        cleanupBasicDescNvshmem(xferBenchIOV &iov);
};
#endif

#endif // __NVSHMEM_WORKER_H
