/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_WORKER_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_WORKER_H

#include "config.h"
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <optional>
#include <memory>
#include <unistd.h>
#include <functional>
#include <nixl.h>
#include <nixl_types.h>
#include "utils/utils.h"
#include "worker/worker.h"
#include <random>
#include "worker/nixl/nixl_mem_region.h"

/**
 * @brief Scenario-owned resource lifecycle for one transfer-request slot.
 *
 * prepare() may acquire resources and replace the slot descriptors before the common worker
 * creates a transfer request. release() runs only after the request and any common registration
 * have been released. Both methods must support cleanup after partial preparation.
 */
class xferBenchNixlIterationLifecycle {
public:
    virtual ~xferBenchNixlIterationLifecycle() = default;

    /** @brief Acquire resources and prepare descriptors for one request. */
    virtual nixl_status_t
    prepare(std::vector<xferBenchIOV> &local_iovs, std::vector<xferBenchIOV> &remote_iovs) = 0;

    /** @brief Observe or validate a completed request before its resources are released. */
    virtual nixl_status_t
    complete(const std::vector<xferBenchIOV> &local_iovs,
             const std::vector<xferBenchIOV> &remote_iovs) = 0;

    /** @brief Release resources acquired by prepare(). */
    virtual nixl_status_t
    release() = 0;
};

using iteration_lifecycle_factory_t =
    std::function<std::unique_ptr<xferBenchNixlIterationLifecycle>(size_t, size_t)>;

// Use shared GusliDeviceConfig and parseGusliDeviceList declared in utils.h

class xferBenchNixlWorker: public xferBenchWorker {
    private:
        nixlAgent* agent;
        nixlBackendH* backend_engine;
        nixl_mem_t seg_type;
        std::vector<xferFileState> remote_fds;
        std::vector<NixlMemRegion> remote_regs_;
        std::vector<NixlMemRegion> local_regs_;
        std::vector<GusliDeviceConfig> gusli_devices;
        std::string remote_agent_name;
        std::optional<xferBenchIOV> completion_counter_iov;

    public:
        explicit xferBenchNixlWorker(const std::vector<std::string> &devices);
        ~xferBenchNixlWorker() override;

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
        synchronizeStart() override;

        // Data operations
        std::variant<xferBenchStats, int>
        transfer(size_t block_size,
                 const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                 const std::vector<std::vector<xferBenchIOV>> &remote_iov_lists) override;

    protected:
        /** @brief Return the configured local memory type. */
        nixl_mem_t
        localMemoryType() const;
        /** @brief Allocate one unregistered local descriptor. */
        std::optional<xferBenchIOV>
        allocateLocalIov(size_t buffer_size, int mem_dev_id);
        /** @brief Fill a local descriptor with one byte value. */
        void
        initializeLocalIov(xferBenchIOV &iov, uint8_t value);
        /** @brief Retain ownership of an open remote file descriptor. */
        void
        retainRemoteFile(int fd, size_t file_size);
        /** @brief Return a retained remote file descriptor by index. */
        std::optional<int>
        remoteFileDescriptor(size_t index) const;
        /** @brief Register local descriptors for the worker lifetime. */
        bool
        registerLocalIovs(std::vector<xferBenchIOV> iovs);
        /** @brief Register remote descriptors for the worker lifetime. */
        bool
        registerRemoteIovs(nixl_mem_t memory_type, std::vector<xferBenchIOV> iovs);
        /** @brief Run transfers with scenario-owned per-request resource lifecycles. */
        std::variant<xferBenchStats, int>
        transferWithLifecycle(size_t block_size,
                              const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                              const std::vector<std::vector<xferBenchIOV>> &remote_iov_lists,
                              const iteration_lifecycle_factory_t &lifecycle_factory);

    private:
        std::optional<xferBenchIOV>
        initBasicDescDram(size_t buffer_size, int mem_dev_id);
        std::optional<xferBenchIOV>
        initBasicDescVram(size_t buffer_size, int mem_dev_id);
        std::optional<xferBenchIOV>
        initBasicDescFile(size_t buffer_size, xferFileState &fstate, int mem_dev_id);
        std::optional<xferBenchIOV>
        initBasicDescObj(size_t buffer_size, int mem_dev_id, std::string name);
        std::optional<xferBenchIOV>
        initBasicDescBlk(size_t buffer_size, int mem_dev_id, size_t dev_offset);
        bool
        ensureFileHasConsistencyData(const GusliDeviceConfig &device, size_t size);
        uint64_t
        getFileOffset(size_t current_offset, size_t max_offset_in_blocks, size_t block_size);
        void
        releaseMemView(nixlMemViewH &mvh);
        nixlMemViewH
        prepareGPULocalView(const std::vector<std::vector<xferBenchIOV>> &local_iov_lists);
        nixlMemViewH
        prepareGPURemoteView(const std::vector<std::vector<xferBenchIOV>> &remote_iov_lists);
        std::optional<xferBenchIOV>
        initCompletionCounterVram();
        bool
        waitForDeviceCompletionCounter(const xferBenchIOV &counter_iov,
                                       uint64_t expected_value,
                                       const char *phase,
                                       const std::function<void()> &checkLiveness);

        std::mt19937_64 default_rng_;
};

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_WORKER_H
