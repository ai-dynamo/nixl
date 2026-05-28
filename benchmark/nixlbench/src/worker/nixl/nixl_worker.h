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
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <optional>
#include <memory>
#include <unistd.h>
#include <nixl.h>
#include "utils/utils.h"
#include "worker/worker.h"

struct xferFileState {
    int fd = -1;
    uint64_t file_size = 0;
    uint64_t offset = 0;

    xferFileState() = default;

    xferFileState(int fd, uint64_t file_size, uint64_t offset)
        : fd(fd),
          file_size(file_size),
          offset(offset) {}

    ~xferFileState() {
        if (fd >= 0) {
            ::close(fd);
        }
    }

    xferFileState(xferFileState &&o) noexcept
        : fd(std::exchange(o.fd, -1)),
          file_size(o.file_size),
          offset(o.offset) {}

    xferFileState &
    operator=(xferFileState &&o) noexcept {
        if (this != &o) {
            if (fd >= 0) {
                ::close(fd);
            }
            fd = std::exchange(o.fd, -1);
            file_size = o.file_size;
            offset = o.offset;
        }
        return *this;
    }

    xferFileState(const xferFileState &) = delete;
    xferFileState &
    operator=(const xferFileState &) = delete;
};

class NixlMemRegion {
    nixlAgent *agent_ = nullptr;
    nixlBackendH *backend_ = nullptr;
    nixl_mem_t seg_type_ = DRAM_SEG;
    std::vector<xferBenchIOV> iovs_;
    std::function<void(xferBenchIOV &)> cleanup_;
    nixl_opt_args_t cached_opt_args_;

public:
    NixlMemRegion() = default;
    NixlMemRegion(nixlAgent *agent,
                  nixlBackendH *backend,
                  nixl_mem_t seg_type,
                  std::vector<xferBenchIOV> iovs,
                  std::function<void(xferBenchIOV &)> cleanup = nullptr);
    ~NixlMemRegion();
    NixlMemRegion(NixlMemRegion &&o) noexcept;
    NixlMemRegion &
    operator=(NixlMemRegion &&o) noexcept;
    NixlMemRegion(const NixlMemRegion &) = delete;
    NixlMemRegion &
    operator=(const NixlMemRegion &) = delete;

    const std::vector<xferBenchIOV> &
    iovs() const {
        return iovs_;
    }

    std::vector<xferBenchIOV> &
    iovs() {
        return iovs_;
    }

    void
    release();
};

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
        synchronizeStart();

        // Data operations
        std::variant<xferBenchStats, int>
        transfer(size_t block_size,
                 const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                 const std::vector<std::vector<xferBenchIOV>> &remote_iov_lists) override;

    private:
        std::optional<xferBenchIOV>
        initBasicDescDram(size_t buffer_size, int mem_dev_id);
        void
        cleanupBasicDescDram(xferBenchIOV &basic_desc);
        std::optional<xferBenchIOV> initBasicDescVram(size_t buffer_size, int mem_dev_id);
        void
        cleanupBasicDescVram(xferBenchIOV &basic_desc);
        std::optional<xferBenchIOV>
        initBasicDescFile(size_t buffer_size, xferFileState &fstate, int mem_dev_id);
        void cleanupBasicDescFile(xferBenchIOV &basic_desc);
        std::optional<xferBenchIOV>
        initBasicDescObj(size_t buffer_size, int mem_dev_id, std::string name);
        void
        cleanupBasicDescObj(xferBenchIOV &basic_desc);
        std::optional<xferBenchIOV>
        initBasicDescBlk(size_t buffer_size, int mem_dev_id, size_t dev_offset);
        void
        cleanupBasicDescBlk(xferBenchIOV &basic_desc);
        bool
        ensureFileHasConsistencyData(const GusliDeviceConfig &device, size_t size);
};

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_WORKER_H
