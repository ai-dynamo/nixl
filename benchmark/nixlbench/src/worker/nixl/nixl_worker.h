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

#ifndef __NIXL_WORKER_H
#define __NIXL_WORKER_H

#include "config.h"
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <optional>
#include <memory>
#include <nixl.h>
#include "utils/utils.h"
#include "worker/worker.h"
#include <array>
#include "runtime/etcd/etcd_rt.h"

struct LatencyResult {
    double total_duration;
    double min_latency;
    double median_latency;
    double p95_latency;
    double p99_latency;
    
    LatencyResult(double total, double min, double med, double p95, double p99)
        : total_duration(total), min_latency(min), median_latency(med), 
          p95_latency(p95), p99_latency(p99) {}
};

class xferBenchNixlWorker: public xferBenchWorker {
    private:
        nixlAgent* agent;
        nixlBackendH* backend_engine;
        nixl_mem_t seg_type;
        std::vector<int> remote_fds;
        std::vector<std::vector<xferBenchIOV>> remote_iovs;
        
        static thread_local double min_latency;
        static thread_local double max_latency;
        static thread_local double median_latency;
        static thread_local double p95_latency;
        static thread_local double p99_latency;
        static thread_local double avg_latency;
        static thread_local double total_duration;
        static thread_local size_t num_operations;
    public:
        xferBenchNixlWorker(int *argc, char ***argv, std::vector<std::string> devices);
        ~xferBenchNixlWorker();  // Custom destructor to clean up resources

        // Memory management
        std::vector<std::vector<xferBenchIOV>> allocateMemory(int num_threads) override;
        void deallocateMemory(std::vector<std::vector<xferBenchIOV>> &iov_lists) override;

        // Communication and synchronization
        int exchangeMetadata() override;
        std::vector<std::vector<xferBenchIOV>> exchangeIOV(const std::vector<std::vector<xferBenchIOV>>
                                                           &local_iov_lists) override;
        void poll(size_t block_size) override;
        int synchronizeStart();

        // Data operations
        std::variant<double, int> transfer(size_t block_size,
                                           const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                                           const std::vector<std::vector<xferBenchIOV>> &remote_iov_lists) override;

       
        void getLastLatencyStats(double& min_lat, double& med_lat, double& max_lat, double& p95_lat, 
                                 double& p99_lat, double& avg_lat, double& total_dur, size_t& num_ops);

    private:
        std::optional<xferBenchIOV> initBasicDescDram(size_t buffer_size, int mem_dev_id);
        void cleanupBasicDescDram(xferBenchIOV &basic_desc);
#if HAVE_CUDA
        std::optional<xferBenchIOV> initBasicDescVram(size_t buffer_size, int mem_dev_id);
        void cleanupBasicDescVram(xferBenchIOV &basic_desc);
#endif
        std::optional<xferBenchIOV> initBasicDescFile(size_t buffer_size, int fd, int mem_dev_id);
        std::optional<xferBenchIOV> initDirectIODescFile(size_t buffer_size, int fd, int mem_dev_id);
        void cleanupBasicDescFile(xferBenchIOV &basic_desc);
        bool initializeFileSeg(int num_lists, std::vector<std::vector<xferBenchIOV>> &remote_iovs);
};

#endif // __NIXL_WORKER_H
