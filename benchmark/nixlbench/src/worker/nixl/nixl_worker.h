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

struct xferFileState {
    int fd;
    uint64_t file_size;
    uint64_t offset;
};

class xferBenchNixlWorker: public xferBenchWorker {
    private:
        nixlAgent* agent;
        nixlBackendH* backend_engine;
        nixl_mem_t seg_type;
        std::vector<xferFileState> remote_fds;
        std::vector<std::vector<xferBenchIOV>> remote_iovs;

        // GDAKI signal buffer support
        std::vector<xferBenchIOV> signal_buffers;
        bool is_gdaki_enabled;

    public:
        xferBenchNixlWorker(int *argc, char ***argv, std::vector<std::string> devices);
        ~xferBenchNixlWorker();  // Custom destructor to clean up resources

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
        void
        device_poll(size_t block_size, unsigned int skip, unsigned int num_iter);
        int
        synchronizeStart();
        int
        send_wireup_notification(const std::string &remote_name);
        int
        wait_for_ack(const std::string &remote_name);
        void
        send_metadata(std::string local_md, int rank);
        int
        recv_and_load_metadata(int rank);

        // Data operations
        std::variant<xferBenchStats, int>
        transfer(size_t block_size,
                 const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                 const std::vector<std::vector<xferBenchIOV>> &remote_iov_lists) override;

    private:
        std::optional<xferBenchIOV>
        initBasicDescDram(size_t buffer_size, int mem_dev_id);
        void cleanupBasicDescDram(xferBenchIOV &basic_desc);
#if HAVE_CUDA
        std::optional<xferBenchIOV> initBasicDescVram(size_t buffer_size, int mem_dev_id);
        void
        cleanupBasicDescVram(xferBenchIOV &basic_desc);
#endif
        // GDAKI signal buffer management
        std::optional<xferBenchIOV>
        initSignalBuffer(int mem_dev_id, size_t sigsize);
        void
        cleanupSignalBuffer(std::optional<xferBenchIOV> &);

        // GDAKI wireup notification
        int
        sendWireupMessage();

        // GDAKI bidirectional metadata exchange
        int
        exchangeMetadataBidirectional();
        std::optional<xferBenchIOV>
        initBasicDescFile(size_t buffer_size, xferFileState &fstate, int mem_dev_id);
        void cleanupBasicDescFile(xferBenchIOV &basic_desc);
        std::optional<xferBenchIOV>
        initBasicDescObj(size_t buffer_size, int mem_dev_id, std::string name);
        void
        cleanupBasicDescObj(xferBenchIOV &basic_desc);
};

#endif // __NIXL_WORKER_H
