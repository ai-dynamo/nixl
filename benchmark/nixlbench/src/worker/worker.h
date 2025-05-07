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

#ifndef __WORKER_H
#define __WORKER_H

#include "runtime/runtime.h"
#include "utils/utils.h"
#include <string>
#include <vector>
#include <variant>
#include <memory>

class xferBenchWorker {
    protected:
        std::string name;
        xferBenchRT *rt;
        static int terminate;

    public:
        xferBenchWorker(int *argc, char ***argv);
        virtual ~xferBenchWorker();

        std::string getName() const;
        bool isMasterRank();
        bool isInitiator();
        bool isTarget();
        int synchronize();
        bool signaled() const { return terminate != 0; }
        static void signalHandler(int signal);

        // Memory management
        virtual std::vector<std::vector<xferBenchIOV>> allocateMemory(int num_threads) = 0;
        virtual void deallocateMemory(std::vector<std::vector<xferBenchIOV>> &iov_lists) = 0;

        // Communication and synchronization
        virtual int exchangeMetadata() = 0;
        virtual std::vector<std::vector<xferBenchIOV>> exchangeIOV(const std::vector<std::vector<xferBenchIOV>>
                                                                   &local_iov_lists) = 0;
        virtual void poll(size_t block_size) = 0;
	virtual int synchronizeStart() = 0;

        // Data operations
        virtual std::variant<double, int> transfer(size_t block_size,
                                                   const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                                                   const std::vector<std::vector<xferBenchIOV>> &remote_iov_lists) = 0;
};

#endif // __WORKER_H
