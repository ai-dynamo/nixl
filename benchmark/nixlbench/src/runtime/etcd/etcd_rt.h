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

#ifndef _ETCD_RT_H
#define _ETCD_RT_H

#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <map>
#include <vector>
#include <thread>
#include "runtime/runtime.h"

// Forward declaration for etcd client
namespace etcd {
    class Client;
}

enum xferBenchEtcdMsgType {
    XFER_BENCH_ETCD_MSG_TYPE_INT = 1,
    XFER_BENCH_ETCD_MSG_TYPE_CHAR = 2
};

/**
 * ETCD-based runtime for XFER benchmark coordination
 * Provides process coordination and data exchange using ETCD
 */
class xferBenchEtcdRT: public xferBenchRT {
private:
    // ETCD connection settings
    std::string etcd_endpoints;
    std::string namespace_prefix;
    std::unique_ptr<etcd::Client> client;

    int my_rank; // Rank information
    int global_size;
    uint64_t barrier_gen;
    int *terminate;

    bool error() const { return terminate != nullptr && *terminate; };
    bool should_retry(int value, int max = 60) const {
	    return !error() && value < max;
    }

    std::string makeTypedKey(const std::string& operation, int src, int dst,
                             xferBenchEtcdMsgType type = XFER_BENCH_ETCD_MSG_TYPE_INT);

    std::string makeKey(std::string name, int rank = -1) const {
        std::string suffix;

        if (rank > -1) {
            suffix = "/" + std::to_string(rank);
        }

        return namespace_prefix + name + suffix;
    }

public:
    xferBenchEtcdRT(const std::string& etcd_endpoints, const int size,
                    int *terminate = nullptr);
    ~xferBenchEtcdRT();

    int getRank() const;
    int getSize() const;
    // Communication methods
    int sendInt(int *buffer, int dest_rank) override;
    int recvInt(int *buffer, int src_rank) override;
    int broadcastInt(int *buffer, size_t count, int root_rank) override;
    int sendChar(char *buffer, size_t count, int dest_rank) override;
    int recvChar(char *buffer, size_t count, int src_rank) override;

    int reduceSumDouble(double *local_value, double *global_value, int dest_rank) override;

    // Barrier synchronization
    int barrier(const std::string& barrier_id) override;
};

#endif // _ETCD_RT_H
