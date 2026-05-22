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

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_RUNTIME_RUNTIME_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_RUNTIME_RUNTIME_H

#include <string>

class xferBenchRT {
    private:
        int size;
        int rank;
    protected:
        void setSize(int s) { size = s; };
        void setRank(int r) { rank = r; };
    public:
        virtual ~xferBenchRT() = default;

        int getSize() const;
        int getRank() const;
        virtual int sendInt(int *buffer, int dest_rank) = 0;
        virtual int recvInt(int *buffer, int src_rank) = 0;
        virtual int broadcastInt(int *buffer, size_t count, int root_rank) = 0;
        virtual int sendChar(char *buffer, size_t count, int dest_rank) = 0;
        virtual int recvChar(char *buffer, size_t count, int src_rank) = 0;
        virtual int reduceSumDouble(double *local_value, double *global_value, int dest_rank) = 0;

        // Add a barrier function to synchronize all processes
        virtual int barrier(const std::string& barrier_id) = 0;

        // Check if all peer processes are still alive; returns true by default
        [[nodiscard]] virtual bool
        areAllPeersAlive() {
            return true;
        }

        // Best-effort cleanup of runtime state (e.g. etcd keys) before a
        // forced exit that bypasses normal destructors.
        virtual void
        cleanupForExit() {}
};

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_RUNTIME_RUNTIME_H
