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

#ifndef __UTILS_H
#define __UTILS_H

#include "config.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <optional>
#include "runtime/runtime.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(result, message)                                           \
    do {                                                                            \
        if (result != cudaSuccess) {                                                \
            std::cerr << "CUDA: " << message << " (Error code: " << result          \
                      << " - " << cudaGetErrorString(result) << ")" << std::endl;   \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while(0)

#define CHECK_CUDA_DRIVER_ERROR(result, message)                                    \
    do {                                                                            \
        if (result != CUDA_SUCCESS) {                                               \
            const char *error_str;                                                  \
            cuGetErrorString(result, &error_str);                                   \
            std::cerr << "CUDA Driver: " << message << " (Error code: "             \
                      << result << " - " << error_str << ")" << std::endl;          \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while(0)
#endif

// TODO: This is true for CX-7, need support for other CX cards and NVLink
#define MAXBW 50.0 // 400 Gbps or 50 GB/sec
#define LARGE_BLOCK_SIZE (1LL * (1 << 20))
#define LARGE_BLOCK_SIZE_ITER_FACTOR 16

#define XFERBENCH_INITIATOR_BUFFER_ELEMENT 0xbb
#define XFERBENCH_TARGET_BUFFER_ELEMENT 0xaa

// Runtime types
#define XFERBENCH_RT_ETCD "ETCD"

// Backend types
#define XFERBENCH_BACKEND_UCX "UCX"
#define XFERBENCH_BACKEND_UCX_MO "UCX_MO"
#define XFERBENCH_BACKEND_GDS "GDS"
#define XFERBENCH_BACKEND_POSIX "POSIX"

// POSIX API types
#define XFERBENCH_POSIX_API_AIO "AIO"
#define XFERBENCH_POSIX_API_URING "URING"

// Scheme types for transfer patterns
#define XFERBENCH_SCHEME_PAIRWISE     "pairwise"
#define XFERBENCH_SCHEME_ONE_TO_MANY  "onetomany"
#define XFERBENCH_SCHEME_MANY_TO_ONE  "manytoone"
#define XFERBENCH_SCHEME_TP           "tp"

// Operation types
#define XFERBENCH_OP_READ  "READ"
#define XFERBENCH_OP_WRITE "WRITE"

// Mode types
#define XFERBENCH_MODE_SG  "SG"
#define XFERBENCH_MODE_MG  "MG"

// Segment types
#define XFERBENCH_SEG_TYPE_DRAM "DRAM"
#define XFERBENCH_SEG_TYPE_VRAM "VRAM"

// Worker types
#define XFERBENCH_WORKER_NIXL     "nixl"
#define XFERBENCH_WORKER_NVSHMEM  "nvshmem"

#define IS_PAIRWISE_AND_SG() (XFERBENCH_SCHEME_PAIRWISE == xferBenchConfig::scheme && \
                              XFERBENCH_MODE_SG == xferBenchConfig::mode)
#define IS_PAIRWISE_AND_MG() (XFERBENCH_SCHEME_PAIRWISE == xferBenchConfig::scheme && \
                              XFERBENCH_MODE_MG == xferBenchConfig::mode)
class xferBenchConfig {
    public:
        static std::string runtime_type;
        static std::string worker_type;
        static std::string backend;
        static std::string initiator_seg_type;
        static std::string target_seg_type;
        static std::string scheme;
        static std::string mode;
        static std::string op_type;
        static bool check_consistency;
        static size_t total_buffer_size;
        static int num_initiator_dev;
        static int num_target_dev;
        static size_t start_block_size;
        static size_t max_block_size;
        static size_t start_batch_size;
        static size_t max_batch_size;
        static int num_iter;
        static int warmup_iter;
        static int num_threads;
        static bool enable_pt;
        static std::string device_list;
        static std::string etcd_endpoints;
        static std::string gds_filepath;
        static bool enable_vmm;
        static int num_files;
        static std::string posix_api_type;
        static std::string posix_filepath;
        static bool storage_enable_direct;
        static int gds_batch_pool_size;
        static int gds_batch_limit;

        static int loadFromFlags();
        static void printConfig();
        static std::vector<std::string> parseDeviceList();
};

// Generic IOV descriptor class independent of NIXL
class xferBenchIOV {
public:
    uintptr_t addr;
    size_t len;
    int devId;
    size_t padded_size;
    unsigned long long handle;

    xferBenchIOV(uintptr_t a, size_t l, int d) :
        addr(a), len(l), devId(d), padded_size(len), handle(0) {}

    xferBenchIOV(uintptr_t a, size_t l, int d, size_t p, unsigned long long h) :
        addr(a), len(l), devId(d), padded_size(p), handle(h) {}
};

class xferBenchUtils {
    private:
        static xferBenchRT *rt;
        static std::string dev_to_use;
    public:
        static void setRT(xferBenchRT *rt);
        static void setDevToUse(std::string dev);
        static std::string getDevToUse();

        static void checkConsistency(std::vector<std::vector<xferBenchIOV>> &desc_lists);
        static void printStatsHeader();
        static void printStats(bool is_target, size_t block_size, size_t batch_size,
			                   double total_duration);
};

#endif // __UTILS_H
