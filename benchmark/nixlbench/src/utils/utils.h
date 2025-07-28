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
#define MIN_WARMUP_ITERS 8

#define XFERBENCH_INITIATOR_BUFFER_ELEMENT 0xbb
#define XFERBENCH_TARGET_BUFFER_ELEMENT 0xaa

// Runtime types
#define XFERBENCH_RT_ETCD "ETCD"

// Backend types
#define XFERBENCH_BACKEND_UCX "UCX"
#define XFERBENCH_BACKEND_UCX_MO "UCX_MO"
#define XFERBENCH_BACKEND_GDS "GDS"
#define XFERBENCH_BACKEND_POSIX "POSIX"
#define XFERBENCH_BACKEND_GPUNETIO "GPUNETIO"
#define XFERBENCH_BACKEND_MOONCAKE "Mooncake"
#define XFERBENCH_BACKEND_HF3FS "HF3FS"
#define XFERBENCH_BACKEND_OBJ "OBJ"

// POSIX API types
#define XFERBENCH_POSIX_API_AIO "AIO"
#define XFERBENCH_POSIX_API_URING "URING"

// OBJ S3 scheme types
#define XFERBENCH_OBJ_SCHEME_HTTP "http"
#define XFERBENCH_OBJ_SCHEME_HTTPS "https"

// OBJ S3 region types
#define XFERBENCH_OBJ_REGION_EU_CENTRAL_1 "eu-central-1"

// OBJ S3 bucket names
#define XFERBENCH_OBJ_BUCKET_NAME_DEFAULT ""

// OBJ S3 required checksum types
#define XFERBENCH_OBJ_REQ_CHECKSUM_SUPPORTED "supported"
#define XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED "required"

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
        static int large_blk_iter_ftr;
        static int warmup_iter;
        static int num_threads;
        static bool enable_pt;
        static std::string device_list;
        static std::string etcd_endpoints;
        static std::string benchmark_group;
        static std::string filepath;
        static bool enable_vmm;
        static int num_files;
        static std::string posix_api_type;
        static bool storage_enable_direct;
        static int gds_batch_pool_size;
        static int gds_batch_limit;
        static std::string gpunetio_device_list;
        static long page_size;
        static std::string obj_access_key;
        static std::string obj_secret_key;
        static std::string obj_session_token;
        static std::string obj_bucket_name;
        static std::string obj_scheme;
        static std::string obj_region;
        static bool obj_use_virtual_addressing;
        static std::string obj_endpoint_override;
        static std::string obj_req_checksum;

        static int loadFromFlags();
        static void printConfig();
        static void
        printOption (const std::string &desc, const std::string &value);
        static std::vector<std::string> parseDeviceList();
        static bool
        isStorageBackend();
};

// Generic IOV descriptor class independent of NIXL
class xferBenchIOV {
public:
    uintptr_t addr;
    size_t len;
    int devId;
    size_t padded_size;
    unsigned long long handle;
    std::string metaInfo;

    xferBenchIOV(uintptr_t a, size_t l, int d) :
        addr(a), len(l), devId(d), padded_size(len), handle(0) {}

    xferBenchIOV(uintptr_t a, size_t l, int d, size_t p, unsigned long long h) :
        addr(a), len(l), devId(d), padded_size(p), handle(h) {}

    xferBenchIOV(uintptr_t a, size_t l, int d, std::string m)
        : addr(a),
          len(l),
          devId(d),
          padded_size(len),
          handle(0),
          metaInfo(m) {}
};

class xferBenchUtils {
    private:
        static xferBenchRT *rt;
        static std::string dev_to_use;
    public:
        static void setRT(xferBenchRT *rt);
        static void setDevToUse(std::string dev);
        static std::string getDevToUse();
        static std::string
        buildAwsCredentials();
        static bool
        putObjS3(size_t buffer_size, const std::string &name);
        static bool
        getObjS3(const std::string &name);
        static bool
        rmObjS3(const std::string &name);

        static void checkConsistency(std::vector<std::vector<xferBenchIOV>> &desc_lists);
        static void printStatsHeader();
        static void printStats(bool is_target, size_t block_size, size_t batch_size,
			                   double total_duration);
};

#endif // __UTILS_H
