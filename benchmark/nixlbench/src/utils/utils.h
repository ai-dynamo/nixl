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

#ifndef __UTILS_H
#define __UTILS_H

#include "benchmark_config.h"
#include "config.h"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <optional>
#include <toml++/toml.hpp>
#include <utils/common/nixl_time.h>
#include "runtime/runtime.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(result, message)                                           \
    do {                                                                            \
        if (result != cudaSuccess) {                                                \
            std::cerr << "CUDA: " << message << " (Error code: " << result << " - " \
                      << cudaGetErrorString(result) << ")" << std::endl;            \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

#define CHECK_CUDA_DRIVER_ERROR(result, message)                                           \
    do {                                                                                   \
        if (result != CUDA_SUCCESS) {                                                      \
            const char *error_str;                                                         \
            cuGetErrorString(result, &error_str);                                          \
            std::cerr << "CUDA Driver: " << message << " (Error code: " << result << " - " \
                      << error_str << ")" << std::endl;                                    \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)
#endif

// TODO: This is true for CX-7, need support for other CX cards and NVLink
#define MAXBW 50.0 // 400 Gbps or 50 GB/sec
#define LARGE_BLOCK_SIZE (1LL * (1 << 20))

#define XFERBENCH_INITIATOR_BUFFER_ELEMENT 0xbb
#define XFERBENCH_TARGET_BUFFER_ELEMENT 0xaa

// Runtime types
#define XFERBENCH_RT_ETCD "ETCD"

// Backend types
#define XFERBENCH_BACKEND_UCX "UCX"
#define XFERBENCH_BACKEND_LIBFABRIC "LIBFABRIC"
#define XFERBENCH_BACKEND_GDS "GDS"
#define XFERBENCH_BACKEND_GDS_MT "GDS_MT"
#define XFERBENCH_BACKEND_POSIX "POSIX"
#define XFERBENCH_BACKEND_GPUNETIO "GPUNETIO"
#define XFERBENCH_BACKEND_MOONCAKE "Mooncake"
#define XFERBENCH_BACKEND_HF3FS "HF3FS"
#define XFERBENCH_BACKEND_OBJ "OBJ"
#define XFERBENCH_BACKEND_GUSLI "GUSLI"
#define XFERBENCH_BACKEND_UCCL "UCCL"
#define XFERBENCH_BACKEND_AZURE_BLOB "AZURE_BLOB"

// POSIX API types
#define XFERBENCH_POSIX_API_AIO "AIO"
#define XFERBENCH_POSIX_API_URING "URING"
#define XFERBENCH_POSIX_API_POSIXAIO "POSIXAIO"

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
#define XFERBENCH_SCHEME_PAIRWISE "pairwise"
#define XFERBENCH_SCHEME_ONE_TO_MANY "onetomany"
#define XFERBENCH_SCHEME_MANY_TO_ONE "manytoone"
#define XFERBENCH_SCHEME_TP "tp"

// Operation types
#define XFERBENCH_OP_READ "READ"
#define XFERBENCH_OP_WRITE "WRITE"

// Mode types
#define XFERBENCH_MODE_SG "SG"
#define XFERBENCH_MODE_MG "MG"

// Segment types
#define XFERBENCH_SEG_TYPE_DRAM "DRAM"
#define XFERBENCH_SEG_TYPE_VRAM "VRAM"
#define XFERBENCH_SEG_TYPE_BLK "BLK"

// Worker types
#define XFERBENCH_WORKER_NIXL "nixl"
#define XFERBENCH_WORKER_NVSHMEM "nvshmem"

#define IS_PAIRWISE_AND_SG(config) \
    (XFERBENCH_SCHEME_PAIRWISE == (config).scheme && XFERBENCH_MODE_SG == (config).mode)
#define IS_PAIRWISE_AND_MG(config) \
    (XFERBENCH_SCHEME_PAIRWISE == (config).scheme && XFERBENCH_MODE_MG == (config).mode)

class xferBenchConfig {
public:
    xferBenchConfig();

    std::string runtime_type;
    std::string worker_type;
    std::string backend;
    std::string initiator_seg_type;
    std::string target_seg_type;
    std::string scheme;
    std::string mode;
    std::string op_type;
    bool check_consistency;
    size_t total_buffer_size;
    bool recreate_xfer;
    int num_initiator_dev;
    int num_target_dev;
    size_t start_block_size;
    size_t max_block_size;
    size_t start_batch_size;
    size_t max_batch_size;
    int num_iter;
    int large_blk_iter_ftr;
    int warmup_iter;
    int num_threads;
    bool enable_pt;
    size_t progress_threads;
    std::string device_list;
    std::string etcd_endpoints;
    std::string benchmark_group;
    std::string filepath;
    std::string filenames;
    bool enable_vmm;
    int num_files;
    std::string posix_api_type;
    int posix_ios_pool_size;
    int posix_kernel_queue_size;
    bool storage_enable_direct;
    int gds_batch_pool_size;
    int gds_batch_limit;
    int gds_mt_num_threads;
    std::string gpunetio_device_list;
    std::string gpunetio_oob_list;
    long page_size;
    std::string obj_access_key;
    std::string obj_secret_key;
    std::string obj_session_token;
    std::string obj_bucket_name;
    std::string obj_scheme;
    std::string obj_region;
    bool obj_use_virtual_addressing;
    std::string obj_endpoint_override;
    std::string obj_req_checksum;
    std::string obj_ca_bundle;
    size_t obj_crt_min_limit;
    bool obj_accelerated_enable;
    std::string obj_accelerated_type;
    std::string azure_blob_account_url;
    std::string azure_blob_container_name;
    std::string azure_blob_connection_string;
    int hf3fs_iopool_size;
    std::string gusli_client_name;
    int gusli_max_simultaneous_requests;
    std::string gusli_config_file;
    std::string gusli_device_byte_offsets;
    std::string gusli_device_security;
    bool cli_help_requested;

    int
    parseConfig(int argc, char *argv[]);
    bool
    cliHelpRequested() const;
    void
    printConfig() const;
    void
    printOption(const std::string &desc, const std::string &value) const;
    void
    printSeparator(const char sep = '-') const;
    std::vector<std::string>
    parseDeviceList() const;
    bool
    isStorageBackend() const;
    bool
    isObjStorageBackend() const;

protected:
    int
    loadParams(void);
};

// Shared GUSLI device config used by utils and nixl_worker
struct GusliDeviceConfig {
    int device_id;
    char device_type; // 'F' for file, 'K' for kernel device, 'N' for networked server
    std::string device_path;
    std::string security_flags;
    size_t dev_offset;
};

// Parser for GUSLI device list: "id:type:path,id:type:path,..." and byte-based device offset list
// security_list: comma-separated security flags; num_devices: expected device count (validation)
std::vector<GusliDeviceConfig>
parseGusliDeviceList(const std::string &device_list,
                     const std::string &security_list,
                     const std::string &dev_offset_list,
                     int num_devices);

// Timer class for measuring durations at high resolution
class xferBenchTimer {
public:
    xferBenchTimer();

    // Return the elapsed time in microseconds
    nixlTime::us_t
    lap();

private:
    nixlTime::us_t start_;
};

// Stats class for measuring arbitrary numeric metrics with multiple samples
class xferMetricStats {
public:
    double
    min() const;
    double
    max() const;
    double
    avg() const;
    double
    p90();
    double
    p95();
    double
    p99();

    void
    add(double value);
    void
    add(const xferMetricStats &other);
    void
    reserve(size_t n);
    void
    clear();

private:
    std::vector<double> samples;
};

// Stats class for measuring benchmark metrics
struct xferBenchStats {
    xferMetricStats total_duration;
    xferMetricStats prepare_duration;
    xferMetricStats post_duration;
    xferMetricStats transfer_duration;

    void
    clear();
    void
    add(const xferBenchStats &other);
    void
    reserve(size_t n);
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

    xferBenchIOV(uintptr_t a, size_t l, int d)
        : addr(a),
          len(l),
          devId(d),
          padded_size(len),
          handle(0) {}

    xferBenchIOV(uintptr_t a, size_t l, int d, size_t p, unsigned long long h)
        : addr(a),
          len(l),
          devId(d),
          padded_size(p),
          handle(h) {}

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
    static int
    createFile(size_t buffer_size, const std::string &filename);
    static void
    cleanupFile(const int fd, const std::string &filename);
    static bool
    putObjAzure(const xferBenchConfig &config, size_t buffer_size, const std::string &name);
    static bool
    getObjAzure(const xferBenchConfig &config, const std::string &name);
    static bool
    rmObjAzure(const xferBenchConfig &config, const std::string &name);
    static std::string
    buildCommonAzCliBlobParams(const xferBenchConfig &config, const std::string &blob_name);

public:
    static void
    setRT(xferBenchRT *rt);
    static void
    setDevToUse(std::string dev);
    static std::string
    getDevToUse();
    static std::string
    buildAwsCredentials(const xferBenchConfig &config);
    static bool
    putObj(const xferBenchConfig &config, size_t buffer_size, const std::string &name);
    static bool
    getObj(const xferBenchConfig &config, const std::string &name);
    static bool
    rmObj(const xferBenchConfig &config, const std::string &name);
    static bool
    putObjS3(const xferBenchConfig &config, size_t buffer_size, const std::string &name);
    static bool
    getObjS3(const xferBenchConfig &config, const std::string &name);
    static bool
    rmObjS3(const xferBenchConfig &config, const std::string &name);

    static bool
    checkConsistency(const xferBenchConfig &config,
                     std::vector<std::vector<xferBenchIOV>> &desc_lists);
    static bool
    validateTransfer(const xferBenchConfig &config,
                     bool is_initiator,
                     std::vector<std::vector<xferBenchIOV>> &local_lists,
                     std::vector<std::vector<xferBenchIOV>> &remote_lists);
    static void
    printStatsHeader(const xferBenchConfig &config);

    static void
    printStats(const nixlbench::benchmarkConfig &config,
               bool is_target,
               size_t block_size,
               size_t batch_size,
               xferBenchStats stats);

    static void
    printStats(const xferBenchConfig &config,
               bool is_target,
               size_t block_size,
               size_t batch_size,
               xferBenchStats stats);
};

#endif // __UTILS_H
